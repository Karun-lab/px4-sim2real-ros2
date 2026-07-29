#!/usr/bin/env python3
"""
iris_icm_map_inference_node.py
================================
ROS 2 inference node for the Isaac-Lab-trained ICM + Spatial Memory exploration policy.

This node uses the 2-channel observation (depth + novelty map) that was used during training.
The novelty map is built online from VIO pose estimates, exactly matching the training
environment's visit-count grid.

Subscribes:
    depth topic     (sensor_msgs/Image)           - /m2h/depth/image
    vio pose topic  (geometry_msgs/PoseStamped)   - /rvio2/trajectory

Publishes:
    action topic    (geometry_msgs/Twist)         - /uav/action_cmd
        linear.x  = forward velocity command (normalised [-1, 1])
        angular.z = yaw rate command (normalised [-1, 1])

Pipeline (mirrors iris_icm_map_env.py):
    1. depth -> clamp -> normalise to [0,1]           (channel 0)
    2. VIO pose -> update visit-count grid -> extract local novelty map (channel 1)
    3. stack last T=3 (depth, novelty) frames         -> (T,H,W,2)
    4. run PPO policy, apply EMA smoothing (alpha=0.6)
    5. publish normalised (vx, yaw)
"""

from collections import deque
import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped

try:
    from cv_bridge import CvBridge
    import cv2
except ImportError as e:
    raise ImportError(
        "This node needs cv_bridge and opencv-python. "
        "Install with: sudo apt install ros-<distro>-cv-bridge && pip install opencv-python"
    ) from e


# =============================================================================
# CONSTANTS — MUST match training (iris_icm_map_env.py / iris_icm_map_agent.py)
# =============================================================================

# Depth
CAM_H, CAM_W = 64, 80
CAM_MIN_DEPTH, CAM_MAX_DEPTH = 0.2, 6.0

# History
HIST_LEN = 3
N_CH = 2   # depth + novelty map (2-channel observation)

# Grid / novelty map (must match training)
GRID_CELL_M = 0.25      # metres per cell
GRID_EXTENT_M = 30.0    # grid covers ±15 m from origin
GRID_N = int(GRID_EXTENT_M / GRID_CELL_M)  # 120 cells per side
LOCAL_MAP_PX = 21       # 21×21 cells novelty crop
HALF_CROP = LOCAL_MAP_PX // 2

# Action smoothing (matches training)
ACTION_ALPHA = 0.6


# =============================================================================
# MODEL — same architecture as IrisICMOfficeModel (iris_icm_map_agent.py)
# =============================================================================

class IrisICMMapPolicyNet(nn.Module):
    """2-channel CNN policy: depth + novelty map."""
    
    def __init__(self, t_steps=HIST_LEN, h=CAM_H, w=CAM_W, n_ch=N_CH, action_dim=2):
        super().__init__()
        self.t_steps = t_steps
        self.h, self.w, self.n_ch = h, w, n_ch

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_ch, h, w)
            cnn_out = self.cnn(dummy).shape[1]

        self.net = nn.Sequential(
            nn.Linear(t_steps * cnn_out, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.policy_mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(256, 1)  # unused at inference

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (1, T, H, W, C) in [0,1] -> returns (1, action_dim) mean action."""
        feats = []
        for t in range(self.t_steps):
            frame = obs[:, t].permute(0, 3, 1, 2)  # (N, C, H, W)
            feats.append(self.cnn(frame))
        shared = self.net(torch.cat(feats, dim=1))
        mean = self.policy_mean(shared)
        return mean


def load_policy(checkpoint_path: str, device: torch.device) -> IrisICMMapPolicyNet:
    """Load checkpoint with 2-channel observation support."""
    model = IrisICMMapPolicyNet(n_ch=N_CH).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "policy" in ckpt:
        state_dict = ckpt["policy"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Clean up keys
    cleaned = {}
    model_keys = set(model.state_dict().keys())
    for k, v in state_dict.items():
        if k in model_keys:
            cleaned[k] = v
        else:
            # Try to match by suffix
            match = next((mk for mk in model_keys if k.endswith(mk)), None)
            if match:
                cleaned[match] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load_policy] WARNING missing keys: {missing}")
    if unexpected:
        print(f"[load_policy] WARNING unexpected keys: {unexpected}")

    model.eval()
    return model


# =============================================================================
# NOVELTY MAP BUILDER
# =============================================================================

class NoveltyMapBuilder:
    """
    Builds and maintains the visit-count grid from VIO pose estimates.
    Mirrors the training environment's grid logic.
    """
    
    def __init__(self, grid_n=GRID_N, cell_m=GRID_CELL_M, half_crop=HALF_CROP):
        self.grid_n = grid_n
        self.cell_m = cell_m
        self.half_crop = half_crop
        self.local_px = 2 * half_crop + 1
        
        # Visit count grid (2D array)
        self.visit_count = np.zeros((grid_n, grid_n), dtype=np.float32)
        
        # Previous pose for drift tracking
        self._last_pos = None
        
    def pos_to_cell(self, x: float, y: float) -> tuple:
        """Convert local XY metres to grid indices (row, col)."""
        half = self.grid_n // 2
        col = int(x / self.cell_m + half)
        row = int(-y / self.cell_m + half)
        col = max(0, min(col, self.grid_n - 1))
        row = max(0, min(row, self.grid_n - 1))
        return row, col
    
    def update(self, x: float, y: float) -> np.ndarray:
        """
        Update visit count at current position and return local novelty map.
        Returns (LOCAL_MAP_PX, LOCAL_MAP_PX) novelty map in [0,1].
        """
        row, col = self.pos_to_cell(x, y)
        
        # Increment visit count
        self.visit_count[row, col] += 1.0
        
        # Extract local crop
        half = self.half_crop
        pad = half
        
        # Pad the grid with zeros (unvisited = high novelty)
        padded = np.pad(self.visit_count, pad, mode='constant', constant_values=0.0)
        
        # Extract crop centred on drone
        pr = row + pad
        pc = col + pad
        crop = padded[pr - half:pr + half + 1, pc - half:pc + half + 1]
        
        # Convert visit counts to novelty: 1/(count+1)
        novelty = 1.0 / (crop + 1.0)  # [0,1], 1=unvisited
        
        return novelty
    
    def reset(self):
        """Reset the grid for a new episode."""
        self.visit_count.fill(0.0)
        self._last_pos = None


# =============================================================================
# ROS 2 NODE
# =============================================================================

class IrisICMMapInferenceNode(Node):
    def __init__(self):
        super().__init__("iris_icm_map_inference_node")

        # ---- parameters ----
        self.declare_parameter("checkpoint_path", "/home/user/ros2_jazzy/src/sim2real/trained_models/icm_map_best.pt")
        self.declare_parameter("depth_topic", "/m2h/depth/image")
        self.declare_parameter("vio_topic", "/rvio2/trajectory")
        self.declare_parameter("action_topic", "/uav/action_cmd")
        self.declare_parameter("inference_rate_hz", 20.0)
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("reset_on_timeout", True)
        self.declare_parameter("grid_reset_timeout_s", 5.0)

        ckpt_path = self.get_parameter("checkpoint_path").value
        depth_topic = self.get_parameter("depth_topic").value
        vio_topic = self.get_parameter("vio_topic").value
        action_topic = self.get_parameter("action_topic").value
        rate_hz = float(self.get_parameter("inference_rate_hz").value)
        device_str = self.get_parameter("device").value
        self.reset_on_timeout = self.get_parameter("reset_on_timeout").value
        self.grid_reset_timeout = float(self.get_parameter("grid_reset_timeout_s").value)

        self.device = torch.device(device_str)
        self.get_logger().info(f"Loading policy checkpoint: {ckpt_path} on {self.device}")
        self.policy = load_policy(ckpt_path, self.device)
        self.get_logger().info("Policy loaded.")

        self.bridge = CvBridge()

        # ---- runtime state ----
        self._latest_depth = np.full((CAM_H, CAM_W), 0.5, dtype=np.float32)
        self._have_depth = False
        self._have_pose = False
        
        # VIO pose (local frame)
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0
        self._last_pose_time = self.get_clock().now()
        
        # Novelty map builder
        self.novelty_builder = NoveltyMapBuilder()
        
        # Frame history
        self._frame_hist = deque(maxlen=HIST_LEN)
        for _ in range(HIST_LEN):
            self._frame_hist.append(
                np.zeros((CAM_H, CAM_W, N_CH), dtype=np.float32))
        
        self._smooth_action = np.zeros(2, dtype=np.float32)
        
        # Stats
        self._step_count = 0
        self._total_visited = 0

        # ---- QoS for VIO (best effort, reliable) ----
        vio_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # ---- pub/sub ----
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        self.sub_vio = self.create_subscription(
            PoseStamped, vio_topic, self._on_vio, vio_qos)
        self.pub_action = self.create_publisher(Twist, action_topic, 10)

        self.timer = self.create_timer(1.0 / rate_hz, self._step)

        self.get_logger().info(
            f"Subscribed depth={depth_topic}, VIO={vio_topic} -> "
            f"publishing {action_topic} at {rate_hz} Hz"
        )
        self.get_logger().info(
            f"Grid: {GRID_N}x{GRID_N} cells @ {GRID_CELL_M}m, "
            f"local crop {LOCAL_MAP_PX}x{LOCAL_MAP_PX}"
        )

    # ------------------------------------------------------------------
    def _on_depth(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"depth conversion failed: {e}")
            return

        depth_m = cv_img.astype(np.float32)
        # 16UC1 depth images (RealSense default) are in millimetres
        if msg.encoding in ("16UC1", "mono16"):
            depth_m = depth_m / 1000.0

        if depth_m.shape != (CAM_H, CAM_W):
            depth_m = cv2.resize(depth_m, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST)

        depth_m = np.nan_to_num(depth_m, nan=CAM_MAX_DEPTH, posinf=CAM_MAX_DEPTH, neginf=CAM_MIN_DEPTH)
        depth_m = np.clip(depth_m, CAM_MIN_DEPTH, CAM_MAX_DEPTH)
        depth_norm = (depth_m - CAM_MIN_DEPTH) / (CAM_MAX_DEPTH - CAM_MIN_DEPTH)

        self._latest_depth = depth_norm.astype(np.float32)
        self._have_depth = True

    # ------------------------------------------------------------------
    def _on_vio(self, msg: PoseStamped):
        """Receive VIO pose (world frame)."""
        self._pos_x = msg.pose.position.x
        self._pos_y = msg.pose.position.y
        self._pos_z = msg.pose.position.z
        self._last_pose_time = self.get_clock().now()
        self._have_pose = True

        # Reset grid if drone has moved significantly (new episode)
        # This happens when the drone is picked up and placed at a new location
        if hasattr(self, '_last_reset_pos'):
            dx = self._pos_x - self._last_reset_pos[0]
            dy = self._pos_y - self._last_reset_pos[1]
            if np.sqrt(dx*dx + dy*dy) > 10.0:  # >10m movement = new episode
                self.novelty_builder.reset()
                self._last_reset_pos = (self._pos_x, self._pos_y)
                self.get_logger().info(f"Grid reset due to large pose jump ({np.sqrt(dx*dx+dy*dy):.1f}m)")
        else:
            self._last_reset_pos = (self._pos_x, self._pos_y)

    # ------------------------------------------------------------------
    def _step(self):
        if not self._have_depth or not self._have_pose:
            return

        # Check for VIO timeout
        now = self.get_clock().now()
        dt = (now - self._last_pose_time).nanoseconds / 1e9
        if dt > self.grid_reset_timeout:
            if self.reset_on_timeout:
                self.novelty_builder.reset()
                self._last_reset_pos = (self._pos_x, self._pos_y)
                self.get_logger().warn(f"Grid reset due to VIO timeout ({dt:.1f}s)")
            # Still use last known pose, but warn
            self.get_logger().debug(f"VIO timeout: {dt:.1f}s")

        # Build observation: depth + novelty map
        depth_frame = self._latest_depth[..., np.newaxis]  # (H, W, 1)
        
        # Update novelty map from VIO pose
        novelty_crop = self.novelty_builder.update(self._pos_x, self._pos_y)
        
        # Resize novelty crop to match depth dimensions (64x80)
        if novelty_crop.shape != (CAM_H, CAM_W):
            novelty_frame = cv2.resize(
                novelty_crop, (CAM_W, CAM_H), 
                interpolation=cv2.INTER_LINEAR
            )
        else:
            novelty_frame = novelty_crop
        
        novelty_frame = novelty_frame[..., np.newaxis]  # (H, W, 1)
        
        # Stack depth + novelty -> (H, W, 2)
        frame = np.concatenate([depth_frame, novelty_frame], axis=-1)
        
        # Update history
        self._frame_hist.append(frame)

        obs = np.stack(list(self._frame_hist), axis=0)  # (T, H, W, 2)
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

        # Policy inference
        mean_action = self.policy.act(obs_t).squeeze(0).cpu().numpy()
        raw = np.clip(mean_action, -1.0, 1.0)

        # EMA smoothing (matches training)
        self._smooth_action = (ACTION_ALPHA * self._smooth_action +
                               (1.0 - ACTION_ALPHA) * raw)

        vx_norm, yaw_norm = self._smooth_action.tolist()

        # Publish
        msg = Twist()
        msg.linear.x = float(vx_norm)
        msg.angular.z = float(yaw_norm)
        self.pub_action.publish(msg)

        # Stats
        self._step_count += 1
        if self._step_count % 100 == 0:
            visited = np.sum(self.novelty_builder.visit_count > 0)
            area = visited * (GRID_CELL_M ** 2)
            self.get_logger().info(
                f"[{self._step_count}] pos=({self._pos_x:.2f}, {self._pos_y:.2f}) "
                f"action=({vx_norm:.2f}, {yaw_norm:.2f}) "
                f"visited={visited} cells ({area:.1f}m²)"
            )

    # ------------------------------------------------------------------
    def reset_grid(self):
        """Public method to manually reset the novelty grid."""
        self.novelty_builder.reset()
        self._last_reset_pos = (self._pos_x, self._pos_y)
        self.get_logger().info("Grid manually reset.")


# =============================================================================
# MAIN
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = IrisICMMapInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()