#!/usr/bin/env python3
"""
iris_icm_inference_node.py  (ENHANCED — VIO-assisted exploration)
======================================================================
ROS 2 inference node for the Isaac-Lab-trained Iris ICM exploration policy.

ENHANCEMENTS:
1. VIO pose integration from /fmu/out/vehicle_local_position_v1
2. Real-time occupancy/visit grid for exploration heatmap
3. Classical frontier-based exploration assistance
4. Room exit detection and retracing
5. Hybrid policy: ICM + classical guidance

Subscribes:
    depth image topic   (sensor_msgs/Image)           - /m2h/depth/image
    VIO pose topic      (VehicleLocalPosition)        - /fmu/out/vehicle_local_position_v1

Publishes:
    uav action topic    (geometry_msgs/Twist)         - /uav/action_cmd
    visit heatmap       (nav_msgs/OccupancyGrid)      - /exploration/heatmap (optional)
"""

from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

try:
    from cv_bridge import CvBridge
    import cv2
except ImportError as e:
    raise ImportError(
        "This node needs cv_bridge and opencv-python. "
        "Install with: sudo apt install ros-<distro>-cv-bridge && pip install opencv-python"
    ) from e

try:
    from px4_msgs.msg import VehicleLocalPosition
except ImportError:
    print("px4_msgs not found. VIO pose will use fallback.")
    VehicleLocalPosition = None


# =============================================================================
# CONSTANTS
# =============================================================================

CAM_H, CAM_W = 64, 80
N_CH = 1
HIST_LEN = 3
CAM_MIN_DEPTH, CAM_MAX_DEPTH = 0.2, 6.0
ACTION_ALPHA = 0.6

# Grid parameters (matches training)
GRID_CELL_M = 0.25
GRID_EXTENT_M = 30.0
GRID_N = int(GRID_EXTENT_M / GRID_CELL_M)  # 120 cells
LOCAL_MAP_PX = 21

# Classical exploration parameters
FRONTIER_RADIUS = 2.0  # meters
EXIT_DETECTION_RADIUS = 1.5
UNVISITED_ATTRACTION = 1.0
ROOM_EXIT_TURN = 1.2
REVISIT_PENALTY = 0.7
MIN_FORWARD_SPEED = 0.2
MAX_FORWARD_SPEED = 0.6
MAX_YAW_RATE = 0.8

# VIO timeout
VIO_TIMEOUT_S = 2.0


# =============================================================================
# MODEL
# =============================================================================

class IrisICMPolicyNet(nn.Module):
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
        self.value_head = nn.Linear(256, 1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        feats = []
        for t in range(self.t_steps):
            frame = obs[:, t].permute(0, 3, 1, 2)
            feats.append(self.cnn(frame))
        shared = self.net(torch.cat(feats, dim=1))
        return self.policy_mean(shared)


def load_policy(checkpoint_path: str, device: torch.device) -> IrisICMPolicyNet:
    model = IrisICMPolicyNet(n_ch=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "policy" in ckpt:
        state_dict = ckpt["policy"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    model_keys = set(model.state_dict().keys())
    for k, v in state_dict.items():
        if k in model_keys:
            cleaned[k] = v
        else:
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
# VISIT GRID / FRONTIER DETECTION
# =============================================================================

class VisitGrid:
    """2D grid tracking visited cells and frontiers."""

    def __init__(self, cell_m=GRID_CELL_M, extent_m=GRID_EXTENT_M):
        self.cell_m = cell_m
        self.n = int(extent_m / cell_m)
        self.half = self.n // 2

        self.visit_count = np.zeros((self.n, self.n), dtype=np.float32)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_set = False

        self.trajectory = []
        self.frontiers = []

    def set_origin(self, x: float, y: float):
        if not self.origin_set:
            self.origin_x = x
            self.origin_y = y
            self.origin_set = True

    def pos_to_cell(self, x: float, y: float):
        if not self.origin_set:
            return 0, 0
        lx = x - self.origin_x
        ly = y - self.origin_y
        col = int(lx / self.cell_m + self.half)
        row = int(-ly / self.cell_m + self.half)
        col = max(0, min(col, self.n - 1))
        row = max(0, min(row, self.n - 1))
        return row, col

    def cell_to_pos(self, row: int, col: int):
        lx = (col - self.half) * self.cell_m
        ly = -(row - self.half) * self.cell_m
        return self.origin_x + lx, self.origin_y + ly

    def visit(self, x: float, y: float):
        row, col = self.pos_to_cell(x, y)
        self.visit_count[row, col] += 1.0
        self.trajectory.append((x, y))

    def get_novelty(self, x: float, y: float) -> float:
        row, col = self.pos_to_cell(x, y)
        return 1.0 / (self.visit_count[row, col] + 1.0)

    def get_novelty_map(self) -> np.ndarray:
        """Return novelty map (1=unvisited, 0=visited many times)."""
        return 1.0 / (self.visit_count + 1.0)

    def get_visitation_grid(self) -> np.ndarray:
        """Return binary visitation grid."""
        return (self.visit_count > 0).astype(np.float32)

    def get_cell_neighbors(self, row: int, col: int, radius: float = 1.0):
        """Get unvisited cells within radius."""
        cell_radius = int(radius / self.cell_m)
        rows, cols = [], []
        for dr in range(-cell_radius, cell_radius + 1):
            for dc in range(-cell_radius, cell_radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    if self.visit_count[nr, nc] == 0:
                        rows.append(nr)
                        cols.append(nc)
        return np.array(rows), np.array(cols)

    def find_frontiers(self, current_x: float, current_y: float, radius: float = FRONTIER_RADIUS):
        """Find frontiers (unvisited cells adjacent to visited cells)."""
        current_row, current_col = self.pos_to_cell(current_x, current_y)
        self.frontiers = []

        # Check cells in a window around current position
        search_radius = int(radius / self.cell_m) + 5

        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                r, c = current_row + dr, current_col + dc
                if 0 <= r < self.n and 0 <= c < self.n:
                    if self.visit_count[r, c] == 0:
                        # Check if adjacent to a visited cell
                        for dr2, dc2 in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr2, c + dc2
                            if 0 <= nr < self.n and 0 <= nc < self.n:
                                if self.visit_count[nr, nc] > 0:
                                    self.frontiers.append((r, c))
                                    break

        return self.frontiers

    def find_nearest_frontier(self, current_x: float, current_y: float) -> tuple:
        """Find the nearest frontier cell to current position."""
        frontiers = self.find_frontiers(current_x, current_y)
        if not frontiers:
            return None

        current_row, current_col = self.pos_to_cell(current_x, current_y)

        min_dist = float('inf')
        nearest = None
        for r, c in frontiers:
            dist = math.sqrt((r - current_row)**2 + (c - current_col)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (r, c)

        return nearest


# =============================================================================
# HYBRID EXPLORATION CONTROLLER
# =============================================================================

class ExplorationController:
    """Combines ICM policy with classical exploration guidance."""

    def __init__(self):
        self.visit_grid = VisitGrid()
        self._last_pose = None
        self._pose_valid = False
        self._room_entered = False
        self._room_exit_attempts = 0
        self._stuck_counter = 0
        self._last_position = None

    def update_pose(self, x: float, y: float):
        """Update current pose and visit grid."""
        self.visit_grid.set_origin(x, y)
        self.visit_grid.visit(x, y)
        self._last_pose = (x, y)
        self._pose_valid = True

        # Detect if stuck (not moving)
        if self._last_position is not None:
            dx = x - self._last_position[0]
            dy = y - self._last_position[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 0.05:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._last_position = (x, y)

    def is_stuck(self) -> bool:
        """Check if the drone is stuck."""
        return self._stuck_counter > 20  # ~2 seconds at 10Hz

    def compute_guidance(self, current_x: float, current_y: float,
                         depth_norm: np.ndarray) -> tuple:
        """
        Compute guidance signal from classical exploration.
        Returns (vx_guidance, yaw_guidance) as normalised values.
        """
        # Get novelty at current position
        novelty = self.visit_grid.get_novelty(current_x, current_y)
        visited_cells = np.sum(self.visit_grid.visit_count > 0)

        # Default: go forward
        vx = MIN_FORWARD_SPEED + (1.0 - MIN_FORWARD_SPEED) * (1.0 - novelty)

        # Check if we're in a room (surrounded by walls/visited cells)
        in_room = self._detect_room(current_x, current_y)

        # Find nearest frontier
        frontier = self.visit_grid.find_nearest_frontier(current_x, current_y)

        # Compute yaw guidance towards frontier
        yaw = 0.0
        if frontier is not None:
            fr, fc = frontier
            fx, fy = self.visit_grid.cell_to_pos(fr, fc)
            dx = fx - current_x
            dy = fy - current_y
            yaw = math.atan2(dy, dx)
            yaw = np.clip(yaw / MAX_YAW_RATE, -1.0, 1.0)

            # If near a frontier, speed up
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1.0:
                vx = max(vx, 0.5)

        # Room exit behavior
        if in_room and visited_cells > 100:
            vx = 0.3
            yaw = self._room_exit_behavior(current_x, current_y)

        # If stuck, try to turn around
        if self.is_stuck():
            vx = 0.1
            yaw = 0.8 * (1.0 if self._stuck_counter % 40 < 20 else -1.0)

        # Wall avoidance from depth
        wall_yaw = self._wall_avoidance_yaw(depth_norm)
        if abs(wall_yaw) > abs(yaw):
            yaw = wall_yaw
            vx = min(vx, 0.3)

        # Ensure minimum forward speed
        vx = max(vx, MIN_FORWARD_SPEED)

        return np.clip(vx, 0.0, 1.0), np.clip(yaw, -1.0, 1.0)

    def _detect_room(self, x: float, y: float) -> bool:
        """Detect if drone is in a room (surrounded by visited cells)."""
        row, col = self.visit_grid.pos_to_cell(x, y)
        radius = 5
        visited_count = 0
        total = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.visit_grid.n and 0 <= c < self.visit_grid.n:
                    total += 1
                    if self.visit_grid.visit_count[r, c] > 0:
                        visited_count += 1

        if total > 0:
            return visited_count / total > 0.8
        return False

    def _room_exit_behavior(self, x: float, y: float) -> float:
        """Turn to exit a room."""
        row, col = self.visit_grid.pos_to_cell(x, y)

        # Find direction with most unvisited cells
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        best_dir = None
        max_unvisited = 0

        for dr, dc in directions:
            r, c = row + dr*3, col + dc*3
            if 0 <= r < self.visit_grid.n and 0 <= c < self.visit_grid.n:
                unvisited = 0
                for dr2 in range(-2, 3):
                    for dc2 in range(-2, 3):
                        nr, nc = r + dr2, c + dc2
                        if 0 <= nr < self.visit_grid.n and 0 <= nc < self.visit_grid.n:
                            if self.visit_grid.visit_count[nr, nc] == 0:
                                unvisited += 1
                if unvisited > max_unvisited:
                    max_unvisited = unvisited
                    best_dir = (dr, dc)

        if best_dir is not None:
            # Convert direction to yaw
            target_x = x + best_dir[1] * 2.0
            target_y = y + best_dir[0] * 2.0
            dx = target_x - x
            dy = target_y - y
            yaw = math.atan2(dy, dx)
            return np.clip(yaw / MAX_YAW_RATE, -1.0, 1.0)

        return 0.0

    def _wall_avoidance_yaw(self, depth_norm: np.ndarray) -> float:
        """Compute yaw to avoid walls from depth image."""
        h, w = depth_norm.shape
        band_w = int(w * 0.3)

        left_band = depth_norm[:, :band_w]
        right_band = depth_norm[:, w - band_w:]

        left_min = float(left_band.min())
        right_min = float(right_band.min())

        # Normalised depth threshold (0=near, 1=far)
        threshold = 0.25

        if left_min < threshold and right_min < threshold:
            # Both sides close - turn towards the more open side
            return 0.8 if left_min <= right_min else -0.8
        elif left_min < threshold:
            return 0.8  # Turn right
        elif right_min < threshold:
            return -0.8  # Turn left

        return 0.0


# =============================================================================
# ROS 2 NODE
# =============================================================================

class IrisICMInferenceNode(Node):
    def __init__(self):
        super().__init__("iris_icm_inference_node")

        # ---- parameters ----
        self.declare_parameter("checkpoint_path", "/root/m2h_ws/src/px4-sim2real-ros2/trained_models/icm_og_best.pt")
        self.declare_parameter("depth_topic", "/m2h/depth/image")
        self.declare_parameter("vio_topic", "/fmu/out/vehicle_local_position_v1")
        self.declare_parameter("action_topic", "/uav/action_cmd")
        self.declare_parameter("inference_rate_hz", 20.0)
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("use_classical_guidance", True)
        self.declare_parameter("mirror_avg", True)

        ckpt_path = self.get_parameter("checkpoint_path").value
        depth_topic = self.get_parameter("depth_topic").value
        vio_topic = self.get_parameter("vio_topic").value
        action_topic = self.get_parameter("action_topic").value
        rate_hz = float(self.get_parameter("inference_rate_hz").value)
        device_str = self.get_parameter("device").value
        self.use_classical = self.get_parameter("use_classical_guidance").value
        self.mirror_avg = self.get_parameter("mirror_avg").value

        self.device = torch.device(device_str)
        self.get_logger().info(f"Loading policy checkpoint: {ckpt_path}")
        self.policy = load_policy(ckpt_path, self.device)
        self.get_logger().info("Policy loaded.")
        self.get_logger().info(
            f"Classical guidance: {self.use_classical}, Mirror avg: {self.mirror_avg}"
        )

        self.bridge = CvBridge()

        # ---- runtime state ----
        self._latest_depth = np.full((CAM_H, CAM_W), 0.5, dtype=np.float32)
        self._have_depth = False
        self._have_pose = False

        # VIO pose
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0
        self._last_pose_time = self.get_clock().now()

        self._frame_hist = deque(maxlen=HIST_LEN)
        for _ in range(HIST_LEN):
            self._frame_hist.append(np.zeros((CAM_H, CAM_W, 1), dtype=np.float32))
        self._smooth_action = np.zeros(2, dtype=np.float32)
        self._step_count = 0

        # Exploration controller
        self.explorer = ExplorationController()

        # ---- pub/sub ----
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self._on_depth, qos_profile_sensor_data)

        # VIO subscription
        vio_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        if VehicleLocalPosition is not None:
            self.sub_vio = self.create_subscription(
                VehicleLocalPosition, vio_topic, self._on_vio, vio_qos)
        else:
            self.get_logger().warn("px4_msgs not available - VIO disabled")
            self.sub_vio = None

        self.pub_action = self.create_publisher(Twist, action_topic, 10)

        # Optional: publish heatmap
        self.pub_heatmap = self.create_publisher(OccupancyGrid, "/exploration/heatmap", 10)

        self.timer = self.create_timer(1.0 / rate_hz, self._step)

        self.get_logger().info(
            f"Subscribed depth={depth_topic}, VIO={vio_topic} -> "
            f"publishing {action_topic} at {rate_hz} Hz"
        )

    # ------------------------------------------------------------------
    def _on_depth(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"depth conversion failed: {e}")
            return

        depth_m = cv_img.astype(np.float32)
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
    def _on_vio(self, msg):
        """Receive VIO position from PX4."""
        if hasattr(msg, 'z_valid') and msg.z_valid:
            self._pos_x = float(msg.x)
            self._pos_y = float(msg.y)
            self._pos_z = -float(msg.z)  # NED to ENU
            self._have_pose = True
            self._last_pose_time = self.get_clock().now()

    # ------------------------------------------------------------------
    def _mirror_average_action(self, obs_t: torch.Tensor) -> np.ndarray:
        """Run policy on original and mirrored observation, average."""
        action_normal = self.policy.act(obs_t).squeeze(0).cpu().numpy()
        obs_mirror = torch.flip(obs_t, dims=[3])
        action_mirror = self.policy.act(obs_mirror).squeeze(0).cpu().numpy()
        action_mirror[1] = -action_mirror[1]
        return 0.5 * (action_normal + action_mirror)

    # ------------------------------------------------------------------
    def _publish_heatmap(self):
        """Publish visitation heatmap as OccupancyGrid."""
        grid = self.explorer.visit_grid
        if not grid.origin_set:
            return

        # Convert novelty to occupancy grid (0=free, 100=occupied/visited)
        novelty = grid.get_novelty_map()
        # Invert: high novelty = unvisited = free (0), low novelty = visited (100)
        occupancy = (1.0 - novelty) * 100
        occupancy = occupancy.astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.width = grid.n
        msg.info.height = grid.n
        msg.info.resolution = grid.cell_m
        msg.info.origin.position.x = grid.origin_x - grid.half * grid.cell_m
        msg.info.origin.position.y = grid.origin_y - grid.half * grid.cell_m
        msg.info.origin.position.z = 0.0
        msg.data = occupancy.flatten().tolist()

        self.pub_heatmap.publish(msg)

    # ------------------------------------------------------------------
    def _step(self):
        if not self._have_depth:
            return

        # Check VIO timeout
        now = self.get_clock().now()
        dt = (now - self._last_pose_time).nanoseconds / 1e9
        if dt > VIO_TIMEOUT_S:
            self._have_pose = False

        # Update pose
        if self._have_pose:
            self.explorer.update_pose(self._pos_x, self._pos_y)

        # Build observation
        frame = self._latest_depth[..., np.newaxis]
        self._frame_hist.append(frame)
        obs = np.stack(list(self._frame_hist), axis=0)
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

        # ── ICM Policy ──────────────────────────────────────────────────────
        if self.mirror_avg:
            raw_action = self._mirror_average_action(obs_t)
        else:
            raw_action = self.policy.act(obs_t).squeeze(0).cpu().numpy()
        raw_action = np.clip(raw_action, -1.0, 1.0)

        # ── Classical Guidance ─────────────────────────────────────────────
        guidance_vx, guidance_yaw = 0.0, 0.0
        if self.use_classical and self._have_pose:
            guidance_vx, guidance_yaw = self.explorer.compute_guidance(
                self._pos_x, self._pos_y, self._latest_depth
            )

        # ── Blend ICM + Classical ─────────────────────────────────────────
        # Use ICM for forward speed, but blend yaw with guidance
        # When novelty is high, trust ICM more; when low, use guidance
        if self._have_pose:
            novelty = self.explorer.visit_grid.get_novelty(self._pos_x, self._pos_y)
            # Weight: high novelty = trust ICM, low novelty = use guidance
            icm_weight = min(1.0, novelty * 2.0)
            guidance_weight = 1.0 - icm_weight

            vx = raw_action[0]
            yaw = raw_action[1] * icm_weight + guidance_yaw * guidance_weight

            # If stuck, force classical guidance
            if self.explorer.is_stuck():
                yaw = guidance_yaw
                vx = max(vx, 0.2)

            # Wall override (always active)
            wall_yaw = self.explorer._wall_avoidance_yaw(self._latest_depth)
            if abs(wall_yaw) > abs(yaw):
                yaw = wall_yaw
                vx = min(vx, 0.3)

        else:
            vx, yaw = raw_action[0], raw_action[1]

        # Clip and smooth
        vx = np.clip(vx, -1.0, 1.0)
        yaw = np.clip(yaw, -1.0, 1.0)

        # Enforce minimum forward speed when moving forward
        if vx > 0:
            vx = max(vx, MIN_FORWARD_SPEED)

        self._smooth_action = (ACTION_ALPHA * self._smooth_action +
                               (1.0 - ACTION_ALPHA) * np.array([vx, yaw]))

        vx_norm, yaw_norm = self._smooth_action.tolist()

        # ── Publish ────────────────────────────────────────────────────────
        msg = Twist()
        msg.linear.x = float(vx_norm)
        msg.angular.z = float(yaw_norm)
        self.pub_action.publish(msg)

        # ── Stats ──────────────────────────────────────────────────────────
        self._step_count += 1
        if self._step_count % 50 == 0:
            visited = np.sum(self.explorer.visit_grid.visit_count > 0)
            area = visited * (GRID_CELL_M ** 2)
            status = "STUCK" if self.explorer.is_stuck() else "OK"
            self.get_logger().info(
                f"[{self._step_count}] pos=({self._pos_x:.2f}, {self._pos_y:.2f}) "
                f"action=({vx_norm:.2f}, {yaw_norm:.2f}) "
                f"visited={visited} cells ({area:.1f}m²) {status}"
            )

        # ── Heatmap ────────────────────────────────────────────────────────
        if self._step_count % 20 == 0:
            self._publish_heatmap()


def main(args=None):
    rclpy.init(args=args)
    node = IrisICMInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()