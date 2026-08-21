#!/usr/bin/env python3
"""
iris_icm_inference_node.py  (ENHANCED — VIO-assisted exploration with Kalman Filter)
======================================================================
ROS 2 inference node with Kalman-filtered pose estimation for accurate mapping.

Subscribes:
    depth image topic   (sensor_msgs/Image)           - /m2h/depth/image
    VIO pose topic      (VehicleLocalPosition)        - /fmu/out/vehicle_local_position_v1

Publishes:
    uav action topic    (geometry_msgs/Twist)         - /uav/action_cmd
    visit heatmap       (nav_msgs/OccupancyGrid)      - /exploration/heatmap
    filtered pose       (geometry_msgs/PoseStamped)   - /exploration/filtered_pose
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
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, Quaternion
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

# Grid parameters
GRID_CELL_M = 0.25
GRID_EXTENT_M = 30.0
GRID_N = int(GRID_EXTENT_M / GRID_CELL_M)
LOCAL_MAP_PX = 21

# Kalman filter parameters
KF_PROCESS_NOISE = 0.01   # Position process noise
KF_MEASUREMENT_NOISE = 0.05  # VIO measurement noise
KF_HEADING_NOISE = 0.01   # Heading measurement noise

# Classical exploration parameters
MIN_FORWARD_SPEED = 0.2
MAX_FORWARD_SPEED = 0.6
MAX_YAW_RATE = 0.8

VIO_TIMEOUT_S = 2.0


# =============================================================================
# KALMAN FILTER FOR POSE ESTIMATION
# =============================================================================

class KalmanFilter2D:
    """
    Simple 2D Kalman filter for position (x, y) with heading.
    State: [x, y, heading, vx, vy, v_heading]
    """
    
    def __init__(self, dt=0.05):
        self.dt = dt
        
        # State vector [x, y, heading, vx, vy, omega]
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1  # Initial uncertainty
        
        # State transition matrix (constant velocity model with heading)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        
        # Measurement matrix (position and heading)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        
        # Process noise covariance
        self.Q = np.eye(6) * KF_PROCESS_NOISE
        self.Q[2, 2] = KF_HEADING_NOISE  # Heading process noise
        
        # Measurement noise covariance
        self.R = np.eye(3) * KF_MEASUREMENT_NOISE
        self.R[2, 2] = KF_HEADING_NOISE * 0.5  # Heading measurement noise
        
        self.initialized = False
        
    def init(self, x, y, heading):
        """Initialize filter with first measurement."""
        self.x = np.array([x, y, heading, 0.0, 0.0, 0.0])
        self.P = np.eye(6) * 0.01
        self.initialized = True
        
    def predict(self):
        """Prediction step."""
        if not self.initialized:
            return self.x[:3]
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]
    
    def update(self, z):
        """Update step with measurement z = [x, y, heading]."""
        if not self.initialized:
            return self.x[:3]
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        # Normalize heading
        self.x[2] = self._normalize_angle(self.x[2])
        
        return self.x[:3]
    
    def get_state(self):
        """Get current filtered state: [x, y, heading]."""
        return self.x[0], self.x[1], self.x[2]
    
    def get_velocity(self):
        """Get current velocity estimates: [vx, vy, omega]."""
        return self.x[3], self.x[4], self.x[5]
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# =============================================================================
# ENHANCED VISIT GRID WITH LOOP CLOSURE
# =============================================================================

class VisitGrid:
    """
    2D grid tracking visited cells with heading-aware updates
    and loop closure detection.
    """
    
    def __init__(self, cell_m=GRID_CELL_M, extent_m=GRID_EXTENT_M):
        self.cell_m = cell_m
        self.n = int(extent_m / cell_m)
        self.half = self.n // 2

        self.visit_count = np.zeros((self.n, self.n), dtype=np.float32)
        self.heading_at_cell = np.zeros((self.n, self.n), dtype=np.float32)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_heading = 0.0
        self.origin_set = False

        self.trajectory = []
        self.filtered_trajectory = []
        self.loop_closure_candidates = []
        self.drift_correction = np.zeros(2)  # Accumulated drift correction
        
        # For loop closure detection
        self.visited_positions = []
        self.visited_headings = []

    def set_origin(self, x: float, y: float, heading: float = 0.0):
        if not self.origin_set:
            self.origin_x = x
            self.origin_y = y
            self.origin_heading = heading
            self.origin_set = True
            print(f"[VisitGrid] Origin set to ({x:.2f}, {y:.2f}) heading {math.degrees(heading):.1f}°")

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

    def visit(self, x: float, y: float, heading: float = 0.0, filtered: bool = True):
        """Visit a cell with heading information."""
        row, col = self.pos_to_cell(x, y)
        self.visit_count[row, col] += 1.0
        
        # Store heading for this cell (weighted average)
        if self.visit_count[row, col] == 1:
            self.heading_at_cell[row, col] = heading
        else:
            # Update heading with exponential moving average
            alpha = 0.3
            self.heading_at_cell[row, col] = self._normalize_angle(
                self.heading_at_cell[row, col] * (1 - alpha) + heading * alpha
            )
        
        if filtered:
            self.filtered_trajectory.append((x, y, heading))
        else:
            self.trajectory.append((x, y, heading))
        
        # Store for loop closure
        self.visited_positions.append((x, y))
        self.visited_headings.append(heading)

    def get_novelty(self, x: float, y: float) -> float:
        row, col = self.pos_to_cell(x, y)
        return 1.0 / (self.visit_count[row, col] + 1.0)

    def get_novelty_map(self) -> np.ndarray:
        return 1.0 / (self.visit_count + 1.0)

    def get_visitation_grid(self) -> np.ndarray:
        return (self.visit_count > 0).astype(np.float32)

    def detect_loop_closure(self, x: float, y: float, heading: float, 
                           radius: float = 2.0, angle_threshold: float = 30.0) -> bool:
        """Detect if current position matches a previously visited position."""
        if len(self.visited_positions) < 10:
            return False
        
        # Check recent positions for loop closure
        recent_positions = self.visited_positions[-50:]  # Check last 50 positions
        
        for i, (px, py) in enumerate(recent_positions[:-5]):  # Skip current position
            dx = x - px
            dy = y - py
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < radius:
                # Check heading similarity
                if len(self.visited_headings) > i:
                    h_diff = self._normalize_angle(heading - self.visited_headings[i])
                    if abs(h_diff) < math.radians(angle_threshold):
                        # Loop closure detected!
                        self.loop_closure_candidates.append((px, py))
                        return True
        return False

    def apply_drift_correction(self, x: float, y: float, correction: np.ndarray):
        """Apply drift correction to map."""
        self.drift_correction += correction
        # Move origin to correct drift
        self.origin_x += correction[0]
        self.origin_y += correction[1]

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


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
        self.declare_parameter("use_kalman_filter", True)

        ckpt_path = self.get_parameter("checkpoint_path").value
        depth_topic = self.get_parameter("depth_topic").value
        vio_topic = self.get_parameter("vio_topic").value
        action_topic = self.get_parameter("action_topic").value
        rate_hz = float(self.get_parameter("inference_rate_hz").value)
        device_str = self.get_parameter("device").value
        self.use_classical = self.get_parameter("use_classical_guidance").value
        self.mirror_avg = self.get_parameter("mirror_avg").value
        self.use_kalman = self.get_parameter("use_kalman_filter").value

        self.device = torch.device(device_str)
        self.get_logger().info(f"Loading policy checkpoint: {ckpt_path}")
        self.policy = load_policy(ckpt_path, self.device)
        self.get_logger().info("Policy loaded.")
        self.get_logger().info(
            f"Classical guidance: {self.use_classical}, "
            f"Mirror avg: {self.mirror_avg}, "
            f"Kalman filter: {self.use_kalman}"
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
        self._heading = 0.0
        self._heading_valid = False
        self._last_pose_time = self.get_clock().now()

        # Kalman filter
        self.kf = KalmanFilter2D(dt=1.0/rate_hz)
        self._filtered_x = 0.0
        self._filtered_y = 0.0
        self._filtered_heading = 0.0

        self._frame_hist = deque(maxlen=HIST_LEN)
        for _ in range(HIST_LEN):
            self._frame_hist.append(np.zeros((CAM_H, CAM_W, 1), dtype=np.float32))
        self._smooth_action = np.zeros(2, dtype=np.float32)
        self._step_count = 0

        # Visit grid
        self.visit_grid = VisitGrid()
        self._loop_closure_detected = False
        self._last_correction = np.zeros(2)

        # ---- pub/sub ----
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self._on_depth, qos_profile_sensor_data)

        vio_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        if VehicleLocalPosition is not None:
            self.sub_vio = self.create_subscription(
                VehicleLocalPosition, vio_topic, self._on_vio, vio_qos)
        else:
            self.get_logger().warn("px4_msgs not available - VIO disabled")
            self.sub_vio = None

        self.pub_action = self.create_publisher(Twist, action_topic, 10)
        self.pub_heatmap = self.create_publisher(OccupancyGrid, "/exploration/heatmap", 10)
        self.pub_filtered_pose = self.create_publisher(PoseStamped, "/exploration/filtered_pose", 10)

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
        """Receive VIO position and heading from PX4."""
        self._pos_x = float(msg.x)
        self._pos_y = float(msg.y)
        self._pos_z = -float(msg.z)
        
        if hasattr(msg, 'heading_valid') and msg.heading_valid:
            self._heading = float(msg.heading)
            self._heading_valid = True
        elif hasattr(msg, 'heading'):
            self._heading = float(msg.heading)
            self._heading_valid = True
            
        self._have_pose = True
        self._last_pose_time = self.get_clock().now()

    # ------------------------------------------------------------------
    def _mirror_average_action(self, obs_t: torch.Tensor) -> np.ndarray:
        action_normal = self.policy.act(obs_t).squeeze(0).cpu().numpy()
        obs_mirror = torch.flip(obs_t, dims=[3])
        action_mirror = self.policy.act(obs_mirror).squeeze(0).cpu().numpy()
        action_mirror[1] = -action_mirror[1]
        return 0.5 * (action_normal + action_mirror)

    # ------------------------------------------------------------------
    def _wall_avoidance_yaw(self, depth_norm: np.ndarray) -> float:
        h, w = depth_norm.shape
        band_w = int(w * 0.3)
        left_band = depth_norm[:, :band_w]
        right_band = depth_norm[:, w - band_w:]
        left_min = float(left_band.min())
        right_min = float(right_band.min())
        threshold = 0.25

        if left_min < threshold and right_min < threshold:
            return 0.8 if left_min <= right_min else -0.8
        elif left_min < threshold:
            return 0.8
        elif right_min < threshold:
            return -0.8
        return 0.0

    # ------------------------------------------------------------------
    def _publish_heatmap(self):
        grid = self.visit_grid
        if not grid.origin_set:
            return

        novelty = grid.get_novelty_map()
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
    def _publish_filtered_pose(self, x: float, y: float, heading: float):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        
        # Convert heading to quaternion
        qx = 0.0
        qy = 0.0
        qz = math.sin(heading / 2.0)
        qw = math.cos(heading / 2.0)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        
        self.pub_filtered_pose.publish(msg)

    # ------------------------------------------------------------------
    def _step(self):
        if not self._have_depth:
            return

        # Check VIO timeout
        now = self.get_clock().now()
        dt = (now - self._last_pose_time).nanoseconds / 1e9
        if dt > VIO_TIMEOUT_S:
            self._have_pose = False

        # ── Kalman Filter Update ──────────────────────────────────────────
        if self._have_pose and self._heading_valid:
            if not self.kf.initialized:
                self.kf.init(self._pos_x, self._pos_y, self._heading)
                self.visit_grid.set_origin(self._pos_x, self._pos_y, self._heading)
                self._filtered_x = self._pos_x
                self._filtered_y = self._pos_y
                self._filtered_heading = self._heading
            else:
                # Predict
                self.kf.predict()
                
                # Update with measurement
                z = np.array([self._pos_x, self._pos_y, self._heading])
                filtered = self.kf.update(z)
                self._filtered_x, self._filtered_y, self._filtered_heading = filtered
                
                # Check for loop closure
                if self.visit_grid.detect_loop_closure(
                    self._filtered_x, self._filtered_y, self._filtered_heading,
                    radius=1.5, angle_threshold=20.0
                ):
                    self._loop_closure_detected = True
                    self.get_logger().info("Loop closure detected! Correcting drift.")
                
                # Apply drift correction if loop closure detected
                if self._loop_closure_detected and len(self.visit_grid.loop_closure_candidates) > 0:
                    # Find the best matching previous position
                    for px, py in self.visit_grid.loop_closure_candidates:
                        # Apply small correction towards the previous position
                        correction = np.array([(px - self._filtered_x) * 0.1,
                                               (py - self._filtered_y) * 0.1])
                        self.visit_grid.apply_drift_correction(
                            self._filtered_x, self._filtered_y, correction
                        )
                        self._last_correction = correction
                        break
                    self._loop_closure_detected = False

        # ── Use filtered or raw pose ─────────────────────────────────────
        if self._have_pose and self.use_kalman and self.kf.initialized:
            pos_x, pos_y, heading = self.kf.get_state()
        elif self._have_pose:
            pos_x, pos_y, heading = self._pos_x, self._pos_y, self._heading
        else:
            pos_x, pos_y, heading = 0.0, 0.0, 0.0

        # ── Update Visit Grid ─────────────────────────────────────────────
        if self._have_pose:
            # Use filtered pose for grid update
            if self.use_kalman and self.kf.initialized:
                fx, fy, fh = self.kf.get_state()
            else:
                fx, fy, fh = self._pos_x, self._pos_y, self._heading
                
            self.visit_grid.visit(fx, fy, fh, filtered=True)
            
            # Also store raw trajectory for comparison
            if self._have_pose:
                self.visit_grid.visit(self._pos_x, self._pos_y, self._heading, filtered=False)

        # ── Build observation ─────────────────────────────────────────────
        frame = self._latest_depth[..., np.newaxis]
        self._frame_hist.append(frame)
        obs = np.stack(list(self._frame_hist), axis=0)
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

        # ── ICM Policy ─────────────────────────────────────────────────────
        if self.mirror_avg:
            raw_action = self._mirror_average_action(obs_t)
        else:
            raw_action = self.policy.act(obs_t).squeeze(0).cpu().numpy()
        raw_action = np.clip(raw_action, -1.0, 1.0)

        # ── Classical Guidance ────────────────────────────────────────────
        vx, yaw = raw_action[0], raw_action[1]
        
        if self.use_classical and self._have_pose:
            novelty = self.visit_grid.get_novelty(pos_x, pos_y)
            icm_weight = min(1.0, novelty * 2.0)
            guidance_weight = 1.0 - icm_weight
            
            # Compute guidance towards nearest frontier
            frontier = self.visit_grid.get_nearest_frontier(pos_x, pos_y)
            if frontier is not None:
                fr, fc = frontier
                fx, fy = self.visit_grid.cell_to_pos(fr, fc)
                dx = fx - pos_x
                dy = fy - pos_y
                guidance_yaw = math.atan2(dy, dx)
                guidance_yaw = np.clip(guidance_yaw / MAX_YAW_RATE, -1.0, 1.0)
                yaw = yaw * icm_weight + guidance_yaw * guidance_weight
                vx = max(vx, MIN_FORWARD_SPEED)

        # ── Wall Avoidance ──────────────────────────────────────────────────
        wall_yaw = self._wall_avoidance_yaw(self._latest_depth)
        if abs(wall_yaw) > abs(yaw):
            yaw = wall_yaw
            vx = min(vx, 0.3)

        # Clip and smooth
        vx = np.clip(vx, -1.0, 1.0)
        yaw = np.clip(yaw, -1.0, 1.0)
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
            visited = np.sum(self.visit_grid.visit_count > 0)
            area = visited * (GRID_CELL_M ** 2)
            if self.use_kalman and self.kf.initialized:
                fx, fy, fh = self.kf.get_state()
                self.get_logger().info(
                    f"[{self._step_count}] filtered=({fx:.2f}, {fy:.2f}) "
                    f"raw=({self._pos_x:.2f}, {self._pos_y:.2f}) "
                    f"action=({vx_norm:.2f}, {yaw_norm:.2f}) "
                    f"visited={visited} cells ({area:.1f}m²)"
                )
            else:
                self.get_logger().info(
                    f"[{self._step_count}] pos=({self._pos_x:.2f}, {self._pos_y:.2f}) "
                    f"action=({vx_norm:.2f}, {yaw_norm:.2f}) "
                    f"visited={visited} cells ({area:.1f}m²)"
                )

        # ── Publish visualizations ────────────────────────────────────────
        if self._step_count % 20 == 0:
            self._publish_heatmap()
            if self.use_kalman and self.kf.initialized:
                fx, fy, fh = self.kf.get_state()
                self._publish_filtered_pose(fx, fy, fh)


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