#!/usr/bin/env python3
"""
iris_icm_inference_node.py  (CORRECTED — heading-aware guidance + Kalman-filtered pose)
======================================================================
ROS 2 inference node for the Isaac-Lab-trained Iris ICM exploration policy.

BUGS FIXED IN THIS VERSION (see inline "# FIX" comments for exact locations)
------------------------------------------------------------------------
FIX 1 — `_on_vio` gated position updates on `z_valid` (altitude validity)
    instead of `xy_valid` (horizontal position validity). These are
    different EKF health flags on PX4; altitude being good tells you
    nothing about whether the optical-flow-derived x/y is trustworthy.

FIX 2 — THE BIG ONE. Every place the classical guidance computed a target
    direction (`compute_guidance`'s frontier-seeking, and
    `_room_exit_behavior`), it did:
        yaw = math.atan2(dy, dx)
        yaw = np.clip(yaw / MAX_YAW_RATE, -1.0, 1.0)
    `atan2(dy,dx)` is an ABSOLUTE world-frame bearing in radians. It was
    never compared against the drone's actual heading — it was just
    divided by a constant and clipped, as if it were already a relative
    steering command. This means the guidance layer was steering toward
    essentially arbitrary values whenever the target bearing exceeded a
    few tenths of a radian, regardless of which way the drone was
    actually facing. Fixed by computing a proper heading error:
        yaw_err = wrap_angle(desired_heading - current_heading)
        yaw_cmd = clip(Kp * yaw_err, -1, 1)
    This requires threading the drone's current heading into both
    functions, which it never had access to before.

FIX 3 — POSITION "MISMATCH DURING LOOPS". Raw PX4 optical-flow position
    is fed straight into the visitation grid with no filtering. Optical
    flow position estimates are least reliable during sustained yaw
    (motion blur / feature loss while rotating) — i.e. exactly during a
    loop manoeuvre — and PX4's EKF can also occasionally emit a step
    discontinuity ("reset") in position. Both corrupt the grid: a cell
    that should register as revisited gets attributed to the wrong
    location, producing a visible seam/mismatch when the drone loops
    back near its own earlier path. Fixed with a small constant-velocity
    Kalman filter (`PoseKalmanFilter`) that:
        - predicts position from velocity between updates,
        - inflates its trust in the raw position measurement in
          proportion to the current yaw rate (less trust while turning),
        - hard-detects large jumps (EKF resets) and snaps to them rather
          than smoothing over them, since smoothing a real reset would
          smear the trajectory across two locations instead of jumping
          cleanly between them.
    A companion `HeadingTracker` lightly smooths the heading estimate and
    derives yaw rate from it (used to drive the KF's turn-aware gating).

FIX 4 — `VisitGrid` grid size was not forced odd, so `half = n // 2` did
    not correspond to an exact centre cell — a small asymmetry that
    compounds over a long flight. Also, cell indexing used `int(...)`
    (truncation) rather than `round(...)`, which biases negative
    coordinates inconsistently at cell boundaries. Both fixed.

FIX 5 — `is_stuck()` used a hard-coded frame count assuming 10 Hz
    ("`> 20  # ~2 seconds at 10Hz`") while the node defaults to 20 Hz —
    actually corresponding to 1 second, not 2. Now derived from the
    actual configured control rate.

FIX 6 — Room-exit trigger included an unrelated global gate
    (`visited_cells > 100`, i.e. >6.25 m² visited ANYWHERE, ever, in the
    whole flight) alongside the correct local density check
    (`_detect_room`). Once ~6 m² had been visited anywhere, room-exit
    behaviour could fire far from any actual enclosed room for the rest
    of the flight. Removed — room-exit now triggers purely from the
    local density check, which is what it's supposed to measure.

FIX 7 — In `_step()`, `guidance_vx` was computed by
    `compute_guidance()` but then silently discarded — only
    `guidance_yaw` was ever blended in; forward speed always came
    straight from ICM regardless of what the classical layer recommended
    (slowing near frontiers, in rooms, while stuck). Now both vx and yaw
    are blended with the same novelty-based weighting.

FIX 8 — `find_frontiers` and `_detect_room` used nested Python double
    loops over the search window every control step (up to ~700 cell
    checks per step for frontiers). Rewritten with vectorised numpy
    operations (array shifting for the 4-neighbour adjacency test) —
    functionally identical result, substantially cheaper per step, which
    also reduces timing jitter that would otherwise feed inconsistent dt
    into the new Kalman filter's predict step.

Subscribes:
    depth image topic   (sensor_msgs/Image)           - /m2h/depth/image
    VIO pose topic      (VehicleLocalPosition)        - /fmu/out/vehicle_local_position_v1

Publishes:
    uav action topic    (geometry_msgs/Twist)         - /uav/action_cmd
    visit heatmap       (nav_msgs/OccupancyGrid)      - /exploration/heatmap (optional)

Deployment note on heading reliability
------------------------------------------------------------------------
PX4's `heading` field is generally a fusion of gyro (+ optionally
magnetometer) with optical flow. Indoors, magnetometer fusion is
frequently degraded by structural steel and electronics (a very common
source of slow heading drift that this code cannot detect or correct in
software). If heading appears to drift steadily in one direction over a
long flight with no corresponding real rotation, check whether EKF2 is
configured to fuse mag indoors (EKF2_MAG_TYPE) — this is a PX4 parameter
issue, not something fixable in this node.
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
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid

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
LOCAL_MAP_PX = 21

# Classical exploration parameters
FRONTIER_RADIUS = 2.0
UNVISITED_ATTRACTION = 1.0
REVISIT_PENALTY = 0.7
MIN_FORWARD_SPEED = 0.2
MAX_FORWARD_SPEED = 0.6
MAX_YAW_RATE = 0.8

# FIX 2 — proportional gains for heading-error-based steering (previously
# these commands were never actually error-driven, so no gain existed)
FRONTIER_KP_YAW   = 0.6
ROOM_EXIT_KP_YAW  = 0.6

# VIO timeout
VIO_TIMEOUT_S = 2.0

# FIX 3 — Kalman filter tuning. All exposed here so they can be retuned
# without hunting through the class body.
KF_Q_POS        = 0.01   # process noise on position (m^2 per second)
KF_Q_VEL        = 0.5    # process noise on velocity (m^2/s^2 per second)
KF_R_POS_BASE   = 0.05   # baseline measurement noise on position (m^2) while not turning
KF_R_TURN_SCALE = 10.0   # multiplies R_pos by (1 + this * |yaw_rate|) while turning
KF_R_VEL        = 0.05   # measurement noise on velocity (m^2/s^2)
KF_MAX_JUMP_M   = 1.0    # position discontinuities larger than this are treated as EKF resets
HEADING_SMOOTH_ALPHA = 0.3   # exponential smoothing factor on heading itself


def wrap_angle(a: float) -> float:
    while a > math.pi:  a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


# =============================================================================
# MODEL — unchanged
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
# FIX 3 — POSE FILTERING: turn-aware Kalman filter + heading tracker
# =============================================================================

class PoseKalmanFilter:
    """
    Constant-velocity Kalman filter over PX4's local NED position/velocity.

    State: [x, y, vx, vy]  (x=North, y=East, metres / m/s — PX4 convention)

    The measurement noise on POSITION is inflated proportionally to the
    current yaw rate: optical flow position estimates are least reliable
    while the drone is rotating (motion blur, feature loss), which is
    exactly when a naive filter would otherwise happily accept a bad
    reading and corrupt the visitation grid — the "mismatch during loops"
    symptom this whole filter exists to address.

    Large position discontinuities (bigger than physically possible given
    dt and a sane max speed) are treated as genuine EKF resets rather than
    measurement noise, and the filter snaps to them immediately instead of
    blending — blending a real reset would smear the trajectory across
    two different real locations, which is worse than either alone.
    """

    def __init__(self,
                 q_pos: float = KF_Q_POS, q_vel: float = KF_Q_VEL,
                 r_pos_base: float = KF_R_POS_BASE,
                 r_turn_scale: float = KF_R_TURN_SCALE,
                 r_vel: float = KF_R_VEL,
                 max_jump_m: float = KF_MAX_JUMP_M):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_pos_base = r_pos_base
        self.r_turn_scale = r_turn_scale
        self.r_vel = r_vel
        self.max_jump_m = max_jump_m
        self._initialised = False

    def initialise(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.x = np.array([x, y, vx, vy], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self._initialised = True

    def predict(self, dt: float):
        if not self._initialised or dt <= 0.0:
            return
        F = np.array([[1.0, 0.0, dt,  0.0],
                      [0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        Q = np.diag([self.q_pos * dt, self.q_pos * dt,
                    self.q_vel * dt, self.q_vel * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas_x: float, meas_y: float,
               meas_vx: float, meas_vy: float,
               yaw_rate_abs: float) -> bool:
        """
        Returns True if this measurement was treated as an EKF reset
        (filter snapped to it) rather than a normal correction — useful
        to log so reset frequency can be monitored during flight tests.
        """
        if not self._initialised:
            self.initialise(meas_x, meas_y, meas_vx, meas_vy)
            return False

        jump_dist = math.hypot(meas_x - self.x[0], meas_y - self.x[1])
        if jump_dist > self.max_jump_m:
            self.initialise(meas_x, meas_y, meas_vx, meas_vy)
            return True

        r_pos = self.r_pos_base * (1.0 + self.r_turn_scale * abs(yaw_rate_abs))
        R = np.diag([r_pos, r_pos, self.r_vel, self.r_vel])
        H = np.eye(4)
        z = np.array([meas_x, meas_y, meas_vx, meas_vy])

        y_resid = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_resid
        self.P = (np.eye(4) - K @ H) @ self.P
        return False

    @property
    def position(self) -> tuple:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple:
        return float(self.x[2]), float(self.x[3])


class HeadingTracker:
    """
    Lightly smooths PX4's heading estimate and derives a smoothed yaw
    rate from it. The yaw rate is used purely to gate the position
    Kalman filter's trust (see PoseKalmanFilter.update) — this is the
    mechanism that reduces grid corruption specifically during turns.
    """

    def __init__(self, alpha: float = HEADING_SMOOTH_ALPHA):
        self.heading  = 0.0
        self.yaw_rate = 0.0
        self._prev_heading = None
        self._prev_t = None
        self.alpha = alpha
        self._initialised = False

    def update(self, heading_raw: float, t: float):
        if not self._initialised:
            self.heading = heading_raw
            self._prev_heading = heading_raw
            self._prev_t = t
            self._initialised = True
            return

        dh = wrap_angle(heading_raw - self.heading)
        self.heading = wrap_angle(self.heading + self.alpha * dh)

        dt = t - self._prev_t
        if dt > 1e-3:
            raw_rate = wrap_angle(self.heading - self._prev_heading) / dt
            self.yaw_rate = 0.5 * self.yaw_rate + 0.5 * raw_rate
        self._prev_heading = self.heading
        self._prev_t = t


# =============================================================================
# VISIT GRID / FRONTIER DETECTION
# =============================================================================

class VisitGrid:
    """2D grid tracking visited cells and frontiers."""

    def __init__(self, cell_m=GRID_CELL_M, extent_m=GRID_EXTENT_M):
        self.cell_m = cell_m
        # FIX 4a — force odd grid size so `half` is an exact centre cell
        self.n = int(extent_m / cell_m) | 1
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
        # FIX 4b — round rather than truncate, avoids asymmetric bias at
        # cell boundaries (matters more once loops repeatedly cross them)
        col = int(round(lx / self.cell_m + self.half))
        row = int(round(-ly / self.cell_m + self.half))
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
        return 1.0 / (self.visit_count + 1.0)

    def get_visitation_grid(self) -> np.ndarray:
        return (self.visit_count > 0).astype(np.float32)

    # FIX 8a — vectorised frontier search (was a nested python double loop)
    def find_frontiers(self, current_x: float, current_y: float,
                       radius: float = FRONTIER_RADIUS):
        current_row, current_col = self.pos_to_cell(current_x, current_y)
        search_radius = int(radius / self.cell_m) + 5

        r0 = max(0, current_row - search_radius)
        r1 = min(self.n, current_row + search_radius + 1)
        c0 = max(0, current_col - search_radius)
        c1 = min(self.n, current_col + search_radius + 1)

        sub = self.visit_count[r0:r1, c0:c1]
        unvisited = sub == 0
        visited   = sub > 0

        adj = np.zeros_like(visited)
        adj[1:, :]  |= visited[:-1, :]
        adj[:-1, :] |= visited[1:, :]
        adj[:, 1:]  |= visited[:, :-1]
        adj[:, :-1] |= visited[:, 1:]

        frontier_mask = unvisited & adj
        rows_local, cols_local = np.nonzero(frontier_mask)
        rows = (rows_local + r0).tolist()
        cols = (cols_local + c0).tolist()
        self.frontiers = list(zip(rows, cols))
        return self.frontiers

    # FIX 8b — vectorised nearest-frontier distance search
    def find_nearest_frontier(self, current_x: float, current_y: float):
        frontiers = self.find_frontiers(current_x, current_y)
        if not frontiers:
            return None

        current_row, current_col = self.pos_to_cell(current_x, current_y)
        arr = np.array(frontiers, dtype=np.float32)   # (N, 2) -> [row, col]
        d = np.hypot(arr[:, 0] - current_row, arr[:, 1] - current_col)
        idx = int(np.argmin(d))
        return tuple(frontiers[idx])


# =============================================================================
# HYBRID EXPLORATION CONTROLLER
# =============================================================================

class ExplorationController:
    """Combines ICM policy with classical exploration guidance."""

    def __init__(self, control_hz: float = 20.0):
        self.visit_grid = VisitGrid()
        self._last_pose = None
        self._pose_valid = False
        self._stuck_counter = 0
        self._last_position = None
        # FIX 5 — stuck threshold now derived from actual control rate
        self._stuck_frames_thresh = max(1, int(1.0 * control_hz))

    def update_pose(self, x: float, y: float):
        self.visit_grid.set_origin(x, y)
        self.visit_grid.visit(x, y)
        self._last_pose = (x, y)
        self._pose_valid = True

        if self._last_position is not None:
            dx = x - self._last_position[0]
            dy = y - self._last_position[1]
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._last_position = (x, y)

    def is_stuck(self) -> bool:
        return self._stuck_counter > self._stuck_frames_thresh

    def compute_guidance(self, current_x: float, current_y: float,
                         current_heading: float,
                         depth_norm: np.ndarray) -> tuple:
        """
        FIX 2 — current_heading is now a required argument, and every
        target-direction computation below produces a proper relative
        steering command (heading error * gain), not a clipped absolute
        bearing.
        """
        novelty = self.visit_grid.get_novelty(current_x, current_y)

        vx = MIN_FORWARD_SPEED + (1.0 - MIN_FORWARD_SPEED) * (1.0 - novelty)

        in_room = self._detect_room(current_x, current_y)
        frontier = self.visit_grid.find_nearest_frontier(current_x, current_y)

        yaw = 0.0
        if frontier is not None:
            fr, fc = frontier
            fx, fy = self.visit_grid.cell_to_pos(fr, fc)
            dx, dy = fx - current_x, fy - current_y
            desired_heading = math.atan2(dy, dx)
            # FIX 2 — proper relative heading error, not a raw absolute bearing
            yaw_err = wrap_angle(desired_heading - current_heading)
            yaw = float(np.clip(FRONTIER_KP_YAW * yaw_err, -1.0, 1.0))

            dist = math.hypot(dx, dy)
            if dist < 1.0:
                vx = max(vx, 0.5)

        # FIX 6 — dropped the unrelated global `visited_cells > 100` gate;
        # room-exit now triggers purely from local density (_detect_room)
        if in_room:
            vx = 0.3
            yaw = self._room_exit_behavior(current_x, current_y, current_heading)

        if self.is_stuck():
            vx = 0.1
            yaw = 0.8 * (1.0 if self._stuck_counter % 40 < 20 else -1.0)

        wall_yaw = self._wall_avoidance_yaw(depth_norm)
        if abs(wall_yaw) > abs(yaw):
            yaw = wall_yaw
            vx = min(vx, 0.3)

        vx = max(vx, MIN_FORWARD_SPEED)
        return float(np.clip(vx, 0.0, 1.0)), float(np.clip(yaw, -1.0, 1.0))

    # FIX 8c — vectorised local-density check (was a nested python loop)
    def _detect_room(self, x: float, y: float) -> bool:
        row, col = self.visit_grid.pos_to_cell(x, y)
        radius = 5
        r0 = max(0, row - radius); r1 = min(self.visit_grid.n, row + radius + 1)
        c0 = max(0, col - radius); c1 = min(self.visit_grid.n, col + radius + 1)
        sub = self.visit_grid.visit_count[r0:r1, c0:c1]
        if sub.size == 0:
            return False
        return float(np.mean(sub > 0)) > 0.8

    def _room_exit_behavior(self, x: float, y: float, current_heading: float) -> float:
        """FIX 2 — same relative-heading-error fix applied here as in compute_guidance."""
        row, col = self.visit_grid.pos_to_cell(x, y)
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        best_dir, max_unvisited = None, 0

        for dr, dc in directions:
            r, c = row + dr * 3, col + dc * 3
            if 0 <= r < self.visit_grid.n and 0 <= c < self.visit_grid.n:
                r0 = max(0, r - 2); r1 = min(self.visit_grid.n, r + 3)
                c0 = max(0, c - 2); c1 = min(self.visit_grid.n, c + 3)
                sub = self.visit_grid.visit_count[r0:r1, c0:c1]
                unvisited = int(np.sum(sub == 0))
                if unvisited > max_unvisited:
                    max_unvisited = unvisited
                    best_dir = (dr, dc)

        if best_dir is not None:
            target_x = x + best_dir[1] * 2.0
            target_y = y + best_dir[0] * 2.0
            dx, dy = target_x - x, target_y - y
            desired_heading = math.atan2(dy, dx)
            yaw_err = wrap_angle(desired_heading - current_heading)
            return float(np.clip(ROOM_EXIT_KP_YAW * yaw_err, -1.0, 1.0))

        return 0.0

    def _wall_avoidance_yaw(self, depth_norm: np.ndarray) -> float:
        h, w = depth_norm.shape
        band_w = int(w * 0.3)
        left_band  = depth_norm[:, :band_w]
        right_band = depth_norm[:, w - band_w:]
        left_min  = float(left_band.min())
        right_min = float(right_band.min())
        threshold = 0.25

        if left_min < threshold and right_min < threshold:
            return 0.8 if left_min <= right_min else -0.8
        elif left_min < threshold:
            return 0.8
        elif right_min < threshold:
            return -0.8
        return 0.0


# =============================================================================
# ROS 2 NODE
# =============================================================================

class IrisICMInferenceNode(Node):
    def __init__(self):
        super().__init__("iris_icm_inference_node")

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

        # FIX 3 — filtered pose replaces raw self._pos_x/_pos_y everywhere downstream
        self.pos_kf = PoseKalmanFilter()
        self.heading_tracker = HeadingTracker()
        self._last_vio_t = None       # for KF predict dt, from msg.timestamp (seconds)
        self._reset_count = 0         # diagnostic: how many EKF resets observed

        self._last_pose_time = self.get_clock().now()

        self._frame_hist = deque(maxlen=HIST_LEN)
        for _ in range(HIST_LEN):
            self._frame_hist.append(np.zeros((CAM_H, CAM_W, 1), dtype=np.float32))
        self._smooth_action = np.zeros(2, dtype=np.float32)
        self._step_count = 0

        self.explorer = ExplorationController(control_hz=rate_hz)

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
        """
        FIX 1 — gate on xy_valid (horizontal position validity), not
        z_valid (altitude validity). These are independent EKF health
        flags on PX4; using the wrong one means the node could trust a
        genuinely invalid horizontal position estimate as long as
        altitude happened to be fine.
        """
        xy_valid = bool(getattr(msg, "xy_valid", True))
        if not xy_valid:
            return

        # Timestamp-based dt for the KF predict step — PX4's own message
        # timestamp (microseconds since boot) is more consistent for this
        # than wall-clock ROS time, which is subject to scheduling jitter.
        t_now = float(msg.timestamp) * 1e-6
        if self._last_vio_t is not None:
            dt = t_now - self._last_vio_t
        else:
            dt = 1.0 / 20.0   # first sample — assume nominal rate
        self._last_vio_t = t_now

        heading_raw = float(getattr(msg, "heading", 0.0))
        self.heading_tracker.update(heading_raw, t_now)

        meas_vx = float(getattr(msg, "vx", 0.0))
        meas_vy = float(getattr(msg, "vy", 0.0))

        self.pos_kf.predict(dt)
        was_reset = self.pos_kf.update(
            float(msg.x), float(msg.y), meas_vx, meas_vy,
            yaw_rate_abs=self.heading_tracker.yaw_rate,
        )
        if was_reset:
            self._reset_count += 1
            self.get_logger().warn(
                f"VIO position discontinuity detected (#{self._reset_count}) — "
                f"filter snapped to new estimate rather than smoothing over it."
            )

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
    def _publish_heatmap(self):
        grid = self.explorer.visit_grid
        if not grid.origin_set:
            return

        novelty = grid.get_novelty_map()
        occupancy = ((1.0 - novelty) * 100).astype(np.int8)

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

        now = self.get_clock().now()
        dt_since_vio = (now - self._last_pose_time).nanoseconds / 1e9
        if dt_since_vio > VIO_TIMEOUT_S:
            self._have_pose = False

        pos_x, pos_y = self.pos_kf.position       # FIX 3 — filtered, not raw
        heading = self.heading_tracker.heading    # FIX 2 — now actually used

        if self._have_pose:
            self.explorer.update_pose(pos_x, pos_y)

        frame = self._latest_depth[..., np.newaxis]
        self._frame_hist.append(frame)
        obs = np.stack(list(self._frame_hist), axis=0)
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

        if self.mirror_avg:
            raw_action = self._mirror_average_action(obs_t)
        else:
            raw_action = self.policy.act(obs_t).squeeze(0).cpu().numpy()
        raw_action = np.clip(raw_action, -1.0, 1.0)

        guidance_vx, guidance_yaw = 0.0, 0.0
        if self.use_classical and self._have_pose:
            guidance_vx, guidance_yaw = self.explorer.compute_guidance(
                pos_x, pos_y, heading, self._latest_depth
            )

        if self._have_pose:
            novelty = self.explorer.visit_grid.get_novelty(pos_x, pos_y)
            icm_weight = min(1.0, novelty * 2.0)
            guidance_weight = 1.0 - icm_weight

            # FIX 7 — vx now blends the same way yaw always did; previously
            # guidance_vx was computed and thrown away.
            vx  = raw_action[0] * icm_weight + guidance_vx  * guidance_weight
            yaw = raw_action[1] * icm_weight + guidance_yaw * guidance_weight

            if self.explorer.is_stuck():
                yaw = guidance_yaw
                vx = max(vx, 0.2)

            wall_yaw = self.explorer._wall_avoidance_yaw(self._latest_depth)
            if abs(wall_yaw) > abs(yaw):
                yaw = wall_yaw
                vx = min(vx, 0.3)
        else:
            vx, yaw = raw_action[0], raw_action[1]

        vx = np.clip(vx, -1.0, 1.0)
        yaw = np.clip(yaw, -1.0, 1.0)
        if vx > 0:
            vx = max(vx, MIN_FORWARD_SPEED)

        self._smooth_action = (ACTION_ALPHA * self._smooth_action +
                               (1.0 - ACTION_ALPHA) * np.array([vx, yaw]))
        vx_norm, yaw_norm = self._smooth_action.tolist()

        msg = Twist()
        msg.linear.x = float(vx_norm)
        msg.angular.z = float(yaw_norm)
        self.pub_action.publish(msg)

        self._step_count += 1
        if self._step_count % 50 == 0:
            visited = int(np.sum(self.explorer.visit_grid.visit_count > 0))
            area = visited * (GRID_CELL_M ** 2)
            status = "STUCK" if self.explorer.is_stuck() else "OK"
            self.get_logger().info(
                f"[{self._step_count}] pos=({pos_x:.2f}, {pos_y:.2f}) "
                f"heading={math.degrees(heading):.0f}deg "
                f"yaw_rate={math.degrees(self.heading_tracker.yaw_rate):.0f}deg/s "
                f"action=({vx_norm:.2f}, {yaw_norm:.2f}) "
                f"visited={visited} cells ({area:.1f}m^2) resets={self._reset_count} {status}"
            )

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