#!/usr/bin/env python3
"""
icm_lstm_inference_node.py
==========================
ROS 2 inference node for the ICM + LSTM exploration policy.

All model classes are defined inline — no dependency on the Isaac Lab
workspace or sim2real training code. The checkpoint loads directly.

Subscribes:
    /m2h/depth/image    (sensor_msgs/Image, 32FC1 or 16UC1)

Publishes:
    /uav/action_cmd     (geometry_msgs/Twist)  normalised [-1,1]
                            linear.x  = vx_norm
                            angular.z = yaw_norm
    /uav/lstm_pose_est  (geometry_msgs/Point)  LSTM XY estimate (m from start)

Parameters (--ros-args -p)
    checkpoint_path   str    required — path to .pt checkpoint
    max_depth_m       float  default 6.0
    min_depth_m       float  default 0.2
    depth_topic       str    default /m2h/depth/image
    action_topic      str    default /uav/action_cmd
    pose_topic        str    default /uav/lstm_pose_est
    publish_hz        float  default 10.0

Usage:
    ros2 run sim2real icm_lstm_inference_node \\
        --ros-args -p checkpoint_path:=/path/to/best_agent.pt
"""

import sys
sys.path.insert(0, "/home/karun/venvs/rl/lib/python3.12/site-packages")

import math
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point


# =============================================================================
# ARCHITECTURE CONSTANTS  — must match training exactly
# =============================================================================
OBS_T       = 3
OBS_H       = 64
OBS_W       = 80
OBS_C       = 1
OBS_FLAT    = OBS_T * OBS_H * OBS_W * OBS_C   # 15360
LSTM_HIDDEN = 256
LSTM_IN_DIM = 3                                 # [vx, yaw_rate, icm_r]
AUG_OBS_DIM = OBS_FLAT + LSTM_HIDDEN           # 15616


# =============================================================================
# DEPTH CNN  (identical to training)
# =============================================================================

class DepthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(OBS_C, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.net(
                torch.zeros(1, OBS_C, OBS_H, OBS_W)).shape[1]

    def forward(self, x):
        return self.net(x)


# =============================================================================
# POLICY MODEL  (identical to training — feedforward from SKRL's perspective)
# =============================================================================

class IrisICMLSTMModel(nn.Module):
    """
    Standalone inference model. Does NOT inherit SKRL mixins — those are
    only needed during training. At deployment we call forward() directly.

    aug_obs: (1, AUG_OBS_DIM) = concat(depth_flat, h_t)
    Returns: action (1, 2) — deterministic mean [vx_norm, yaw_norm]
    """

    def __init__(self):
        super().__init__()

        self.cnn      = DepthCNN()
        cnn_total     = OBS_T * self.cnn.out_dim

        self.net = nn.Sequential(
            nn.Linear(cnn_total + LSTM_HIDDEN, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.policy_mean = nn.Linear(256, 2)
        self.log_std     = nn.Parameter(torch.zeros(2))   # kept for ckpt compat
        self.value_head  = nn.Linear(256, 1)              # kept for ckpt compat
        self.pose_head   = nn.Linear(LSTM_HIDDEN, 2)      # LSTM → XY estimate

    def forward(self, aug_obs: torch.Tensor) -> torch.Tensor:
        """aug_obs: (1, AUG_OBS_DIM) → action (1, 2)"""
        depth_flat = aug_obs[:, :OBS_FLAT]                # (1, OBS_FLAT)
        h_t        = aug_obs[:, OBS_FLAT:]                # (1, LSTM_HIDDEN)

        obs  = depth_flat.reshape(1, OBS_T, OBS_H, OBS_W, OBS_C)
        feats = []
        for t in range(OBS_T):
            frame = obs[:, t].permute(0, 3, 1, 2)        # (1, C, H, W)
            feats.append(self.cnn(frame))
        cnn_out = torch.cat(feats, dim=1)                 # (1, T*cnn_out)

        shared = self.net(torch.cat([cnn_out, h_t], dim=-1))
        return self.policy_mean(shared)                   # (1, 2)

    def estimate_pose(self, h_t: torch.Tensor) -> tuple[float, float]:
        """Project LSTM hidden state → (x, y) displacement in metres."""
        xy = self.pose_head(h_t)
        return float(xy[0, 0]), float(xy[0, 1])


# =============================================================================
# TRAJECTORY LSTM  (identical to training — stateless at each step)
# =============================================================================

class TrajectoryLSTM(nn.Module):
    """
    Maintains LSTM hidden state across deployment steps.
    Input: [vx_prev, yaw_prev, icm_r=0.0]
    Output: h_t (1, LSTM_HIDDEN)
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(
            input_size=LSTM_IN_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        # Hidden state: (num_layers=1, batch=1, hidden)
        self.h = torch.zeros(1, 1, LSTM_HIDDEN, device=device)
        self.c = torch.zeros(1, 1, LSTM_HIDDEN, device=device)

    def reset(self):
        """Zero hidden and cell state. Call at episode/flight start."""
        self.h.zero_()
        self.c.zero_()

    def step(self, lstm_input: torch.Tensor) -> torch.Tensor:
        """
        lstm_input: (1, 3) — [vx_prev, yaw_prev, icm_r]
        Returns h_t: (1, LSTM_HIDDEN)
        """
        x = lstm_input.unsqueeze(1)              # (1, 1, 3)
        _, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        self.h = self.h.detach()
        self.c = self.c.detach()
        return self.h.squeeze(0)                 # (1, LSTM_HIDDEN)


# =============================================================================
# CHECKPOINT LOADER
# =============================================================================

def load_checkpoint(path: str, device: torch.device
                    ) -> tuple[IrisICMLSTMModel, TrajectoryLSTM]:
    """
    Load a training checkpoint into standalone inference classes.

    The checkpoint was saved by SKRL and contains the full state dict
    of IrisICMLSTMModel (which inherits SKRL Model + GaussianMixin).
    We load with strict=False so SKRL-specific keys that don't exist in
    our standalone model are silently ignored.
    """
    model = IrisICMLSTMModel().to(device)
    lstm  = TrajectoryLSTM(device=device).to(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # SKRL saves weights under "policy" key; handle raw state_dicts too
    if isinstance(ckpt, dict) and "policy" in ckpt:
        sd = ckpt["policy"]
    else:
        sd = ckpt

    # Strip DataParallel prefix if present
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Copy LSTM weights from policy model state dict into TrajectoryLSTM
    # The training model stored LSTM weights under "lstm.*" keys
    lstm_sd = {k.replace("lstm.", ""): v
               for k, v in sd.items() if k.startswith("lstm.")}
    if lstm_sd:
        m, u = lstm.lstm.load_state_dict(lstm_sd, strict=False)
        if m:
            print(f"[load] LSTM missing keys: {m}")
    else:
        print("[load] WARNING: no 'lstm.*' keys found in checkpoint — "
              "LSTM weights not loaded. Check checkpoint structure.")

    # Report any unexpected mismatches (not errors — strict=False)
    skrl_only = [k for k in unexpected if any(
        k.startswith(p) for p in ("log_std", "value_head", "policy_mean"))]
    other_unexpected = [k for k in unexpected if k not in skrl_only]
    if other_unexpected:
        print(f"[load] unexpected keys (non-SKRL): {other_unexpected}")
    if missing:
        print(f"[load] missing keys: {missing}")

    model.eval()
    lstm.eval()
    print(f"[load] Checkpoint loaded from {path}  device={device}")
    return model, lstm


# =============================================================================
# INFERENCE NODE
# =============================================================================

class ICMLSTMInferenceNode(Node):

    def __init__(self):
        super().__init__("icm_lstm_inference_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("max_depth_m",     6.0)
        self.declare_parameter("min_depth_m",     0.2)
        self.declare_parameter("depth_topic",     "/m2h/depth/image")
        self.declare_parameter("action_topic",    "/uav/action_cmd")
        self.declare_parameter("pose_topic",      "/uav/lstm_pose_est")
        self.declare_parameter("publish_hz",      10.0)

        ckpt_path    = self.get_parameter("checkpoint_path").value
        self._dmax   = float(self.get_parameter("max_depth_m").value)
        self._dmin   = float(self.get_parameter("min_depth_m").value)
        self._hz     = float(self.get_parameter("publish_hz").value)
        depth_topic  = self.get_parameter("depth_topic").value
        action_topic = self.get_parameter("action_topic").value
        pose_topic   = self.get_parameter("pose_topic").value

        if not ckpt_path:
            raise ValueError(
                "checkpoint_path is empty.\n"
                "Pass: --ros-args -p "
                "checkpoint_path:=/path/to/best_agent.pt")

        # ── Device ────────────────────────────────────────────────────────────
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Inference device: {self._device}")

        # ── Model ─────────────────────────────────────────────────────────────
        self.get_logger().info(f"Loading checkpoint: {ckpt_path}")
        self._model, self._lstm = load_checkpoint(ckpt_path, self._device)
        self.get_logger().info("Model ready.")

        # ── Depth history: (1, T, H, W, C)  init to 0.5 (neutral) ────────────
        self._depth_hist = torch.full(
            (1, OBS_T, OBS_H, OBS_W, OBS_C), 0.5,
            dtype=torch.float32, device=self._device)
        self._hist_lock = threading.Lock()

        # ── Previous action for LSTM input ────────────────────────────────────
        # icm_r is always 0.0 at deployment (ICM not run, see module docstring)
        self._prev_vx  = 0.0
        self._prev_yaw = 0.0
        self._action_lock = threading.Lock()

        # ── Latest computed action (timer publishes this) ─────────────────────
        self._last_vx  = 0.0
        self._last_yaw = 0.0

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_action = self.create_publisher(Twist, action_topic, 10)
        self._pub_pose   = self.create_publisher(Point, pose_topic,   10)

        # ── Subscriber ────────────────────────────────────────────────────────
        self.create_subscription(
            Image, depth_topic, self._depth_cb, qos_profile_sensor_data)

        # ── Publish timer ─────────────────────────────────────────────────────
        self.create_timer(1.0 / self._hz, self._publish_action)

        self.get_logger().info(
            f"\nICM+LSTM inference node ready\n"
            f"  depth  ← {depth_topic}\n"
            f"  action → {action_topic}  (Twist, normalised [-1,1])\n"
            f"  pose   → {pose_topic}    (Point, metres from start)\n"
            f"  rate   : {self._hz} Hz\n"
            f"  device : {self._device}"
        )

    # ── Depth callback — runs at M2H-MX rate (~10 Hz) ─────────────────────────

    def _depth_cb(self, msg: Image):

        # 1. Decode
        depth_m = self._decode_depth(msg)
        if depth_m is None:
            self.get_logger().warn(
                "Invalid depth frame — skipping.",
                throttle_duration_sec=2.0)
            return

        # 2. Resize + normalise
        if depth_m.shape[:2] != (OBS_H, OBS_W):
            depth_m = cv2.resize(
                depth_m, (OBS_W, OBS_H),
                interpolation=cv2.INTER_NEAREST)

        d = np.clip((depth_m - self._dmin) / (self._dmax - self._dmin),
                    0.0, 1.0).astype(np.float32)
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)

        frame = torch.tensor(d, device=self._device)\
                     .unsqueeze(0).unsqueeze(-1)    # (1, H, W, 1)

        # 3. Roll history
        with self._hist_lock:
            self._depth_hist = torch.cat(
                [self._depth_hist[:, 1:], frame.unsqueeze(1)], dim=1)
            depth_hist = self._depth_hist.clone()

        # 4. LSTM input: [vx_prev, yaw_prev, 0.0]
        with self._action_lock:
            vx_p, yaw_p = self._prev_vx, self._prev_yaw

        lstm_in = torch.tensor(
            [[vx_p, yaw_p, 0.0]],
            dtype=torch.float32, device=self._device)

        # 5. Step LSTM
        with torch.no_grad():
            h_t = self._lstm.step(lstm_in)           # (1, LSTM_HIDDEN)

        # 6. Augmented observation
        depth_flat = depth_hist.reshape(1, OBS_FLAT)
        aug_obs    = torch.cat([depth_flat, h_t], dim=-1)  # (1, AUG_OBS_DIM)

        # 7. Policy forward
        with torch.no_grad():
            action = self._model(aug_obs)             # (1, 2)

        vx  = float(action[0, 0].clamp(-1.0, 1.0))
        yaw = float(action[0, 1].clamp(-1.0, 1.0))

        # 8. Pose estimate
        with torch.no_grad():
            px, py = self._model.estimate_pose(h_t)

        # 9. Store
        with self._action_lock:
            self._last_vx  = vx
            self._last_yaw = yaw
            self._prev_vx  = vx
            self._prev_yaw = yaw

        # Publish pose immediately
        pt = Point(); pt.x = px; pt.y = py; pt.z = 0.0
        self._pub_pose.publish(pt)

        self.get_logger().debug(
            f"vx={vx:+.3f}  yaw={yaw:+.3f}  "
            f"pose=({px:.2f}, {py:.2f})")

    # ── Depth decoder ─────────────────────────────────────────────────────────

    def _decode_depth(self, msg: Image) -> np.ndarray | None:
        try:
            if msg.encoding == "32FC1":
                d = np.frombuffer(msg.data, dtype=np.float32)\
                      .reshape(msg.height, msg.width).copy()
            elif msg.encoding == "16UC1":
                d = np.frombuffer(msg.data, dtype=np.uint16)\
                      .reshape(msg.height, msg.width)\
                      .astype(np.float32) / 1000.0
            else:
                from cv_bridge import CvBridge
                d = CvBridge()\
                      .imgmsg_to_cv2(msg, desired_encoding="32FC1")\
                      .astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Depth decode error: {e}")
            return None
        d = np.where(np.isfinite(d), d, self._dmax)
        return d.astype(np.float32)

    # ── Action publish timer ───────────────────────────────────────────────────

    def _publish_action(self):
        with self._action_lock:
            vx, yaw = self._last_vx, self._last_yaw
        cmd           = Twist()
        cmd.linear.x  = vx
        cmd.angular.z = yaw
        self._pub_action.publish(cmd)

    # ── Episode reset (call between flights) ──────────────────────────────────

    def reset_episode(self):
        with self._hist_lock:
            self._depth_hist.fill_(0.5)
        with self._action_lock:
            self._prev_vx = self._prev_yaw = 0.0
            self._last_vx = self._last_yaw = 0.0
        self._lstm.reset()
        self.get_logger().info("Episode reset: LSTM + depth history cleared.")


# =============================================================================
# MAIN
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ICMLSTMInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()