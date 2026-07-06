#!/usr/bin/env python3
"""
iris_icm_inference_node.py
===========================
ROS 2 inference node for the Isaac-Lab-trained Iris ICM exploration policy.

Pure depth-driven version: no pose/SLAM input, no visited-heatmap. The policy
was trained with a 2-channel observation (depth + heatmap); here channel 1
is just fed as zeros (i.e. "nothing visited yet" at every step) so the
checkpoint loads and runs unmodified — exploration drive comes from the ICM
intrinsic reward baked into the policy via depth novelty, the heatmap was
only a secondary anti-backtracking signal.

Subscribes:
    depth image topic   (sensor_msgs/Image)   - real depth camera (e.g. RealSense)

Publishes:
    uav action topic     (geometry_msgs/Twist) - NORMALISED command in [-1, 1]
        linear.x  = forward velocity command  (multiply by max_forward_vel downstream)
        angular.z = yaw rate command           (multiply by max_yaw_rate downstream)

Per-step pipeline (mirrors _get_observations()/_pre_physics_step() in the
training env, minus the heatmap):
    1. depth -> clamp[cam_min,cam_max] -> normalise to [0,1]   (channel 0)
    2. zeros                                                    (channel 1)
    3. stack last T=3 (depth, zeros) frames                     -> (T,H,W,2)
    4. run PPO policy mean action, apply EMA action smoothing
       (alpha=0.6, same as training), clip to [-1, 1]
    5. publish normalised (vx, yaw) — actual m/s and rad/s scaling
       happens in your PX4 offboard control node.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

try:
    from cv_bridge import CvBridge
    import cv2
except ImportError as e:
    raise ImportError(
        "This node needs cv_bridge and opencv-python. "
        "Install with: sudo apt install ros-<distro>-cv-bridge && pip install opencv-python"
    ) from e



# CONSTANTS, make sire it must match training (iris_icm_office_agent.py / iris_icm_exploration.py)


CAM_H, CAM_W = 64, 80
N_CH         = 2   # checkpoint was trained with 2 channels; channel 1 fed as zeros here
HIST_LEN     = 3
CAM_MIN_DEPTH, CAM_MAX_DEPTH = 0.2, 6.0

ACTION_ALPHA = 0.6   # EMA smoothing factor used by the training env



# MODEL — same architecture as IrisICMOfficeModel, stripped of SKRL mixins so it
# can be loaded and run standalone. Only the policy mean is needed for inference.

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
        self.value_head = nn.Linear(256, 1)  # unused at inference, kept for ckpt key match

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (1, T, H, W, C) in [0,1] -> returns (1, action_dim) mean action."""
        feats = []
        for t in range(self.t_steps):
            frame = obs[:, t].permute(0, 3, 1, 2)  # (N,C,H,W)
            feats.append(self.cnn(frame))
        shared = self.net(torch.cat(feats, dim=1))
        mean = self.policy_mean(shared)
        return mean


def load_policy(checkpoint_path: str, device: torch.device) -> IrisICMPolicyNet:
    model = IrisICMPolicyNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

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



# ROS 2 NODE

class IrisICMInferenceNode(Node):
    def __init__(self):
        super().__init__("iris_icm_inference_node")

        # ---- parameters ----
        self.declare_parameter("checkpoint_path", "/home/user/ros2_jazzy/src/sim2real/trained_models/icm_best_agent.pt")
        self.declare_parameter("depth_topic", "/m2h/depth/image")
        self.declare_parameter("action_topic", "/uav/action_cmd")
        self.declare_parameter("inference_rate_hz", 20.0)
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path    = self.get_parameter("checkpoint_path").value
        depth_topic  = self.get_parameter("depth_topic").value
        action_topic = self.get_parameter("action_topic").value
        rate_hz      = float(self.get_parameter("inference_rate_hz").value)
        device_str   = self.get_parameter("device").value

        self.device = torch.device(device_str)
        self.get_logger().info(f"Loading policy checkpoint: {ckpt_path} on {self.device}")
        self.policy = load_policy(ckpt_path, self.device)
        self.get_logger().info("Policy loaded.")

        self.bridge = CvBridge()

        # ---- runtime state ----
        self._latest_depth = np.full((CAM_H, CAM_W), 0.5, dtype=np.float32)  # normalised [0,1]
        self._have_depth = False

        self._zero_channel = np.zeros((CAM_H, CAM_W), dtype=np.float32)

        self._frame_hist = deque(maxlen=HIST_LEN)
        for _ in range(HIST_LEN):
            self._frame_hist.append(np.stack(
                [np.full((CAM_H, CAM_W), 0.5, dtype=np.float32), self._zero_channel], axis=-1))

        self._smooth_action = np.zeros(2, dtype=np.float32)

        # ---- pub/sub ----
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        self.pub_action = self.create_publisher(Twist, action_topic, 10)

        self.timer = self.create_timer(1.0 / rate_hz, self._step)

        self.get_logger().info(
            f"Subscribed depth={depth_topic} -> publishing {action_topic} at {rate_hz} Hz"
        )

    # ------------------------------------------------------------------
    def _on_depth(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"depth conversion failed: {e}")
            return

        depth_m = cv_img.astype(np.float32)
        # 16UC1 depth images (RealSense default) are in millimetres.
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
    def _step(self):
        if not self._have_depth:
            return  # wait for first depth frame before publishing anything

        frame = np.stack([self._latest_depth, self._zero_channel], axis=-1)  # (H,W,2)
        self._frame_hist.append(frame)

        obs = np.stack(list(self._frame_hist), axis=0)  # (T,H,W,2)
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)  # (1,T,H,W,2)

        mean_action = self.policy.act(obs_t).squeeze(0).cpu().numpy()
        raw = np.clip(mean_action, -1.0, 1.0)

        # Reproduce the same EMA smoothing the training env applied before
        # converting actions to velocity setpoints.
        self._smooth_action = (ACTION_ALPHA * self._smooth_action +
                                (1.0 - ACTION_ALPHA) * raw)

        vx_norm, yaw_norm = self._smooth_action.tolist()

        msg = Twist()
        msg.linear.x = float(vx_norm)
        msg.angular.z = float(yaw_norm)
        self.pub_action.publish(msg)


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