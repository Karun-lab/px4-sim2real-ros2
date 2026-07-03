#!/usr/bin/env python3
"""
rl_inference_node.py
====================
ROS 2 inference node for Iris drone RL policies.
Supports:
  - Ball-tracking policy  (RGB camera   → 4-channel obs)


Set TASK = "ball" or "task = "door" below, or pass as a ROS parameter.

Published topic : /rl_action  (Float32MultiArray, [vx, yaw_rate])
Subscribed topic: /camera/rgb/image_raw   (ball task)
                  /camera/depth/image_raw (door task)
"""

import sys
sys.path.insert(0, "/home/karun/venvs/rl/lib/python3.12/site-packages")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import torch
import cv2
import numpy as np

from sim2real.ball_tracking_inference import (
    detect_yellow,
    detect_opening,
    load_policy,
    load_door_policy,
    build_obs_history,
    run_policy,
)


# =============================================================================
# CONFIG  —  change TASK here or via ROS parameter "task"
# =============================================================================
DEFAULT_TASK      = "ball"          # "ball"
BALL_CHECKPOINT   = "/home/karun/ros2_jazzy/src/sim2real/sim2real/trained_models/ball_tracking_best_agent.pt"
DEPTH_CLIP        = 1.5            # metres — must match training cfg
HISTORY_LEN       = 3
IMG_H = IMG_W     = 64
CONTROL_HZ        = 50


# =============================================================================
# NODE
# =============================================================================

class RLInferenceNode(Node):

    def __init__(self):
        super().__init__("rl_inference_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("task",            DEFAULT_TASK)
        self.declare_parameter("ball_checkpoint", BALL_CHECKPOINT)


        task       = self.get_parameter("task").value
        ball_ckpt  = self.get_parameter("ball_checkpoint").value


        self.task  = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Policy ────────────────────────────────────────────────────────────
        if task == "ball":
            self.get_logger().info(f"Loading BALL policy from {ball_ckpt}")
            self.model = load_policy(ball_ckpt, self.device)
        else:
            raise ValueError(f"Unknown task '{task}'. Choose 'ball' or 'door'.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.history     = None
        self.bridge      = CvBridge()
        self.last_action = [0.0, 0.0]

        # ── Subscriptions ─────────────────────────────────────────────────────
        if task == "ball":
            self.create_subscription(
                Image, "/camera/rgb/image_raw", self._ball_camera_cb, 10)
        else:
            self.create_subscription(
                Image, "/camera/depth/image_raw", self._door_camera_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.action_pub = self.create_publisher(Float32MultiArray, "/rl_action", 10)

        # ── Control timer ─────────────────────────────────────────────────────
        self.create_timer(1.0 / CONTROL_HZ, self._timer_cb)

        self.get_logger().info(
            f"RL inference node ready — task={task}, device={self.device}")

    # ── Ball camera callback ───────────────────────────────────────────────────
    def _ball_camera_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        cv_img = cv2.resize(cv_img, (IMG_W, IMG_H),
                            interpolation=cv2.INTER_LINEAR)

        # (H, W, 3) uint8 → float32 [0, 1]
        rgb = torch.tensor(cv_img, dtype=torch.float32,
                           device=self.device) / 255.0   # (64, 64, 3)

        # Build search_active channel
        rgb_b = rgb.unsqueeze(0)                          # (1, 64, 64, 3)
        visible, _, _ = detect_yellow(rgb_b, threshold=0.002)
        search_ch = (~visible).float().view(1, 1, 1, 1).expand(1, IMG_H, IMG_W, 1)
        frame = torch.cat([rgb_b, search_ch], dim=-1)    # (1, 64, 64, 4)

        self.history    = build_obs_history(frame, self.history, HISTORY_LEN)
        self.last_action = run_policy(self.model, self.history)

    # ── Door (depth) camera callback ──────────────────────────────────────────
    def _door_camera_cb(self, msg: Image):
        # Depth images arrive as 32FC1 (float32, metres) or 16UC1 (uint16, mm)
        try:
            if msg.encoding == "16UC1":
                cv_img = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000.0
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f"cv_bridge depth error: {e}")
            return

        cv_img = cv2.resize(cv_img, (IMG_W, IMG_H),
                            interpolation=cv2.INTER_NEAREST)   # nearest for depth

        depth = torch.tensor(cv_img, dtype=torch.float32,
                             device=self.device).unsqueeze(0).unsqueeze(-1)
        # depth: (1, H, W, 1)

        visible, _, _, depth_norm = detect_opening(
            depth, depth_clip=DEPTH_CLIP)
        # depth_norm: (1, H, W)

        search_ch  = (~visible).float().view(1, 1, 1, 1).expand(1, IMG_H, IMG_W, 1)
        depth_ch   = depth_norm.unsqueeze(-1)              # (1, H, W, 1)
        frame      = torch.cat([depth_ch, search_ch], dim=-1)  # (1, 64, 64, 2)

        self.history     = build_obs_history(frame, self.history, HISTORY_LEN)
        self.last_action = run_policy(self.model, self.history)

    # ── Control timer ─────────────────────────────────────────────────────────
    def _timer_cb(self):
        msg      = Float32MultiArray()
        msg.data = self.last_action
        self.action_pub.publish(msg)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = RLInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()