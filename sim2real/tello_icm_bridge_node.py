#!/usr/bin/env python3
"""
tello_icm_bridge_node.py
=========================
ROS 2 bridge for the DJI Tello when driven by the ICM exploration inference node.

Publishes:
    /tello_stream   (sensor_msgs/Image)   - BGR8 frames, resized to stream_w × stream_h

Subscribes:
    /uav/action_cmd (geometry_msgs/Twist) - normalised commands from ICM inference node
        linear.x  in [-1, 1]  -> forward velocity
        angular.z in [-1, 1]  -> yaw rate

Parameters (all ros-args settable):
    max_forward_cm_s   int   default 30      max forward speed  (cm/s, Tello: 10–100)
    max_yaw_deg_s      int   default 60      max yaw rate       (deg/s, Tello: 1–100)
    max_updown_cm_s    int   default 0       vertical trim      (cm/s, 0 = hold altitude)
    cmd_timeout_s      float default 0.5     hover if no cmd received in this window
    stream_fps         float default 20.0    video publish rate (Hz)
    stream_w           int   default 320     published frame width  (px)
    stream_h           int   default 240     published frame height (px)
    takeoff_on_start   bool  default True    auto-takeoff on node start
    video_topic        str   default /tello_stream
    action_topic       str   default /uav/action_cmd

Dependencies:
    pip install djitellopy opencv-python
    sudo apt install ros-<distro>-cv-bridge
"""

import time
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge

try:
    from djitellopy import Tello
except ImportError as e:
    raise ImportError("Run: pip install djitellopy") from e


# Seconds to wait after streamon() before the decoder thread produces valid frames.
# The H264 decoder needs a few keyframes before it outputs anything usable.
STREAM_WARMUP_S = 2.0


class TelloICMBridgeNode(Node):
    def __init__(self):
        super().__init__("tello_icm_bridge_node")

        # ── Parameters ─────────────────────────────────────────────────────────
        self.declare_parameter("max_forward_cm_s", 30)
        self.declare_parameter("max_yaw_deg_s",    60)
        self.declare_parameter("max_updown_cm_s",  0)
        self.declare_parameter("cmd_timeout_s",    0.5)
        self.declare_parameter("stream_fps",       20.0)
        self.declare_parameter("stream_w",         320)
        self.declare_parameter("stream_h",         240)
        self.declare_parameter("takeoff_on_start", True)
        self.declare_parameter("video_topic",      "/tello_stream")
        self.declare_parameter("action_topic",     "/uav/action_cmd")

        self._max_fwd    = int(self.get_parameter("max_forward_cm_s").value)
        self._max_yaw    = int(self.get_parameter("max_yaw_deg_s").value)
        self._max_ud     = int(self.get_parameter("max_updown_cm_s").value)
        self._cmd_to     = float(self.get_parameter("cmd_timeout_s").value)
        self._fps        = float(self.get_parameter("stream_fps").value)
        self._stream_w   = int(self.get_parameter("stream_w").value)
        self._stream_h   = int(self.get_parameter("stream_h").value)
        self._auto_to    = bool(self.get_parameter("takeoff_on_start").value)
        video_topic      = self.get_parameter("video_topic").value
        action_topic     = self.get_parameter("action_topic").value

        # ── Tello connection ───────────────────────────────────────────────────
        self.tello = Tello()
        self.tello.connect()
        bat = self.tello.get_battery()
        self.get_logger().info(f"Tello connected. Battery: {bat}%")
        if bat < 15:
            self.get_logger().error("Battery too low (<15%). Aborting.")
            raise RuntimeError("Low battery — will not take off.")

        # streamon() must be called before get_frame_read().
        # Then we wait STREAM_WARMUP_S for the H264 decoder to produce
        # valid frames — without this the first frames are None or garbage.
        self.tello.streamon()
        self.get_logger().info(
            f"Stream started. Waiting {STREAM_WARMUP_S}s for decoder warm-up…")
        time.sleep(STREAM_WARMUP_S)
        self._frame_reader = self.tello.get_frame_read()

        if self._auto_to:
            self.get_logger().info("Taking off…")
            self.tello.takeoff()
            time.sleep(2.0)
            self.get_logger().info("Airborne.")

        # ── State ──────────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self._last_cmd_time = time.time()
        self._vx_norm  = 0.0
        self._yaw_norm = 0.0
        self._lock = threading.Lock()

        # ── ROS interfaces ─────────────────────────────────────────────────────
        self.pub_video = self.create_publisher(Image, video_topic, 10)

        self.sub_action = self.create_subscription(
            Twist, action_topic, self._on_action, qos_profile_sensor_data)

        self.ctrl_timer  = self.create_timer(0.05,           self._control_loop)
        self.video_timer = self.create_timer(1.0 / self._fps, self._publish_frame)

        self.get_logger().info(
            f"Bridge ready  |  video → {video_topic} @ "
            f"{self._stream_w}×{self._stream_h} {self._fps:.0f}Hz  |  "
            f"actions ← {action_topic}"
        )

    # ── Action subscriber ──────────────────────────────────────────────────────
    def _on_action(self, msg: Twist):
        with self._lock:
            self._vx_norm       = float(np.clip(msg.linear.x,  -1.0, 1.0))
            self._yaw_norm      = float(np.clip(msg.angular.z, -1.0, 1.0))
            self._last_cmd_time = time.time()

    # ── Control loop ───────────────────────────────────────────────────────────
    def _control_loop(self):
        with self._lock:
            vx_n  = self._vx_norm
            yaw_n = self._yaw_norm
            fresh = (time.time() - self._last_cmd_time) < self._cmd_to

        if not fresh:
            self.tello.send_rc_control(0, 0, 0, 0)
            return

        fwd = int(np.clip(vx_n  * self._max_fwd, -100, 100))
        yaw = int(np.clip(yaw_n * self._max_yaw, -100, 100))
        ud  = int(np.clip(self._max_ud,           -100, 100))

        # send_rc_control(left_right, fwd_back, up_down, yaw)
        self.tello.send_rc_control(0, fwd, ud, yaw)

    # ── Video publisher ─────────────────────────────────────────────────────────
    def _normalize_frame(self, frame) -> np.ndarray | None:
        """
        Robustly convert whatever djitellopy gives us into a contiguous
        BGR uint8 numpy array at the target resolution.

        djitellopy can return:
          - None / empty                → skip
          - BGR  (H, W, 3) uint8        → standard, just resize
          - BGRA (H, W, 4) uint8        → drop alpha, make contiguous
          - Non-contiguous slices       → cv2_to_imgmsg will throw error 16
          - Wrong dtype (float32, etc.) → convert
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Drop alpha channel if present (BGRA → BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Ensure exactly 3 channels
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None

        # Resize to target resolution
        if frame.shape[1] != self._stream_w or frame.shape[0] != self._stream_h:
            frame = cv2.resize(
                frame, (self._stream_w, self._stream_h),
                interpolation=cv2.INTER_LINEAR)

        # cv2_to_imgmsg requires a C-contiguous array — a slice or resize
        # result may not be. This is the root cause of error code 16.
        return np.ascontiguousarray(frame)

    def _publish_frame(self):
        raw = self._frame_reader.frame
        frame = self._normalize_frame(raw)
        if frame is None:
            return
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp    = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "tello_camera"
            self.pub_video.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Frame publish failed: {e}", throttle_duration_sec=5.0)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def shutdown(self):
        self.get_logger().info("Shutting down — landing…")
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.2)
            self.tello.land()
            self.tello.streamoff()
            self.tello.end()
        except Exception as e:
            self.get_logger().warn(f"Shutdown error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TelloICMBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()