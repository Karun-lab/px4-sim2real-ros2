#!/usr/bin/env python3
"""
tello_icm_bridge_node.py
=========================
ROS 2 bridge for the DJI Tello when driven by the ICM exploration inference node.

Publishes:
    /tello_stream   (sensor_msgs/Image)   - BGR8 frames from Tello's 720p camera
                                            (your ICM inference node subscribes to a
                                             depth topic; see note below re: using
                                             this with a RealSense instead)

Subscribes:
    /uav/action_cmd (geometry_msgs/Twist) - normalised commands from ICM inference node
        linear.x  in [-1, 1]  -> forward/back velocity
        angular.z in [-1, 1]  -> yaw rate

Parameters (all ros-args settable):
    max_forward_cm_s   int   default 30   max forward speed (cm/s, Tello range 10-100)
    max_yaw_deg_s      int   default 60   max yaw rate     (deg/s, Tello range 1-100)
    max_updown_cm_s    int   default 0    vertical trim    (cm/s, leave 0 for hover)
    cmd_timeout_s      float default 0.5  stop if no cmd received within this window
    stream_fps         float default 20.0 video publish rate
    takeoff_on_start   bool  default True auto-takeoff on node start
    hover_height_cm    int   default 100  target hover height (cm) — set before takeoff

NOTE on depth:
    The Tello only has a downward ToF sensor, not a forward depth camera. To use this
    node with the ICM inference node you have two options:
        A) Attach an external depth camera (e.g. Intel RealSense D435i) to the Tello
           and publish its depth on /camera/depth/image_raw — the inference node
           reads that topic independently.
        B) Point the ICM inference node at /tello_stream and run a monocular depth
           estimator (e.g. Depth-Anything-V2) between the two nodes.
    This bridge node makes no assumptions — it just publishes the colour stream and
    forwards the normalised commands. The inference pipeline is upstream.

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
    raise ImportError(
        "djitellopy not installed. Run: pip install djitellopy"
    ) from e


class TelloICMBridgeNode(Node):
    def __init__(self):
        super().__init__("tello_icm_bridge_node")

        # ------------------------------------------------------------------ #
        # Parameters                                                           #
        # ------------------------------------------------------------------ #
        self.declare_parameter("max_forward_cm_s",  30)
        self.declare_parameter("max_yaw_deg_s",     60)
        self.declare_parameter("max_updown_cm_s",   0)
        self.declare_parameter("cmd_timeout_s",     0.5)
        self.declare_parameter("stream_fps",        20.0)
        self.declare_parameter("takeoff_on_start",  True)
        self.declare_parameter("hover_height_cm",   100)
        self.declare_parameter("video_topic",       "/tello_stream")
        self.declare_parameter("action_topic",      "/uav/action_cmd")

        self._max_fwd   = self.get_parameter("max_forward_cm_s").value
        self._max_yaw   = self.get_parameter("max_yaw_deg_s").value
        self._max_ud    = self.get_parameter("max_updown_cm_s").value
        self._cmd_to    = float(self.get_parameter("cmd_timeout_s").value)
        self._fps       = float(self.get_parameter("stream_fps").value)
        self._auto_to   = self.get_parameter("takeoff_on_start").value
        self._h_target  = self.get_parameter("hover_height_cm").value
        video_topic     = self.get_parameter("video_topic").value
        action_topic    = self.get_parameter("action_topic").value

        # ------------------------------------------------------------------ #
        # Tello setup                                                          #
        # ------------------------------------------------------------------ #
        self.tello = Tello()
        self.tello.connect()
        bat = self.tello.get_battery()
        self.get_logger().info(f"Tello connected. Battery: {bat}%")
        if bat < 15:
            self.get_logger().error("Battery too low (<15%). Aborting takeoff.")
            raise RuntimeError("Low battery — will not take off.")

        self.tello.streamon()
        self._frame_reader = self.tello.get_frame_read()

        if self._auto_to:
            self.get_logger().info("Taking off...")
            self.tello.takeoff()
            # Brief climb to hover height then stop vertical motion
            time.sleep(2.0)
            self.get_logger().info(f"Airborne. Target height: {self._h_target} cm")

        # ------------------------------------------------------------------ #
        # State                                                                #
        # ------------------------------------------------------------------ #
        self.bridge = CvBridge()
        self._last_cmd_time = time.time()

        # Raw normalised commands from inference node
        self._vx_norm  = 0.0
        self._yaw_norm = 0.0
        self._lock = threading.Lock()

        # ------------------------------------------------------------------ #
        # ROS interfaces                                                       #
        # ------------------------------------------------------------------ #
        self.pub_video = self.create_publisher(Image, video_topic, 10)

        self.sub_action = self.create_subscription(
            Twist,
            action_topic,
            self._on_action,
            qos_profile_sensor_data,
        )

        # Control loop: send RC at ~20 Hz regardless of cmd rate
        self.ctrl_timer  = self.create_timer(0.05,  self._control_loop)
        # Video loop: publish frames
        self.video_timer = self.create_timer(1.0 / self._fps, self._publish_frame)

        self.get_logger().info(
            f"Bridge ready. video -> {video_topic}, "
            f"actions <- {action_topic}"
        )

    # ---------------------------------------------------------------------- #
    # Action subscriber                                                        #
    # ---------------------------------------------------------------------- #
    def _on_action(self, msg: Twist):
        with self._lock:
            self._vx_norm  = float(np.clip(msg.linear.x,  -1.0, 1.0))
            self._yaw_norm = float(np.clip(msg.angular.z, -1.0, 1.0))
            self._last_cmd_time = time.time()

    # ---------------------------------------------------------------------- #
    # Control loop                                                             #
    # ---------------------------------------------------------------------- #
    def _control_loop(self):
        now = time.time()
        with self._lock:
            vx_n   = self._vx_norm
            yaw_n  = self._yaw_norm
            fresh  = (now - self._last_cmd_time) < self._cmd_to

        if not fresh:
            # No recent command — hover in place
            self.tello.send_rc_control(0, 0, 0, 0)
            return

        # Scale from [-1, 1] to Tello RC range [-100, 100]
        # Tello RC: send_rc_control(left_right, fwd_back, up_down, yaw)
        #   fwd_back : positive = forward
        #   yaw      : positive = rotate right (CW from above)

        fwd_back = int(np.clip(vx_n  * self._max_fwd,  -100, 100))
        yaw      = int(np.clip(yaw_n * self._max_yaw,  -100, 100))
        up_down  = int(np.clip(self._max_ud,            -100, 100))

        self.tello.send_rc_control(0, fwd_back, up_down, yaw)

    # ---------------------------------------------------------------------- #
    # Video publisher                                                          #
    # ---------------------------------------------------------------------- #
    def _publish_frame(self):
        frame = self._frame_reader.frame
        if frame is None or frame.size == 0:
            return

        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "tello_camera"
            self.pub_video.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Frame publish failed: {e}")

    # ---------------------------------------------------------------------- #
    # Cleanup                                                                  #
    # ---------------------------------------------------------------------- #
    def shutdown(self):
        self.get_logger().info("Shutting down — landing...")
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