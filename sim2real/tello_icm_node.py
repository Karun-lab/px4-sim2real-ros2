#!/usr/bin/env python3
"""
tello_icm_bridge_node.py
=========================
ROS 2 bridge for the DJI Tello — ICM exploration control + sensor publishing.

Publishes:
    /tello_stream   (sensor_msgs/Image)    - BGR8 frames at stream_w × stream_h
    /imu            (sensor_msgs/Imu)      - Tello IMU data, same timestamp as image

Subscribes:
    /uav/action_cmd (geometry_msgs/Twist)  - normalised ICM commands
        linear.x  in [-1, 1]  -> forward velocity
        angular.z in [-1, 1]  -> yaw rate

IMPORTANT — Tello IMU limitations
-----------------------------------
The Tello SDK broadcasts state (pitch/roll/yaw, accelerations) over UDP at ~10 Hz.
This node publishes IMU at stream_fps (default 20 Hz), so readings between Tello
state updates will be duplicates. For production VIO you need raw IMU — this is
suitable for pipeline development and slow-motion testing.

IMU data available from Tello SDK
-----------------------------------
    agx / agy / agz  : accelerometer  in 0.001 g  → converted to m/s²
    pitch / roll / yaw: Euler angles   in degrees  → converted to quaternion
    angular velocity  : NOT directly available — computed via finite difference
                        of Euler angles. Noisy; treat angular_velocity_covariance
                        accordingly.

Coordinate frame (REP-103 / ROS standard: body frame, right-hand)
-------------------------------------------------------------------
    Tello body:   X=forward  Y=left  Z=up
    sensor_msgs/Imu expects:
        linear_acceleration in body frame (X forward, Y left, Z up)
        angular_velocity    in body frame (same)
        orientation         relative to a fixed world frame

Parameters (all tunable via --ros-args -p)
------------------------------------------
    max_forward_cm_s    int    default 40
    max_yaw_deg_s       int    default 40
    max_updown_cm_s     int    default 0
    cmd_timeout_s       float  default 0.5
    stream_fps          float  default 20.0   (video AND IMU publish rate)
    stream_w            int    default 320
    stream_h            int    default 240
    takeoff_on_start    bool   default True
    video_topic         str    default /tello_stream
    imu_topic           str    default /imu
    action_topic        str    default /uav/action_cmd

Dependencies:
    pip install djitellopy opencv-python
"""

import math
import time
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist, Vector3, Quaternion
from std_msgs.msg import Header

try:
    from djitellopy import Tello
except ImportError as e:
    raise ImportError("Run: pip install djitellopy") from e


STREAM_WARMUP_S = 2.0
G_TO_MS2        = 9.80665   # 1 g in m/s²


# ── Helpers ────────────────────────────────────────────────────────────────────

def rpy_to_quaternion(roll_rad: float, pitch_rad: float,
                      yaw_rad: float) -> Quaternion:
    """ZYX Euler angles → unit quaternion (ROS convention: x,y,z,w)."""
    cr, sr = math.cos(roll_rad  * 0.5), math.sin(roll_rad  * 0.5)
    cp, sp = math.cos(pitch_rad * 0.5), math.sin(pitch_rad * 0.5)
    cy, sy = math.cos(yaw_rad   * 0.5), math.sin(yaw_rad   * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def angle_diff(a_rad: float, b_rad: float) -> float:
    """Shortest signed difference a - b, wrapped to (-π, π]."""
    d = a_rad - b_rad
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d


# ── Node ───────────────────────────────────────────────────────────────────────

class TelloICMBridgeNode(Node):
    def __init__(self):
        super().__init__("tello_icm_bridge_node")

        # ── Parameters ──────────────────────────────────────────────────────────
        self.declare_parameter("max_forward_cm_s",  40)
        self.declare_parameter("max_yaw_deg_s",     40)
        self.declare_parameter("max_updown_cm_s",   0)
        self.declare_parameter("cmd_timeout_s",     0.5)
        self.declare_parameter("stream_fps",        20.0)
        self.declare_parameter("stream_w",          320)
        self.declare_parameter("stream_h",          240)
        self.declare_parameter("takeoff_on_start",  True)
        self.declare_parameter("video_topic",       "/tello_stream")
        self.declare_parameter("imu_topic",         "/imu")
        self.declare_parameter("action_topic",      "/uav/action_cmd")

        self._max_fwd   = int(self.get_parameter("max_forward_cm_s").value)
        self._max_yaw   = int(self.get_parameter("max_yaw_deg_s").value)
        self._max_ud    = int(self.get_parameter("max_updown_cm_s").value)
        self._cmd_to    = float(self.get_parameter("cmd_timeout_s").value)
        self._fps       = float(self.get_parameter("stream_fps").value)
        self._stream_w  = int(self.get_parameter("stream_w").value)
        self._stream_h  = int(self.get_parameter("stream_h").value)
        self._auto_to   = bool(self.get_parameter("takeoff_on_start").value)
        video_topic     = self.get_parameter("video_topic").value
        imu_topic       = self.get_parameter("imu_topic").value
        action_topic    = self.get_parameter("action_topic").value

        # ── Tello connection ─────────────────────────────────────────────────────
        self.tello = Tello()
        self.tello.connect()
        bat = self.tello.get_battery()
        self.get_logger().info(f"Tello connected. Battery: {bat}%")
        if bat < 15:
            self.get_logger().error("Battery too low (<15%). Aborting.")
            raise RuntimeError("Low battery — will not take off.")

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

        # ── Control state ────────────────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._last_cmd_time = time.time()
        self._vx_norm       = 0.0
        self._yaw_norm      = 0.0

        # ── IMU state ────────────────────────────────────────────────────────────
        # Previous Euler angles for finite-difference angular velocity.
        # Initialised to current reading so first derivative is zero.
        self._prev_roll_rad  = 0.0
        self._prev_pitch_rad = 0.0
        self._prev_yaw_rad   = 0.0
        self._prev_imu_time  = time.time()

        # ── Covariance matrices (row-major 3×3, stored as list of 9 floats) ──────
        #
        # orientation: derived from Euler angles, ≈ 2° (0.035 rad) error typical
        _or_var = (0.035) ** 2          # ~2° std-dev → variance
        self._orient_cov = [_or_var, 0.0, 0.0,
                            0.0, _or_var, 0.0,
                            0.0, 0.0, _or_var]

        # angular_velocity: finite-difference of ~10 Hz Euler angles published at
        # 20 Hz — expect significant noise. Set high diagonal variance.
        _av_var = (0.1) ** 2            # ~0.1 rad/s std-dev
        self._angvel_cov = [_av_var, 0.0, 0.0,
                            0.0, _av_var, 0.0,
                            0.0, 0.0, _av_var]

        # linear_acceleration: Tello MEMS accelerometer, ≈ 0.1 m/s² noise floor
        _la_var = (0.1) ** 2
        self._linacc_cov = [_la_var, 0.0, 0.0,
                            0.0, _la_var, 0.0,
                            0.0, 0.0, _la_var]

        # ── ROS interfaces ───────────────────────────────────────────────────────
        self.pub_video = self.create_publisher(Image, video_topic, 10)
        self.pub_imu   = self.create_publisher(Imu,   imu_topic,   10)

        self.sub_action = self.create_subscription(
            Twist, action_topic, self._on_action, qos_profile_sensor_data)

        # Single timer fires both _publish_frame and _publish_imu with the
        # SAME ROS timestamp, which is required for VIO synchronisation.
        self.ctrl_timer    = self.create_timer(0.05,           self._control_loop)
        self.sensors_timer = self.create_timer(1.0 / self._fps, self._publish_sensors)

        self.get_logger().info(
            f"Bridge ready\n"
            f"  video → {video_topic} @ {self._stream_w}×{self._stream_h} {self._fps:.0f}Hz\n"
            f"  imu   → {imu_topic}   @ {self._fps:.0f}Hz  "
            f"(Tello SDK state ~10Hz — readings repeat between updates)\n"
            f"  actions ← {action_topic}"
        )

    # ── Action subscriber ────────────────────────────────────────────────────────
    def _on_action(self, msg: Twist):
        with self._lock:
            self._vx_norm       = float(np.clip(msg.linear.x,  -0.2, 1.0))
            self._yaw_norm      = float(np.clip(msg.angular.z, -1.0, 1.0))
            self._last_cmd_time = time.time()

    # ── Control loop ─────────────────────────────────────────────────────────────
    def _control_loop(self):
        with self._lock:
            vx_n  = self._vx_norm
            yaw_n = self._yaw_norm
            vx_n  = float(np.clip(vx_n  * 1.4, -0.2, 1.0))
            yaw_n = float(np.clip(yaw_n * 0.7, -1.0, 1.0))
            fresh = (time.time() - self._last_cmd_time) < self._cmd_to

        if not fresh:
            self.tello.send_rc_control(0, 0, 0, 0)
            return

        if vx_n > 0.7:
            yaw_n = 0.0
        if abs(yaw_n) < 0.4:
            yaw_n = 0.0
        if abs(yaw_n) < 0.7 and vx_n < 0.2:
            vx_n = 0.2

        fwd = int(np.clip(vx_n  * self._max_fwd, -100, 100))
        yaw = int(np.clip(yaw_n * self._max_yaw, -100, 100))
        ud  = int(np.clip(self._max_ud,           -100, 100))
        self.tello.send_rc_control(0, fwd, ud, yaw)

    # ── Sensor publisher (video + IMU at identical timestamp) ────────────────────
    def _publish_sensors(self):
        """
        Called at stream_fps Hz. Generates one shared ROS timestamp and passes
        it to both _publish_frame and _publish_imu so image and IMU messages
        are time-aligned for VIO.
        """
        stamp = self.get_clock().now().to_msg()
        self._publish_frame(stamp)
        self._publish_imu(stamp)

    # ── Video ─────────────────────────────────────────────────────────────────────
    def _normalize_frame(self, frame) -> np.ndarray | None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        if frame.shape[1] != self._stream_w or frame.shape[0] != self._stream_h:
            frame = cv2.resize(
                frame, (self._stream_w, self._stream_h),
                interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(frame)

    def _publish_frame(self, stamp):
        raw   = self._frame_reader.frame
        frame = self._normalize_frame(raw)
        if frame is None:
            return

        msg = Image()
        msg.header.stamp    = stamp
        msg.header.frame_id = "tello_camera"
        msg.height          = frame.shape[0]
        msg.width           = frame.shape[1]
        msg.encoding        = "bgr8"
        msg.is_bigendian    = False
        msg.step            = frame.shape[1] * 3
        msg.data            = frame.tobytes()
        self.pub_video.publish(msg)

    # ── IMU ──────────────────────────────────────────────────────────────────────
    def _publish_imu(self, stamp):
        """
        Build and publish sensor_msgs/Imu from Tello SDK state.

        Tello body frame (right-hand, REP-103 compatible):
            X = forward   (pitch axis)
            Y = left      (roll axis)
            Z = up

        Acceleration mapping (Tello → ROS body frame):
            agx → linear_acceleration.x  (forward, +forward = nose pull-up)
            agy → linear_acceleration.y  (left,    +left    = left roll)
            agz → linear_acceleration.z  (up,      includes +1g when hovering)

        Note: Tello agz reads ≈ +1000 (0.001g units = +1g) when stationary
        because the accelerometer measures specific force (reaction to gravity).
        This is the standard for IMUs used in VIO.
        """
        try:
            # ── Read Tello state ────────────────────────────────────────────────
            # Use get_current_state() for atomic snapshot of all fields
            state = self.tello.get_current_state()

            # Euler angles (degrees → radians)
            roll_rad  = math.radians(float(state.get("roll",  0)))
            pitch_rad = math.radians(float(state.get("pitch", 0)))
            yaw_rad   = math.radians(float(state.get("yaw",   0)))

            # Accelerations: 0.001 g → m/s²
            # agx/agy/agz from state dict (fallback 0 if missing)
            ax = float(state.get("agx", 0.0)) * 0.001 * G_TO_MS2
            ay = float(state.get("agy", 0.0)) * 0.001 * G_TO_MS2
            az = float(state.get("agz", 0.0)) * 0.001 * G_TO_MS2

        except Exception as e:
            self.get_logger().warn(
                f"IMU state read failed: {e}", throttle_duration_sec=5.0)
            return

        # ── Angular velocity via finite difference ──────────────────────────────
        now = time.time()
        dt  = now - self._prev_imu_time

        if dt > 0.001:   # guard against zero division at first call
            wx = angle_diff(roll_rad,  self._prev_roll_rad)  / dt
            wy = angle_diff(pitch_rad, self._prev_pitch_rad) / dt
            wz = angle_diff(yaw_rad,   self._prev_yaw_rad)   / dt
        else:
            wx = wy = wz = 0.0

        self._prev_roll_rad  = roll_rad
        self._prev_pitch_rad = pitch_rad
        self._prev_yaw_rad   = yaw_rad
        self._prev_imu_time  = now

        # ── Orientation quaternion ──────────────────────────────────────────────
        q = rpy_to_quaternion(roll_rad, pitch_rad, yaw_rad)

        # ── Build message ───────────────────────────────────────────────────────
        msg = Imu()
        msg.header.stamp    = stamp
        msg.header.frame_id = "tello_imu"

        msg.orientation   = q
        msg.orientation_covariance = self._orient_cov

        msg.angular_velocity.x = wx
        msg.angular_velocity.y = wy
        msg.angular_velocity.z = wz
        msg.angular_velocity_covariance = self._angvel_cov

        # Tello agx/agy/agz → ROS body frame (X forward, Y left, Z up)
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        msg.linear_acceleration_covariance = self._linacc_cov

        self.pub_imu.publish(msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────────
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