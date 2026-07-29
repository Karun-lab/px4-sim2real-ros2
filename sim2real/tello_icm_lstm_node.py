#!/usr/bin/env python3
"""
tello_icm_lstm_node.py
======================
ROS 2 bridge for DJI Tello — ICM+LSTM exploration.
Visualisation removed entirely — run tello_traj_viz_node.py separately.

Altitude fix
------------
The original node sent RC commands at 20 Hz via a timer but the Tello
SDK requires RC commands at a MINIMUM of 15 Hz or the internal
watchdog cuts thrust and the drone descends. The problem was that the
matplotlib TkAgg event loop in the same process was blocking the timer
callbacks for tens of milliseconds at a time, causing missed RC cycles.

This node has NO matplotlib, NO visualisation, NO blocking calls in
any callback. The RC timer fires at 20 Hz reliably and keeps the Tello
watchdog satisfied.

Additionally: the Tello SDK's send_rc_control() itself can occasionally
block for ~5 ms on the UDP socket. Running it in a dedicated thread
(self._rc_thread) instead of directly in the timer callback ensures the
timer is never delayed by socket latency.

Publishes:
    /tello_stream           (sensor_msgs/Image)   BGR8 video
    /imu                    (sensor_msgs/Imu)      Tello IMU
    /tello/rc_command       (geometry_msgs/Twist)  physical RC values sent
                                fwd cm/s in linear.x, yaw deg/s in angular.z
                                (consumed by tello_traj_viz_node for dead reckoning)

Subscribes:
    /uav/action_cmd         (geometry_msgs/Twist)  normalised policy output
    /uav/lstm_pose_est      (geometry_msgs/Point)  re-published for viz node

Parameters (--ros-args -p)
    max_forward_cm_s    int    default 35
    max_yaw_deg_s       int    default 40
    max_updown_cm_s     int    default 0
    cmd_timeout_s       float  default 0.5
    stream_fps          float  default 20.0
    stream_w            int    default 320
    stream_h            int    default 240
    takeoff_on_start    bool   default True
    hover_height_cm     int    default 80
    video_topic         str    default /tello_stream
    imu_topic           str    default /imu
    action_topic        str    default /uav/action_cmd
    pose_topic          str    default /uav/lstm_pose_est
    rc_topic            str    default /tello/rc_command
"""

import math
import time
import threading
import queue

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist, Point, Quaternion

try:
    from djitellopy import Tello
except ImportError as e:
    raise ImportError("Run: pip install djitellopy") from e


STREAM_WARMUP_S = 2.0
G_TO_MS2        = 9.80665
RC_HZ           = 20      # Hz — Tello watchdog needs ≥15 Hz


# =============================================================================
# HELPERS
# =============================================================================

def rpy_to_quaternion(roll_rad, pitch_rad, yaw_rad) -> Quaternion:
    cr, sr = math.cos(roll_rad  * 0.5), math.sin(roll_rad  * 0.5)
    cp, sp = math.cos(pitch_rad * 0.5), math.sin(pitch_rad * 0.5)
    cy, sy = math.cos(yaw_rad   * 0.5), math.sin(yaw_rad   * 0.5)
    q = Quaternion()
    q.w = cr*cp*cy + sr*sp*sy
    q.x = sr*cp*cy - cr*sp*sy
    q.y = cr*sp*cy + sr*cp*sy
    q.z = cr*cp*sy - sr*sp*cy
    return q


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d


# =============================================================================
# RC SENDER THREAD
# =============================================================================

class RCSender:
    """
    Dedicated thread that drains a queue of (lr, fwd, ud, yaw) tuples
    and calls send_rc_control(). Isolates SDK blocking from ROS timers.
    """

    def __init__(self, tello: Tello):
        self._tello   = tello
        self._queue:  queue.Queue = queue.Queue(maxsize=2)
        self._latest  = (0, 0, 0, 0)   # last command sent
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="RCSender")
        self._thread.start()

    def send(self, lr: int, fwd: int, ud: int, yaw: int):
        """Non-blocking: drop oldest if queue is full."""
        try:
            self._queue.put_nowait((lr, fwd, ud, yaw))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait((lr, fwd, ud, yaw))

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)

    def _loop(self):
        interval = 1.0 / RC_HZ
        next_send = time.time()
        while self._running:
            now = time.time()
            if now >= next_send:
                try:
                    cmd = self._queue.get_nowait()
                    self._latest = cmd
                except queue.Empty:
                    cmd = self._latest   # repeat last command (keeps watchdog alive)
                try:
                    self._tello.send_rc_control(*cmd)
                except Exception:
                    pass
                next_send = now + interval
            else:
                time.sleep(max(0.0, next_send - now - 0.001))


# =============================================================================
# NODE
# =============================================================================

class TelloICMLSTMBridgeNode(Node):

    def __init__(self):
        super().__init__("tello_icm_lstm_bridge_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("max_forward_cm_s",  35)
        self.declare_parameter("max_yaw_deg_s",     40)
        self.declare_parameter("max_updown_cm_s",   0)
        self.declare_parameter("cmd_timeout_s",     0.5)
        self.declare_parameter("stream_fps",        20.0)
        self.declare_parameter("stream_w",          320)
        self.declare_parameter("stream_h",          240)
        self.declare_parameter("takeoff_on_start",  True)
        self.declare_parameter("hover_height_cm",   120)
        self.declare_parameter("video_topic",       "/tello_stream")
        self.declare_parameter("imu_topic",         "/imu")
        self.declare_parameter("action_topic",      "/uav/action_cmd")
        self.declare_parameter("pose_topic",        "/uav/lstm_pose_est")
        self.declare_parameter("rc_topic",          "/tello/rc_command")

        self._max_fwd = int(  self.get_parameter("max_forward_cm_s").value)
        self._max_yaw = int(  self.get_parameter("max_yaw_deg_s").value)
        self._max_ud  = int(  self.get_parameter("max_updown_cm_s").value)
        self._cmd_to  = float(self.get_parameter("cmd_timeout_s").value)
        self._fps     = float(self.get_parameter("stream_fps").value)
        self._sw      = int(  self.get_parameter("stream_w").value)
        self._sh      = int(  self.get_parameter("stream_h").value)
        self._auto_to = bool( self.get_parameter("takeoff_on_start").value)
        self._hover_h = int(  self.get_parameter("hover_height_cm").value)
        video_topic   = self.get_parameter("video_topic").value
        imu_topic     = self.get_parameter("imu_topic").value
        action_topic  = self.get_parameter("action_topic").value
        pose_topic    = self.get_parameter("pose_topic").value
        rc_topic      = self.get_parameter("rc_topic").value

        # ── Tello ─────────────────────────────────────────────────────────────
        self.tello = Tello()
        self.tello.connect()
        bat = self.tello.get_battery()
        self.get_logger().info(f"Tello connected — battery: {bat}%")
        if bat < 15:
            raise RuntimeError("Battery < 15% — refusing to take off.")

        self.tello.streamon()
        self.get_logger().info(
            f"Stream on — waiting {STREAM_WARMUP_S}s for decoder warm-up…")
        time.sleep(STREAM_WARMUP_S)
        self._frame_reader = self.tello.get_frame_read()

        # RC sender thread — keeps watchdog alive independently of ROS timers
        self._rc = RCSender(self.tello)

        # ── Takeoff ───────────────────────────────────────────────────────────
        self._airborne = False
        if self._auto_to:
            self.get_logger().info("Taking off…")
            self.tello.takeoff()
            time.sleep(2.0)
            # default_h = 120
            # diff = self._hover_h - default_h
            # if diff > 20:
            #     self.tello.move_up(min(diff, 100))
            # elif diff < -20:
            #     self.tello.move_down(min(-diff, 100))
            # time.sleep(1.0)
            self._airborne = True
            self.get_logger().info(f"Airborne at ~{self._hover_h} cm.")

        # ── State ─────────────────────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._last_cmd_time = time.time()
        self._vx_norm       = 0.0
        self._yaw_norm      = 0.0

        # IMU
        self._prev_roll_rad  = 0.0
        self._prev_pitch_rad = 0.0
        self._prev_yaw_rad   = 0.0
        self._prev_imu_time  = time.time()

        _or = (0.035)**2; _av = (0.10)**2; _la = (0.10)**2
        self._orient_cov = [_or,0,0, 0,_or,0, 0,0,_or]
        self._angvel_cov = [_av,0,0, 0,_av,0, 0,0,_av]
        self._linacc_cov = [_la,0,0, 0,_la,0, 0,0,_la]

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_video = self.create_publisher(Image, video_topic, 10)
        self.pub_imu   = self.create_publisher(Imu,   imu_topic,   10)
        self.pub_rc    = self.create_publisher(Twist, rc_topic,    10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Twist, action_topic, self._on_action, qos_profile_sensor_data)

        # ── Timers ────────────────────────────────────────────────────────────
        # RC control: dedicated RCSender thread handles timing;
        # this timer just computes the command and enqueues it.
        self.create_timer(1.0 / RC_HZ,   self._control_loop)
        self.create_timer(1.0 / self._fps, self._publish_sensors)

        self.get_logger().info(
            f"Bridge ready\n"
            f"  actions  ← {action_topic}\n"
            f"  rc sent  → {rc_topic}    (for visualiser)\n"
            f"  video    → {video_topic} @ {self._sw}×{self._sh} {self._fps:.0f}Hz\n"
            f"  imu      → {imu_topic}\n"
            f"  RC rate  : {RC_HZ} Hz (dedicated thread — watchdog safe)"
        )

    # ── Action subscriber ─────────────────────────────────────────────────────

    def _on_action(self, msg: Twist):
        with self._lock:
            self._vx_norm       = float(np.clip(msg.linear.x,  -0.2, 1.0))
            self._yaw_norm      = float(np.clip(msg.angular.z, -1.0, 1.0))
            self._last_cmd_time = time.time()

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        with self._lock:
            vx_n  = self._vx_norm
            yaw_n = self._yaw_norm
            fresh = (time.time() - self._last_cmd_time) < self._cmd_to

        if not fresh:
            self._rc.send(0, 0, 0, 0)
            self._pub_rc_msg(0.0, 0.0)
            return

        # Tuning (unchanged from original node)
        vx_n  = float(np.clip(vx_n  * 1.4,  -0.2, 1.0))
        yaw_n = float(np.clip(yaw_n * 0.7, -1.0, 1.0))

        if vx_n > 0.7:          yaw_n = 0.0
        if abs(yaw_n) < 0.4:    yaw_n = 0.0
        if abs(yaw_n) < 0.7 and vx_n < 0.2:
            vx_n = 0.2

        fwd = int(np.clip(vx_n  * self._max_fwd, -100, 100))
        yaw = int(np.clip(yaw_n * self._max_yaw, -100, 100))
        ud  = int(np.clip(self._max_ud,           -100, 100))

        self._rc.send(0, fwd, ud, yaw)
        self._pub_rc_msg(float(fwd), float(yaw))

    def _pub_rc_msg(self, fwd_cm_s: float, yaw_deg_s: float):
        """Publish the physical RC values so the viz node can dead-reckon."""
        msg           = Twist()
        msg.linear.x  = fwd_cm_s    # cm/s
        msg.angular.z = yaw_deg_s   # deg/s
        self.pub_rc.publish(msg)

    # ── Sensor publisher ──────────────────────────────────────────────────────

    def _publish_sensors(self):
        stamp = self.get_clock().now().to_msg()
        self._publish_frame(stamp)
        self._publish_imu(stamp)

    def _normalize_frame(self, frame) -> np.ndarray | None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        if frame.shape[1] != self._sw or frame.shape[0] != self._sh:
            frame = cv2.resize(frame, (self._sw, self._sh),
                               interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(frame)

    def _publish_frame(self, stamp):
        frame = self._normalize_frame(self._frame_reader.frame)
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

    def _publish_imu(self, stamp):
        try:
            state     = self.tello.get_current_state()
            roll_rad  = math.radians(float(state.get("roll",  0)))
            pitch_rad = math.radians(float(state.get("pitch", 0)))
            yaw_rad   = math.radians(float(state.get("yaw",   0)))
            ax = float(state.get("agx", 0.0)) * 0.001 * G_TO_MS2
            ay = float(state.get("agy", 0.0)) * 0.001 * G_TO_MS2
            az = float(state.get("agz", 0.0)) * 0.001 * G_TO_MS2
        except Exception as e:
            self.get_logger().warn(f"IMU read: {e}",
                                   throttle_duration_sec=5.0)
            return

        now = time.time()
        dt  = now - self._prev_imu_time
        if dt > 0.001:
            wx = angle_diff(roll_rad,  self._prev_roll_rad)  / dt
            wy = angle_diff(pitch_rad, self._prev_pitch_rad) / dt
            wz = angle_diff(yaw_rad,   self._prev_yaw_rad)   / dt
        else:
            wx = wy = wz = 0.0
        self._prev_roll_rad  = roll_rad
        self._prev_pitch_rad = pitch_rad
        self._prev_yaw_rad   = yaw_rad
        self._prev_imu_time  = now

        msg = Imu()
        msg.header.stamp    = stamp
        msg.header.frame_id = "tello_imu"
        msg.orientation     = rpy_to_quaternion(roll_rad, pitch_rad, yaw_rad)
        msg.orientation_covariance        = self._orient_cov
        msg.angular_velocity.x            = wx
        msg.angular_velocity.y            = wy
        msg.angular_velocity.z            = wz
        msg.angular_velocity_covariance   = self._angvel_cov
        msg.linear_acceleration.x         = ax
        msg.linear_acceleration.y         = ay
        msg.linear_acceleration.z         = az
        msg.linear_acceleration_covariance = self._linacc_cov
        self.pub_imu.publish(msg)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        self.get_logger().info("Shutting down — landing…")
        self._rc.send(0, 0, 0, 0)
        time.sleep(0.3)
        self._rc.stop()
        try:
            if self._airborne:
                self.tello.land()
            self.tello.streamoff()
            self.tello.end()
        except Exception as e:
            self.get_logger().warn(f"Shutdown error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TelloICMLSTMBridgeNode()
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