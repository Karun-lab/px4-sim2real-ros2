#!/usr/bin/env python3
"""
px4_icm_guidance.py
===================
Simple PX4 guidance node for ICM exploration.
Converts ICM commands to PX4 setpoints - NO feedback required.

Subscribes:
    /uav/action_cmd (geometry_msgs/Twist) - normalised ICM commands
        linear.x  in [-1, 1]  → forward velocity
        angular.z in [-1, 1]  → yaw rate

Publishes (to PX4 via uXRCE-DDS):
    /fmu/in/offboard_control_mode  - must publish >2 Hz
    /fmu/in/trajectory_setpoint    - velocity setpoints (body frame)

Parameters (all tunable via --ros-args -p)
------------------------------------------
    max_forward_m_s     float  default 0.5    max forward speed (m/s)
    max_yaw_rate_rad_s  float  default 0.5    max yaw rate (rad/s)
    cmd_timeout_s       float  default 0.5    hover after no command (s)
    setpoint_rate_hz    float  default 20.0   control loop rate (Hz)

Dependencies:
    px4_msgs (build from source or apt install)
"""

import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy,
)
from geometry_msgs.msg import Twist

try:
    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
    )
except ImportError as e:
    raise ImportError(
        "px4_msgs not found. Build px4_ros_com / px4_msgs and source the workspace."
    ) from e


# ── QoS for PX4 uXRCE-DDS bridge ────────────────────────────────────────────
PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class PX4ICMGuidanceNode(Node):
    """
    Simple PX4 guidance node - NO feedback required.
    Converts ICM commands to PX4 body-frame velocity setpoints.
    """

    def __init__(self):
        super().__init__("px4_icm_guidance")

        # ── Parameters ──────────────────────────────────────────────────────────
        self.declare_parameter("max_forward_m_s",    1.5)
        self.declare_parameter("max_yaw_rate_rad_s", 0.5)
        self.declare_parameter("cmd_timeout_s",      0.5)
        self.declare_parameter("setpoint_rate_hz",   20.0)
        self.declare_parameter("use_body_frame",     True)  # True = body frame velocities

        self._max_fwd   = float(self.get_parameter("max_forward_m_s").value)
        self._max_yaw   = float(self.get_parameter("max_yaw_rate_rad_s").value)
        self._cmd_to    = float(self.get_parameter("cmd_timeout_s").value)
        rate_hz         = float(self.get_parameter("setpoint_rate_hz").value)
        self._use_body  = bool(self.get_parameter("use_body_frame").value)

        # ── Control state ───────────────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._last_cmd_time = time.time()
        self._vx_norm       = 0.0
        self._yaw_norm      = 0.0

        # ── ROS publishers ──────────────────────────────────────────────────────
        self._pub_ocm = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self._pub_sp  = self.create_publisher(
            TrajectorySetpoint,  "/fmu/in/trajectory_setpoint",  PX4_QOS)
        self._pub_vc  = self.create_publisher(
            VehicleCommand,      "/fmu/in/vehicle_command",       PX4_QOS)

        # ── ROS subscribers ─────────────────────────────────────────────────────
        self.create_subscription(
            Twist, "/uav/action_cmd", self._on_action, 10)

        # ── Timer ──────────────────────────────────────────────────────────────
        self._ctrl_timer = self.create_timer(1.0 / rate_hz, self._control_loop)
        self._diag_timer = self.create_timer(2.0, self._log_status)

        self.get_logger().info(
            f"PX4 ICM Guidance ready\n"
            f"  max_forward  = {self._max_fwd} m/s\n"
            f"  max_yaw_rate = {self._max_yaw} rad/s\n"
            f"  use_body_frame = {self._use_body}\n"
            f"  cmd_timeout  = {self._cmd_to}s → hover\n"
            f"  rate         = {rate_hz} Hz"
        )

    # ── Action subscriber ──────────────────────────────────────────────────────
    def _on_action(self, msg: Twist):
        """Receive ICM commands."""
        with self._lock:
            self._vx_norm       = float(np.clip(msg.linear.x,  -1.0, 1.0))
            self._yaw_norm      = float(np.clip(msg.angular.z, -1.0, 1.0))
            self._last_cmd_time = time.time()

    # ── Control loop ──────────────────────────────────────────────────────────
    def _control_loop(self):
        """
        Main control loop - runs at setpoint_rate_hz.
        OffboardControlMode MUST be published at >2 Hz.
        """
        with self._lock:
            vx_n = self._vx_norm
            yaw_n = self._yaw_norm
            fresh = (time.time() - self._last_cmd_time) < self._cmd_to

        # ── Command timeout → hover ──────────────────────────────────────────
        if not fresh:
            vx_n = 0.0
            yaw_n = 0.0

        # ── Action shaping (mirrors Tello bridge logic) ──────────────────────
        # Amplify forward slightly, dampen yaw
        vx_n = float(np.clip(vx_n * 1.4, -0.2, 1.0))
        yaw_n = float(np.clip(yaw_n * 0.7, -1.0, 1.0))

        # Suppress yaw when moving forward fast
        if vx_n > 0.7:
            yaw_n = 0.0

        # Yaw deadzone
        if abs(yaw_n) < 0.4:
            yaw_n = 0.0

        # Minimum forward nudge when not yawing
        if abs(yaw_n) < 0.7 and vx_n < 0.2 and vx_n > 0:
            vx_n = 0.2

        # ── Convert to physical units ──────────────────────────────────────────
        vx_body = vx_n * self._max_fwd      # m/s in body frame
        yaw_rate = yaw_n * self._max_yaw    # rad/s

        # ── Publish setpoints ──────────────────────────────────────────────────
        self._publish_ocm()
        
        if self._use_body:
            # Body frame velocities (PX4 handles everything)
            self._publish_setpoint_body(vx_body, 0.0, 0.0, yaw_rate)
        else:
            # NED frame velocities (requires heading feedback)
            self._publish_setpoint_ned(vx_body, 0.0, 0.0, yaw_rate)

    # ── PX4 message builders ──────────────────────────────────────────────────
    def _ts(self) -> int:
        return int(time.time() * 1e6)

    def _publish_ocm(self):
        """Publish offboard control mode (must be >2Hz)."""
        msg = OffboardControlMode()
        msg.timestamp = self._ts()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._pub_ocm.publish(msg)

    def _publish_setpoint_body(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Publish velocity setpoint in body frame.
        This is the simplest mode - PX4 handles everything.
        """
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.timestamp = self._ts()
        msg.position = [nan, nan, nan]
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = nan
        msg.yawspeed = float(yaw_rate)
        self._pub_sp.publish(msg)

    def _publish_setpoint_ned(self, vn: float, ve: float, vz: float, yaw_rate: float):
        """
        Publish velocity setpoint in NED frame.
        Requires heading feedback from PX4 (more complex).
        """
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.timestamp = self._ts()
        msg.position = [nan, nan, nan]
        msg.velocity = [float(vn), float(ve), float(vz)]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = nan
        msg.yawspeed = float(yaw_rate)
        self._pub_sp.publish(msg)

    def _send_land(self):
        """Send land command."""
        msg = VehicleCommand()
        msg.timestamp = self._ts()
        msg.command = VehicleCommand.VEHICLE_CMD_NAV_LAND
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._pub_vc.publish(msg)
        self.get_logger().warn("LAND command sent")

    # ── Diagnostics ────────────────────────────────────────────────────────────
    def _log_status(self):
        with self._lock:
            vx_n = self._vx_norm
            yaw_n = self._yaw_norm
            age = time.time() - self._last_cmd_time

        self.get_logger().info(
            f"[GUIDANCE]  "
            f"vx_n={vx_n:+.2f}  yaw_n={yaw_n:+.2f}  "
            f"cmd_age={age:.2f}s"
        )

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def shutdown(self):
        self.get_logger().info("Shutting down — zeroing setpoints.")
        try:
            self._publish_setpoint_body(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.1)
        except Exception as e:
            self.get_logger().warn(f"Shutdown error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PX4ICMGuidanceNode()
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