#!/usr/bin/env python3
"""
px4_icm_guidance.py
===================
PX4 guidance node for ICM exploration.
Converts ICM commands (body frame) to PX4 setpoints (NED frame).

Subscribes:
    /uav/action_cmd (geometry_msgs/Twist) - normalised ICM commands
        linear.x  in [-1, 1]  → forward body velocity
        angular.z in [-1, 1]  → yaw rate

    /fmu/out/vehicle_local_position - for heading feedback (NED frame)

Publishes (to PX4 via uXRCE-DDS):
    /fmu/in/offboard_control_mode  - must publish >2 Hz
    /fmu/in/trajectory_setpoint    - velocity setpoints (NED frame)
"""

import time
import threading
import math
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
        VehicleLocalPosition,
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
    PX4 guidance node - converts body-frame commands to NED frame.
    Requires heading feedback from PX4.
    """

    def __init__(self):
        super().__init__("px4_icm_guidance")

        # ── Parameters ──────────────────────────────────────────────────────────
        self.declare_parameter("max_forward_m_s",    0.5)
        self.declare_parameter("max_yaw_rate_rad_s", 1.0)
        self.declare_parameter("cmd_timeout_s",      0.5)
        self.declare_parameter("setpoint_rate_hz",   20.0)

        self._max_fwd   = float(self.get_parameter("max_forward_m_s").value)
        self._max_yaw   = float(self.get_parameter("max_yaw_rate_rad_s").value)
        self._cmd_to    = float(self.get_parameter("cmd_timeout_s").value)
        rate_hz         = float(self.get_parameter("setpoint_rate_hz").value)

        # ── Control state ───────────────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._last_cmd_time = time.time()
        self._vx_norm       = 0.0
        self._yaw_norm      = 0.0

        # ── Heading from PX4 ───────────────────────────────────────────────────
        self._heading = 0.0      # radians (NED frame, 0 = North)
        self._heading_valid = False

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
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1",
            self._on_local_pos, PX4_QOS)

        # ── Timer ──────────────────────────────────────────────────────────────
        self._ctrl_timer = self.create_timer(1.0 / rate_hz, self._control_loop)
        self._diag_timer = self.create_timer(2.0, self._log_status)

        self.get_logger().info(
            f"PX4 ICM Guidance ready\n"
            f"  max_forward  = {self._max_fwd} m/s\n"
            f"  max_yaw_rate = {self._max_yaw} rad/s\n"
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

    # ── PX4 telemetry ──────────────────────────────────────────────────────────
    def _on_local_pos(self, msg: VehicleLocalPosition):
        """Receive heading from PX4."""
        with self._lock:
            if hasattr(msg, 'heading_valid') and msg.heading_valid:
                self._heading = float(msg.heading)
                self._heading_valid = True
            elif hasattr(msg, 'heading'):
                self._heading = float(msg.heading)
                self._heading_valid = True

    # ── Control loop ──────────────────────────────────────────────────────────
    def _control_loop(self):
        """
        Main control loop - runs at setpoint_rate_hz.
        Converts body-frame commands to NED frame using current heading.
        """
        with self._lock:
            vx_n = self._vx_norm
            yaw_n = self._yaw_norm
            heading = self._heading
            heading_valid = self._heading_valid
            fresh = (time.time() - self._last_cmd_time) < self._cmd_to

        # ── Command timeout → hover ──────────────────────────────────────────
        if not fresh:
            vx_n = 0.0
            yaw_n = 0.0

        # ── Action shaping ──────────────────────────────────────────────────────
        # Amplify forward slightly
        vx_n = float(np.clip(vx_n * 1.4, -0.4, 1.0))
        
        # Dampen yaw
        yaw_n = float(np.clip(yaw_n * 0.8, -1.0, 1.0))

        # ── FIX: Suppress yaw when moving forward fast ─────────────────────────
        # This must be done AFTER scaling but BEFORE deadzone
        if vx_n > 0.7:
            yaw_n = 0.0

        # Yaw deadzone (only applies if yaw wasn't suppressed)
        if abs(yaw_n) < 0.6:
            yaw_n = 0.0

        # Minimum forward nudge when not yawing
        if abs(yaw_n) < 0.7 and vx_n < 0.2 and vx_n > 0:
            vx_n = 0.2

        # ── Convert to physical units ──────────────────────────────────────────
        vx_body = vx_n * self._max_fwd      # m/s in body frame
        yaw_rate = yaw_n * self._max_yaw    # rad/s

        # ── Rotate body velocity to NED frame using heading ────────────────────
        if heading_valid:
            vn = vx_body * math.cos(heading)
            ve = vx_body * math.sin(heading)
        else:
            self.get_logger().warn("Heading not valid - using body frame")
            vn = vx_body
            ve = 0.0

        # ── Publish setpoints ──────────────────────────────────────────────────
        self._publish_ocm()
        self._publish_setpoint_ned(vn, ve, 0.0, yaw_rate)

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

    def _publish_setpoint_ned(self, vn: float, ve: float, vz: float, yaw_rate: float):
        """
        Publish velocity setpoint in NED frame.
        This is what PX4 expects for TrajectorySetpoint.
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
            heading_deg = math.degrees(self._heading) if self._heading_valid else 0.0
            age = time.time() - self._last_cmd_time

        self.get_logger().info(
            f"[GUIDANCE]  "
            f"vx_n={vx_n:+.2f}  yaw_n={yaw_n:+.2f}  "
            f"heading={heading_deg:.1f}°  "
            f"cmd_age={age:.2f}s"
        )

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def shutdown(self):
        self.get_logger().info("Shutting down — zeroing setpoints.")
        try:
            self._publish_setpoint_ned(0.0, 0.0, 0.0, 0.0)
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