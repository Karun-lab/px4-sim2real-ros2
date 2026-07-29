#!/usr/bin/env python3
"""
px4_icm_offboard_node.py
=========================
ROS 2 offboard control node for a PX4 drone driven by ICM exploration.
Fixed heading interpretation and velocity scaling for optical flow localization.

Subscribes:
    /uav/action_cmd  (geometry_msgs/Twist) — normalised ICM commands
        linear.x  in [-1, 1]  → forward body velocity
        angular.z in [-1, 1]  → yaw rate

Subscribes (from PX4 via uXRCE-DDS bridge):
    /fmu/out/vehicle_local_position  — altitude + heading
    /fmu/out/vehicle_status          — arm / nav-mode state

Publishes (to PX4 via uXRCE-DDS bridge):
    /fmu/in/offboard_control_mode    — must publish >2 Hz to stay in OFFBOARD
    /fmu/in/trajectory_setpoint      — velocity setpoints in NED frame
    /fmu/in/vehicle_command          — LAND on safety trigger
"""

import math
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
        VehicleLocalPosition,
        VehicleStatus,
    )
except ImportError as e:
    raise ImportError(
        "px4_msgs not found. Build px4_ros_com / px4_msgs and source the workspace."
    ) from e


# ── QoS required by PX4 uXRCE-DDS bridge ────────────────────────────────────
PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class PX4ICMOffboardNode(Node):

    # PX4 nav-state constants (VehicleStatus.nav_state)
    NAV_STATE_OFFBOARD = 14
    NAV_STATE_ALTCTL = 5
    NAV_STATE_POSCTL = 6
    NAV_STATE_AUTO_LOITER = 4
    
    # PX4 arming-state constants (VehicleStatus.arming_state)
    ARMING_STATE_ARMED = 2

    def __init__(self):
        super().__init__("px4_icm_offboard_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("max_forward_m_s",    0.5)
        self.declare_parameter("max_yaw_rate_rad_s", 0.5)
        self.declare_parameter("target_alt_m",       1.2)
        self.declare_parameter("alt_kp",             0.8)
        self.declare_parameter("max_vz_m_s",         0.5)
        self.declare_parameter("cmd_timeout_s",      0.5)
        self.declare_parameter("land_timeout_s",     3.0)
        self.declare_parameter("yaw_deadzone",       0.1)
        self.declare_parameter("yaw_scale",          1.0)
        self.declare_parameter("fwd_scale",          1.0)
        self.declare_parameter("min_fwd_norm",       0.1)
        self.declare_parameter("setpoint_rate_hz",   20.0)
        self.declare_parameter("use_heading",        True)  # Use heading from VIO/optical flow

        self._max_fwd   = float(self.get_parameter("max_forward_m_s").value)
        self._max_yaw   = float(self.get_parameter("max_yaw_rate_rad_s").value)
        self._tgt_alt   = float(self.get_parameter("target_alt_m").value)
        self._alt_kp    = float(self.get_parameter("alt_kp").value)
        self._max_vz    = float(self.get_parameter("max_vz_m_s").value)
        self._cmd_to    = float(self.get_parameter("cmd_timeout_s").value)
        self._land_to   = float(self.get_parameter("land_timeout_s").value)
        self._yaw_dz    = float(self.get_parameter("yaw_deadzone").value)
        self._yaw_scale = float(self.get_parameter("yaw_scale").value)
        self._fwd_scale = float(self.get_parameter("fwd_scale").value)
        self._min_fwd   = float(self.get_parameter("min_fwd_norm").value)
        self._use_heading = bool(self.get_parameter("use_heading").value)
        rate_hz         = float(self.get_parameter("setpoint_rate_hz").value)

        # ── State ──────────────────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._vx_norm       = 0.0
        self._vy_norm       = 0.0  # Not used by ICM, but keep for completeness
        self._yaw_norm      = 0.0
        self._last_cmd_time = 0.0
        self._cmd_ever_recv = False

        # From PX4 telemetry
        self._heading       = 0.0      # radians from North (NED)
        self._current_alt   = 0.0      # metres, positive up
        self._in_offboard   = False
        self._armed         = False
        self._alt_valid     = False
        self._heading_valid = False

        # Latched once LAND is triggered
        self._landing       = False
        self._landing_start_time = 0.0

        # ── Publishers ──────────────────────────────────────────────────────
        self._pub_ocm = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self._pub_sp  = self.create_publisher(
            TrajectorySetpoint,  "/fmu/in/trajectory_setpoint",  PX4_QOS)
        self._pub_vc  = self.create_publisher(
            VehicleCommand,      "/fmu/in/vehicle_command",       PX4_QOS)

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            Twist, "/uav/action_cmd",
            self._on_action, 10)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._on_local_pos, PX4_QOS)
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._on_vehicle_status, PX4_QOS)

        # ── Timers ──────────────────────────────────────────────────────────
        self._ctrl_timer = self.create_timer(1.0 / rate_hz, self._control_loop)
        self._diag_timer = self.create_timer(2.0, self._log_status)

        self.get_logger().info(
            f"PX4 ICM Offboard node ready\n"
            f"  max_forward      = {self._max_fwd} m/s\n"
            f"  max_yaw_rate     = {self._max_yaw} rad/s\n"
            f"  target_alt       = {self._tgt_alt} m\n"
            f"  use_heading      = {self._use_heading}\n"
            f"  cmd_timeout      = {self._cmd_to}s → hover\n"
            f"  land_timeout     = {self._land_to}s → LAND\n"
        )

    # ── Action subscriber ────────────────────────────────────────────────────
    def _on_action(self, msg: Twist):
        with self._lock:
            self._vx_norm       = float(np.clip(msg.linear.x,  -1.0, 1.0))
            self._yaw_norm      = float(np.clip(msg.angular.z, -1.0, 1.0))
            self._last_cmd_time = time.time()
            self._cmd_ever_recv = True

    # ── PX4 telemetry ──────────────────────────────────────────────────────
    def _on_local_pos(self, msg: VehicleLocalPosition):
        with self._lock:
            # PX4 z is NED (negative = up). Convert to positive-up metres.
            if msg.z_valid:
                self._current_alt = -float(msg.z)
                self._alt_valid = True
            if msg.heading_valid:
                self._heading = float(msg.heading)
                self._heading_valid = True

    def _on_vehicle_status(self, msg: VehicleStatus):
        with self._lock:
            self._in_offboard = (msg.nav_state == self.NAV_STATE_OFFBOARD)
            self._armed       = (msg.arming_state == self.ARMING_STATE_ARMED)

    # ── Control loop ──────────────────────────────────────────────────────
    def _control_loop(self):
        """
        Runs at setpoint_rate_hz. OffboardControlMode MUST be published at >2 Hz.
        """
        if self._landing:
            # Keep publishing OCM during landing
            self._publish_ocm()
            return

        now = time.time()

        with self._lock:
            vx_n        = self._vx_norm
            yaw_n       = self._yaw_norm
            last_cmd    = self._last_cmd_time
            ever_recv   = self._cmd_ever_recv
            current_alt = self._current_alt
            heading     = self._heading
            in_offboard = self._in_offboard
            armed       = self._armed
            alt_valid   = self._alt_valid
            heading_valid = self._heading_valid

        # ── Safety: Don't send commands if not armed ──────────────────────
        if not armed:
            return

        # ── Safety: Don't send commands if not in OFFBOARD ────────────────
        if not in_offboard:
            self._publish_ocm()  # Keep OCM publishing for OFFBOARD transition
            return

        # ── Safety: cmd stream timeout ────────────────────────────────────
        if ever_recv:
            elapsed = now - last_cmd
            if elapsed > self._land_to:
                self.get_logger().error(
                    f"CMD lost for {elapsed:.1f}s — LANDING")
                self._trigger_land()
                return
            elif elapsed > self._cmd_to:
                self.get_logger().warn(f"CMD timeout {elapsed:.1f}s — hovering")
                vx_n = 0.0
                yaw_n = 0.0

        # ── Safety: altitude bounds ──────────────────────────────────────
        if alt_valid:
            min_alt = 0.3
            max_alt = self._tgt_alt + 2.0
            if current_alt < min_alt or current_alt > max_alt:
                self.get_logger().error(
                    f"Altitude {current_alt:.2f}m outside safe range "
                    f"[{min_alt:.1f}, {max_alt:.1f}]m — LANDING")
                self._trigger_land()
                return

        # ── Action shaping ──────────────────────────────────────────────────
        # Apply scaling
        vx_n = float(np.clip(vx_n * self._fwd_scale, -1.0, 1.0))
        yaw_n = float(np.clip(yaw_n * self._yaw_scale, -1.0, 1.0))

        # Yaw dead zone
        if abs(yaw_n) < self._yaw_dz:
            yaw_n = 0.0

        # Minimum forward nudge when not yawing
        if abs(yaw_n) < 0.3 and vx_n < self._min_fwd and vx_n > 0:
            vx_n = self._min_fwd

        # ── Convert to physical units ──────────────────────────────────────
        # Forward velocity in body frame (m/s)
        vx_body = vx_n * self._max_fwd
        
        # Yaw rate (rad/s) - positive = clockwise in NED
        yaw_rate = yaw_n * self._max_yaw

        # ── Rotate to NED world frame ──────────────────────────────────────
        # IMPORTANT: PX4 NED frame:
        #   - North = +X, East = +Y, Down = +Z
        #   - Heading = 0 = North, increases clockwise (East = +90°)
        #   - Body frame: +X = forward, +Y = right
        #
        # For optical flow localization:
        #   - The heading from optical flow is in NED frame
        #   - Need to rotate body velocity to NED using heading
        if self._use_heading and heading_valid:
            # Rotate body forward velocity to NED frame
            # v_N = v_forward * cos(heading)
            # v_E = v_forward * sin(heading)
            vn = vx_body * math.cos(heading)
            ve = vx_body * math.sin(heading)
        else:
            # Use body-frame velocities directly (vehicle_local_position provides this)
            # In this case, we send velocity in body frame, not world frame
            # PX4 will handle the rotation if we set the appropriate flags
            vn = vx_body
            ve = 0.0

        # ── Altitude control ──────────────────────────────────────────────
        # PX4 NED: positive z = down, so climb = negative vz
        if alt_valid:
            alt_error = self._tgt_alt - current_alt
            vz_ned = -np.clip(self._alt_kp * alt_error, -self._max_vz, self._max_vz)
        else:
            vz_ned = 0.0

        # ── Publish setpoints ──────────────────────────────────────────────
        self._publish_ocm()
        
        # If we have heading, use yawspeed
        if self._use_heading and heading_valid:
            self._publish_setpoint_velocity(vn, ve, vz_ned, yaw_rate)
        else:
            # Without heading, send body velocity (PX4 will handle rotation)
            self._publish_setpoint_body(vx_body, 0.0, vz_ned, yaw_rate)

    # ── PX4 message builders ────────────────────────────────────────────────
    def _ts(self) -> int:
        return int(time.time() * 1e6)

    def _publish_ocm(self):
        msg = OffboardControlMode()
        msg.timestamp    = self._ts()
        msg.position     = False
        msg.velocity     = True
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self._pub_ocm.publish(msg)

    def _publish_setpoint_velocity(self, vn: float, ve: float, 
                                   vz_ned: float, yaw_rate: float):
        """
        Publish velocity setpoint in NED world frame.
        Used when heading is available from VIO/optical flow.
        """
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.timestamp    = self._ts()
        msg.position     = [nan, nan, nan]
        msg.velocity     = [float(vn), float(ve), float(vz_ned)]
        msg.acceleration = [nan, nan, nan]
        msg.yaw          = nan
        msg.yawspeed     = float(yaw_rate)
        self._pub_sp.publish(msg)

    def _publish_setpoint_body(self, vx_body: float, vy_body: float,
                               vz_ned: float, yaw_rate: float):
        """
        Publish velocity setpoint in body frame.
        Used when heading is NOT available (optical flow only).
        PX4 will handle the rotation.
        """
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.timestamp    = self._ts()
        msg.position     = [nan, nan, nan]
        msg.velocity     = [float(vx_body), float(vy_body), float(vz_ned)]
        msg.acceleration = [nan, nan, nan]
        msg.yaw          = nan
        msg.yawspeed     = float(yaw_rate)
        self._pub_sp.publish(msg)

    def _trigger_land(self):
        """Send LAND command and latch."""
        if self._landing:
            return
            
        self._landing = True
        self._landing_start_time = time.time()

        # Send zero setpoint
        self._publish_ocm()
        self._publish_setpoint_velocity(0.0, 0.0, 0.0, 0.0)

        cmd = VehicleCommand()
        cmd.timestamp        = self._ts()
        cmd.command          = VehicleCommand.VEHICLE_CMD_NAV_LAND
        cmd.target_system    = 1
        cmd.target_component = 1
        cmd.source_system    = 1
        cmd.source_component = 1
        cmd.from_external    = True
        cmd.param1           = 0.0
        cmd.param7           = 0.0
        self._pub_vc.publish(cmd)

        self.get_logger().warn("LAND command sent.")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def _log_status(self):
        with self._lock:
            vx_n   = self._vx_norm
            yaw_n  = self._yaw_norm
            alt    = self._current_alt
            hdg    = math.degrees(self._heading) if self._heading_valid else 0.0
            armed  = self._armed
            offbd  = self._in_offboard
            ever   = self._cmd_ever_recv
            age    = time.time() - self._last_cmd_time if ever else -1.0
            alt_valid = self._alt_valid

        if self._landing:
            state = "LANDING"
        elif not armed:
            state = "DISARMED"
        elif not offbd:
            state = "WAITING FOR OFFBOARD"
        else:
            state = "OFFBOARD"

        self.get_logger().info(
            f"[{state}]  "
            f"alt={alt:.2f}m (tgt={self._tgt_alt}m)  "
            f"hdg={hdg:.1f}°  "
            f"vx_n={vx_n:+.2f}  yaw_n={yaw_n:+.2f}  "
            f"cmd_age={age:.2f}s  "
            f"alt_valid={alt_valid}"
        )

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def shutdown(self):
        self.get_logger().info("Shutting down — zeroing setpoints.")
        try:
            self._publish_setpoint_velocity(0.0, 0.0, 0.0, 0.0)
        except Exception as e:
            self.get_logger().warn(f"Shutdown error: {e}")


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = PX4ICMOffboardNode()
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