#!/usr/bin/env python3
"""
px4_icm_offboard_node.py
=========================
ROS 2 offboard control node for a PX4 drone driven by the ICM exploration
inference node. Mirrors the structure of tello_icm_bridge_node.py.

NO video streaming. NO auto-arming. NO auto-takeoff.
Operator arms and switches to OFFBOARD manually (RC or QGC) at ~1.2 m.
This node then holds altitude and forwards ICM action commands.

Subscribes:
    /uav/action_cmd  (geometry_msgs/Twist) — normalised ICM commands
        linear.x  in [-1, 1]  → forward body velocity
        angular.z in [-1, 1]  → yaw rate

Subscribes (from PX4 via uXRCE-DDS bridge):
    /fmu/out/vehicle_local_position  — altitude + heading
    /fmu/out/vehicle_status          — arm / nav-mode state

Publishes (to PX4 via uXRCE-DDS bridge):
    /fmu/in/offboard_control_mode    — must publish >2 Hz to stay in OFFBOARD
    /fmu/in/trajectory_setpoint      — NED velocity + yaw rate
    /fmu/in/vehicle_command          — LAND on safety trigger

Safety behaviour
----------------
    cmd_timeout_s  : no /uav/action_cmd → hover (zero horizontal velocity)
    land_timeout_s : cmd stream absent this long → send LAND and latch

Parameters (all tunable via --ros-args -p or launch file)
----------------------------------------------------------
    max_forward_m_s       float  default 0.4    max horizontal speed (m/s)
    max_yaw_rate_rad_s    float  default 0.3    max yaw rate (rad/s)
    target_alt_m          float  default 1.2    altitude to hold (m, +up)
    alt_kp                float  default 0.8    altitude P-gain (m/s per m error)
    max_vz_m_s            float  default 0.3    max vertical correction (m/s)
    cmd_timeout_s         float  default 0.5    hover threshold (s)
    land_timeout_s        float  default 3.0    land threshold (s)
    yaw_deadzone          float  default 0.15   |yaw_norm| below this → 0
    yaw_scale             float  default 0.6    yaw damping multiplier
    fwd_scale             float  default 1.4    forward amplification multiplier
    min_fwd_norm          float  default 0.2    min forward push when not yawing hard
    setpoint_rate_hz      float  default 20.0   setpoint publish rate (Hz)

NED frame reminder
------------------
    PX4 uses North-East-Down:
        z negative = up  →  1.2 m altitude = z = -1.2 in PX4
        heading 0 = North, increases clockwise
        body +X forward → NED:  vN = vx·cos(hdg),  vE = vx·sin(hdg)

Dependencies
------------
    px4_msgs  — build from source matching your PX4 firmware, or:
    sudo apt install ros-jazzy-px4-msgs  (if available)
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


# ── QoS required by PX4 uXRCE-DDS bridge 
PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class PX4ICMOffboardNode(Node):

    # PX4 nav-state constants (VehicleStatus.nav_state)
    NAV_STATE_OFFBOARD = 14
    # PX4 arming-state constants (VehicleStatus.arming_state)
    ARMING_STATE_ARMED = 2

    def __init__(self):
        super().__init__("px4_icm_offboard_node")

        # ── Parameters 
        self.declare_parameter("max_forward_m_s",    0.4)
        self.declare_parameter("max_yaw_rate_rad_s", 0.3)
        self.declare_parameter("target_alt_m",       1.2)
        self.declare_parameter("alt_kp",             0.8)
        self.declare_parameter("max_vz_m_s",         0.3)
        self.declare_parameter("cmd_timeout_s",      0.5)
        self.declare_parameter("land_timeout_s",     3.0)
        self.declare_parameter("yaw_deadzone",       0.15)
        self.declare_parameter("yaw_scale",          0.6)
        self.declare_parameter("fwd_scale",          1.4)
        self.declare_parameter("min_fwd_norm",       0.2)
        self.declare_parameter("setpoint_rate_hz",   20.0)

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
        rate_hz         = float(self.get_parameter("setpoint_rate_hz").value)

        # ── State 
        self._lock          = threading.Lock()
        self._vx_norm       = 0.0
        self._yaw_norm      = 0.0
        self._last_cmd_time = 0.0       # 0 = no command ever received
        self._cmd_ever_recv = False

        # From PX4 telemetry
        self._heading       = 0.0      # radians from North (NED)
        self._current_alt   = 0.0      # metres, positive up
        self._in_offboard   = False
        self._armed         = False

        # Latched once LAND is triggered — stops further setpoints
        self._landing       = False

        # ── Publishers ─────────────────────────────────────────────────────────
        self._pub_ocm = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self._pub_sp  = self.create_publisher(
            TrajectorySetpoint,  "/fmu/in/trajectory_setpoint",  PX4_QOS)
        self._pub_vc  = self.create_publisher(
            VehicleCommand,      "/fmu/in/vehicle_command",       PX4_QOS)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(
            Twist, "/uav/action_cmd",
            self._on_action, 10)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1",
            self._on_local_pos, PX4_QOS)
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._on_vehicle_status, PX4_QOS)

        # ── Timers ─────────────────────────────────────────────────────────────
        self._ctrl_timer = self.create_timer(1.0 / rate_hz, self._control_loop)
        self._diag_timer = self.create_timer(2.0,           self._log_status)

        self.get_logger().info(
            f"PX4 ICM Offboard node ready\n"
            f"  max_forward      = {self._max_fwd} m/s\n"
            f"  max_yaw_rate     = {self._max_yaw} rad/s\n"
            f"  target_alt       = {self._tgt_alt} m\n"
            f"  cmd_timeout      = {self._cmd_to}s → hover\n"
            f"  land_timeout     = {self._land_to}s → LAND\n"
            f"  yaw_deadzone     = {self._yaw_dz}  "
            f"yaw_scale = {self._yaw_scale}  "
            f"fwd_scale = {self._fwd_scale}"
        )

    # ── Action subscriber ──────────────────────────────────────────────────────
    def _on_action(self, msg: Twist):
        with self._lock:
            self._vx_norm       = float(np.clip(msg.linear.x,  -1.0, 1.0))
            self._yaw_norm      = float(np.clip(msg.angular.z, -1.0, 1.0))
            self._last_cmd_time = time.time()
            self._cmd_ever_recv = True

    # ── PX4 telemetry ──────────────────────────────────────────────────────────
    def _on_local_pos(self, msg: VehicleLocalPosition):
        with self._lock:
            # PX4 z is NED (negative = up). Convert to positive-up metres.
            self._current_alt = -float(msg.z)
            self._heading     = float(msg.heading)   # rad from North, NED

    def _on_vehicle_status(self, msg: VehicleStatus):
        with self._lock:
            self._in_offboard = (msg.nav_state == self.NAV_STATE_OFFBOARD)
            self._armed       = (msg.arming_state == self.ARMING_STATE_ARMED)

    # ── Control loop ──────────────────────────────────────────────────────────
    def _control_loop(self):
        """
        Runs at setpoint_rate_hz (default 20 Hz).
        OffboardControlMode MUST be published at >2 Hz or PX4 exits OFFBOARD.
        """
        if self._landing:
            self._publish_ocm()     # keep publishing so PX4 doesn't complain
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

        # ── Safety: cmd stream timeout ─────────────────────────────────────────
        if ever_recv:
            elapsed = now - last_cmd
            if elapsed > self._land_to:
                self.get_logger().error(
                    f"/uav/action_cmd lost for {elapsed:.1f}s — sending LAND.")
                self._trigger_land()
                return
            elif elapsed > self._cmd_to:
                vx_n  = 0.0     # hover: kill horizontal motion, hold altitude
                yaw_n = 0.0

        # ── Safety: altitude bounds ────────────────────────────────────────────
        min_alt = max(0.3, self._tgt_alt - 0.8)
        max_alt = self._tgt_alt + 1.5
        if ever_recv and in_offboard and (
                current_alt < min_alt or current_alt > max_alt):
            self.get_logger().error(
                f"Altitude {current_alt:.2f}m outside safe range "
                f"[{min_alt:.1f}, {max_alt:.1f}]m — sending LAND.")
            self._trigger_land()
            return

        # ── Action shaping (mirrors tello _control_loop) ───────────────────────
        # Amplify forward, dampen yaw
        vx_n  = float(np.clip(vx_n  * self._fwd_scale, -1.0, 1.0))
        yaw_n = float(np.clip(yaw_n * self._yaw_scale, -1.0, 1.0))

        # Yaw dead zone — small ICM yaw outputs mean "roughly straight"
        if abs(yaw_n) < self._yaw_dz:
            yaw_n = 0.0

        # Suppress yaw when strongly moving forward
        if vx_n > 0.7:
            yaw_n = 0.0

        # Minimum forward nudge when not yawing hard
        if abs(yaw_n) < 0.5 and vx_n < self._min_fwd:
            vx_n = self._min_fwd

        # ── Scale to physical units ────────────────────────────────────────────
        vx_body  = vx_n  * self._max_fwd   # m/s in body frame
        yaw_rate = yaw_n * self._max_yaw   # rad/s (positive = clockwise in NED)

        # Rotate body forward velocity → NED world frame using current heading.
        # heading = 0 → North.  body +X → vN = vx·cos(hdg), vE = vx·sin(hdg)
        vn = vx_body * math.cos(heading)
        ve = vx_body * math.sin(heading)

        # Altitude P controller.
        # error positive (below target) → need to climb → NED vz negative (up)
        alt_error = self._tgt_alt - current_alt
        vz_ned    = float(-np.clip(
            self._alt_kp * alt_error, -self._max_vz, self._max_vz))

        # ── Publish setpoints ──────────────────────────────────────────────────
        self._publish_ocm()
        self._publish_setpoint(vn, ve, vz_ned, yaw_rate)

    # ── PX4 message builders ───────────────────────────────────────────────────
    def _ts(self) -> int:
        """PX4 expects timestamps in microseconds."""
        return int(time.time() * 1e6)

    def _publish_ocm(self):
        """
        OffboardControlMode — tells PX4 which setpoint fields are valid.
        velocity=True only: PX4 reads velocity[0..2] and yawspeed from
        TrajectorySetpoint. Position controller is bypassed for XY;
        altitude is handled by our own vz P-controller above.
        """
        msg = OffboardControlMode()
        msg.timestamp    = self._ts()
        msg.position     = False
        msg.velocity     = True
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self._pub_ocm.publish(msg)

    def _publish_setpoint(self, vn: float, ve: float,
                          vz_ned: float, yaw_rate: float):
        """
        TrajectorySetpoint in NED frame (m/s).
        position = [nan, nan, nan]  → ignored (velocity-only mode)
        yaw      = nan              → use yawspeed instead
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

    def _trigger_land(self):
        """Send MAV_CMD_NAV_LAND and latch — no more velocity setpoints."""
        self._landing = True

        # One final zero setpoint before handing off to PX4 land controller
        self._publish_ocm()
        self._publish_setpoint(0.0, 0.0, 0.0, 0.0)

        cmd = VehicleCommand()
        cmd.timestamp        = self._ts()
        cmd.command          = VehicleCommand.VEHICLE_CMD_NAV_LAND
        cmd.target_system    = 1
        cmd.target_component = 1
        cmd.source_system    = 1
        cmd.source_component = 1
        cmd.from_external    = True
        cmd.param1           = 0.0   # abort altitude (0 = firmware default)
        cmd.param7           = 0.0   # landing target altitude
        self._pub_vc.publish(cmd)

        self.get_logger().warn(
            "LAND command sent. Velocity setpoints zeroed. "
            "OCM will keep publishing until node is shut down.")

    # ── Diagnostics ────────────────────────────────────────────────────────────
    def _log_status(self):
        with self._lock:
            vx_n   = self._vx_norm
            yaw_n  = self._yaw_norm
            alt    = self._current_alt
            hdg    = math.degrees(self._heading)
            armed  = self._armed
            offbd  = self._in_offboard
            ever   = self._cmd_ever_recv
            age    = time.time() - self._last_cmd_time if ever else -1.0

        if self._landing:
            state = "LANDING"
        elif not armed:
            state = "DISARMED"
        elif not offbd:
            state = "WAITING FOR OFFBOARD"
        else:
            state = "OFFBOARD — FLYING"

        self.get_logger().info(
            f"[{state}]  "
            f"alt={alt:.2f}m (tgt={self._tgt_alt}m)  "
            f"hdg={hdg:.1f}°  "
            f"vx_n={vx_n:+.2f}  yaw_n={yaw_n:+.2f}  "
            f"cmd_age={age:.2f}s"
        )

    # ── Cleanup 
    def shutdown(self):
        self.get_logger().info("Shutting down — zeroing setpoints.")
        try:
            self._publish_setpoint(0.0, 0.0, 0.0, 0.0)
        except Exception as e:
            self.get_logger().warn(f"Shutdown error: {e}")


# ENTRY POINT
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