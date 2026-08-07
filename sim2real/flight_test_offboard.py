#!/usr/bin/env python3
"""
px4_offboard_test.py
====================
Standalone open-loop test for verifying PX4 offboard body-frame velocity
setpoints work correctly. NOT connected to the ICM policy — this sends a
hardcoded motion sequence so you can confirm the offboard control mode /
trajectory setpoint plumbing is correct in isolation.

Sequence
--------
    Phase 0  ARM_STREAM   : publish zero setpoints for ~2s before requesting
                             offboard mode (PX4 requires a setpoint stream
                             for at least 10 setpoints / ~1s before it will
                             accept the OFFBOARD mode switch)
    Phase 1  FORWARD      : vx = +forward_speed   for forward_duration_s
    Phase 2  RIGHT        : vy = +right_speed     for right_duration_s
    Phase 3  HOVER        : vx = vy = 0, yaw_rate = 0, then node exits
                             (does NOT auto-land — land manually or via RC)

Body-frame convention (PX4 FRD — Forward, Right, Down)
--------------------------------------------------------
    vx > 0  → forward
    vy > 0  → right
    vz > 0  → down (not used in this test, stays 0)
    yawspeed → rad/s, not used in this test, stays 0

This matches _publish_setpoint_body() in px4_icm_guidance.py exactly:
    msg.velocity = [vx, vy, vz]   body frame, NaN position/accel, NaN yaw

SAFETY
------
    - This script does NOT arm the drone or switch flight modes itself.
      You must arm and switch to OFFBOARD manually (RC switch or QGC)
      AFTER this node starts publishing setpoints. If you switch to
      OFFBOARD before the node is running, PX4 will reject the mode
      switch or immediately fail back to the previous mode.
    - Recommended first test: run this with the drone TETHERED or with
      props OFF, watch /fmu/in/trajectory_setpoint in `ros2 topic echo`,
      and confirm phase transitions happen at the right times before
      ever running it with props spinning.
    - The node auto-hovers (zero velocity) at the end of the sequence
      and keeps publishing zero setpoints indefinitely — it will NOT
      exit and drop the offboard stream, which would trigger a PX4
      failsafe. Stop with Ctrl-C only after you've landed manually.

Run:
    ros2 run sim2real px4_offboard_test \\
        --ros-args \\
        -p forward_speed_m_s:=0.3 \\
        -p right_speed_m_s:=0.3 \\
        -p forward_duration_s:=2.0 \\
        -p right_duration_s:=2.0

Parameters (--ros-args -p)
    forward_speed_m_s    float  default 0.3   body-frame +X speed during FORWARD phase
    right_speed_m_s      float  default 0.3   body-frame +Y speed during RIGHT phase
    forward_duration_s   float  default 2.0   duration of FORWARD phase
    right_duration_s     float  default 2.0   duration of RIGHT phase
    arm_stream_s         float  default 2.0   zero-setpoint warm-up before phases start
    setpoint_rate_hz      float default 20.0  publish rate (PX4 requires ≥2 Hz, use ≥20 Hz)
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy,
)

try:
    from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
except ImportError as e:
    raise ImportError(
        "px4_msgs not found. Build px4_ros_com / px4_msgs and source the workspace."
    ) from e


PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

NAN = float("nan")


class PX4OffboardTestNode(Node):
    """
    Open-loop phase sequencer: ARM_STREAM → FORWARD → RIGHT → HOVER.
    Publishes body-frame TrajectorySetpoint + OffboardControlMode heartbeat
    at a fixed rate, advancing phases purely on elapsed wall-clock time.
    """

    def __init__(self):
        super().__init__("px4_offboard_test")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("forward_speed_m_s",  0.3)
        self.declare_parameter("right_speed_m_s",    0.3)
        self.declare_parameter("forward_duration_s", 2.0)
        self.declare_parameter("right_duration_s",   2.0)
        self.declare_parameter("arm_stream_s",       2.0)
        self.declare_parameter("setpoint_rate_hz",   20.0)

        self._fwd_speed  = float(self.get_parameter("forward_speed_m_s").value)
        self._right_speed= float(self.get_parameter("right_speed_m_s").value)
        self._fwd_dur    = float(self.get_parameter("forward_duration_s").value)
        self._right_dur  = float(self.get_parameter("right_duration_s").value)
        self._arm_dur    = float(self.get_parameter("arm_stream_s").value)
        rate_hz          = float(self.get_parameter("setpoint_rate_hz").value)

        # ── Phase schedule — cumulative end times from node start ────────────────
        self._t_arm_end     = self._arm_dur
        self._t_forward_end = self._t_arm_end + self._fwd_dur
        self._t_right_end   = self._t_forward_end + self._right_dur
        # After t_right_end → HOVER indefinitely

        self._t_start   = time.time()
        self._last_phase = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_ocm = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self._pub_sp  = self.create_publisher(
            TrajectorySetpoint,  "/fmu/in/trajectory_setpoint",  PX4_QOS)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / rate_hz, self._control_loop)

        self.get_logger().info(
            f"\nPX4 offboard open-loop test ready\n"
            f"  Phase 0  ARM_STREAM : 0.0s → {self._t_arm_end:.1f}s   "
            f"(zero setpoints — arm + switch to OFFBOARD now)\n"
            f"  Phase 1  FORWARD    : {self._t_arm_end:.1f}s → {self._t_forward_end:.1f}s   "
            f"(vx=+{self._fwd_speed:.2f} m/s)\n"
            f"  Phase 2  RIGHT      : {self._t_forward_end:.1f}s → {self._t_right_end:.1f}s   "
            f"(vy=+{self._right_speed:.2f} m/s)\n"
            f"  Phase 3  HOVER      : {self._t_right_end:.1f}s → forever   "
            f"(zero velocity — land manually)\n"
            f"  rate     : {rate_hz} Hz\n"
        )

    # ── Phase logic ───────────────────────────────────────────────────────────

    def _current_phase(self, t: float) -> tuple[str, float, float]:
        """
        Returns (phase_name, vx, vy) for the current elapsed time t.
        vx, vy are body-frame FRD velocities in m/s.
        """
        if t < self._t_arm_end:
            return "ARM_STREAM", 0.0, 0.0
        elif t < self._t_forward_end:
            return "FORWARD", self._fwd_speed, 0.0
        elif t < self._t_right_end:
            return "RIGHT", 0.0, self._right_speed
        else:
            return "HOVER", 0.0, 0.0

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        t = time.time() - self._t_start
        phase, vx, vy = self._current_phase(t)

        # Log on phase transition only — avoids spamming the console
        if phase != self._last_phase:
            self.get_logger().info(f"  >>> Entering phase: {phase}  "
                                   f"(t={t:.2f}s)  vx={vx:+.2f}  vy={vy:+.2f}")
            self._last_phase = phase

        self._publish_ocm()
        self._publish_setpoint_body(vx, vy, 0.0, 0.0)

    # ── PX4 message builders ──────────────────────────────────────────────────

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

    def _publish_setpoint_body(self, vx: float, vy: float,
                                vz: float, yaw_rate: float):
        """
        Identical to _publish_setpoint_body() in px4_icm_guidance.py.
        Body-frame FRD velocity setpoint. Position/accel/yaw left as NaN
        so PX4 ignores them and uses velocity + yawspeed only.
        """
        msg = TrajectorySetpoint()
        msg.timestamp    = self._ts()
        msg.position     = [NAN, NAN, NAN]
        msg.velocity     = [float(vx), float(vy), float(vz)]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw          = NAN
        msg.yawspeed     = float(yaw_rate)
        self._pub_sp.publish(msg)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        self.get_logger().info(
            "Shutting down — publishing zero setpoint. LAND MANUALLY if airborne.")
        try:
            self._publish_setpoint_body(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.1)
        except Exception as e:
            self.get_logger().warn(f"Shutdown error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PX4OffboardTestNode()
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