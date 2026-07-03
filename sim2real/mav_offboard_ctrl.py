#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleOdometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

NAN = float('nan')

# ── Safety limits — never exceed these regardless of model output ─────────────
MAX_VX_SAFE       = 1.5    # m/s  — must match training max_forward_vel
MAX_YAW_RATE_SAFE = 1.0    # rad/s — must match training max_yaw_rate
ACTION_CLIP       = 1.0    # model output clipped to [-1, 1] before scaling
MAX_ACTION_AGE_S  = 0.5    # seconds — if no new action, stop the drone


class OffboardRLControl(Node):
    def __init__(self):
        super().__init__("offboard_rl_control")

        self.MAX_VX       = MAX_VX_SAFE
        self.MAX_YAW_RATE = MAX_YAW_RATE_SAFE

        # Safe defaults — zero velocity until model sends something valid
        self.last_action      = [0.0, 0.0]
        self.last_action_time = None   # None = no action received yet

        # State from odometry
        self.current_yaw = 0.0
        self.current_pos = [0.0, 0.0, 0.0]
        self.state_valid = False
        self.hold_pos    = [0.0, 0.0, 0.0]

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            Float32MultiArray, "/rl_action", self.action_callback, 10)
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry",
            self.odom_callback, qos)

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)

        self.timer = self.create_timer(0.02, self.timer_callback)   # 50Hz
        self.get_logger().info("RL Offboard Control started — safety layer active")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def action_callback(self, msg):
        if len(msg.data) < 2:
            return

        # ── SAFETY LAYER 1: clip model output to [-1, 1] before anything ──────
        vx_raw       = float(msg.data[0])
        yaw_raw      = float(msg.data[1])

        vx_clipped   = max(-ACTION_CLIP, min(ACTION_CLIP, vx_raw))
        yaw_clipped  = max(-ACTION_CLIP, min(ACTION_CLIP, yaw_raw))

        # ── SAFETY LAYER 2: log if model was out of bounds ────────────────────
        if abs(vx_raw) > ACTION_CLIP or abs(yaw_raw) > ACTION_CLIP:
            self.get_logger().warn(
                f"[SAFETY] Model output clipped: "
                f"vx {vx_raw:+.3f}→{vx_clipped:+.3f}  "
                f"yaw {yaw_raw:+.3f}→{yaw_clipped:+.3f}",
                throttle_duration_sec=1.0,
            )

        self.last_action      = [vx_clipped, yaw_clipped]
        self.last_action_time = self.get_clock().now()

    def odom_callback(self, msg):
        self.current_pos = [msg.position[0], msg.position[1], msg.position[2]]
        q = msg.q   # [w, x, y, z]
        self.current_yaw = math.atan2(
            2.0 * (q[0]*q[3] + q[1]*q[2]),
            1.0 - 2.0 * (q[2]**2 + q[3]**2),
        )
        if not self.state_valid:
            self.hold_pos    = self.current_pos.copy()
            self.state_valid = True

    # ── Main control loop ─────────────────────────────────────────────────────

    def timer_callback(self):
        timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # ── SAFETY LAYER 3: action timeout — stop if inference node dies ──────
        action_stale = False
        if self.last_action_time is None:
            action_stale = True
        else:
            age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9
            if age > MAX_ACTION_AGE_S:
                action_stale = True
                self.get_logger().warn(
                    f"[SAFETY] Action stale ({age:.2f}s) — holding position",
                    throttle_duration_sec=1.0,
                )

        # ── Offboard heartbeat — must publish at >2Hz or PX4 exits offboard ───
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = timestamp
        offboard_msg.position  = True
        offboard_msg.velocity  = True
        self.offboard_pub.publish(offboard_msg)

        # ── Trajectory setpoint ───────────────────────────────────────────────
        traj = TrajectorySetpoint()
        traj.timestamp = timestamp

        if not self.state_valid or action_stale:
            # Hold current position — safest possible response
            traj.position    = [self.hold_pos[0], self.hold_pos[1], self.hold_pos[2]]
            traj.velocity    = [NAN, NAN, NAN]
            traj.yaw         = self.current_yaw
            traj.yawspeed    = NAN

        else:
            vx_body  = self.last_action[0] * self.MAX_VX       # now safe: [-1.5, 1.5]
            yaw_rate = self.last_action[1] * self.MAX_YAW_RATE  # now safe: [-2.0, 2.0]

            # Body → NED world frame
            vx_ned = vx_body * math.cos(self.current_yaw)
            vy_ned = vx_body * math.sin(self.current_yaw)

            if abs(vx_body) > 0.05:
                # Velocity mode
                traj.position    = [NAN, NAN, NAN]
                traj.velocity    = [vx_ned, vy_ned, 0.0]
                self.hold_pos    = self.current_pos.copy()
            else:
                # Position hold mode — snap to last known position
                traj.position    = [self.hold_pos[0], self.hold_pos[1], self.hold_pos[2]]
                traj.velocity    = [NAN, NAN, NAN]

            traj.yaw      = NAN
            traj.yawspeed = yaw_rate

        self.traj_pub.publish(traj)

        self.get_logger().info(
            f"yaw={math.degrees(self.current_yaw):+.1f}°  "
            f"vx={self.last_action[0]*self.MAX_VX:+.2f}m/s  "
            f"yr={self.last_action[1]*self.MAX_YAW_RATE:+.2f}rad/s  "
            f"{'STALE' if action_stale else 'VEL' if abs(self.last_action[0])>0.05 else 'HOLD'}",
            throttle_duration_sec=0.5,
        )


def main(args=None):
    rclpy.init(args=args)
    node = OffboardRLControl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()