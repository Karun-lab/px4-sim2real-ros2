#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class OffboardControl(Node):
    def __init__(self):
        super().__init__('offboard_circle')

        # --- Parameters ---
        self.radius = 10.0      # circle radius in meters
        self.altitude = 10.0    # fixed altitude (assume already taken off)
        self.omega = 0.3        # angular speed (rad/s)
        self.timer_period = 0.5  # 2 Hz
        self.theta = 0.0

        # --- QoS for PX4 topics ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.offboard_pub = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos)

        # Timer to send setpoints continuously
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("OffboardControl node started. Waiting for Offboard mode in QGC...")

    def timer_callback(self):
        # --- Publish Offboard mode (position) ---
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.offboard_pub.publish(offboard_msg)

        # --- Compute circular trajectory setpoint ---
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        traj_msg.position[0] = self.radius * np.cos(self.theta)
        traj_msg.position[1] = self.radius * np.sin(self.theta)
        traj_msg.position[2] = -self.altitude  # PX4 NED frame: negative Z is up

        # Publish trajectory
        self.traj_pub.publish(traj_msg)

        # Increment angle
        self.theta += self.omega * self.timer_period
        if self.theta > 2 * np.pi:
            self.theta -= 2 * np.pi  # keep it within 0..2pi

def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
