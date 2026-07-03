#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition

class PX4TaskTest(Node):

    def __init__(self):
        super().__init__("px4_task_test")

        # --- Parameters ---
        # Options: "forward" or "yaw_360"
        self.declare_parameter("task", "forward")
        self.task = self.get_parameter("task").value
        
        self.target_altitude = 2.0  # Safe altitude in meters
        self.timer_period = 0.05    # 20 Hz loop rate
        
        # Task specific variables
        self.forward_distance = 2.0  # 2 meters forward
        self.yaw_speed = math.radians(45)  # Rotate at 45 deg/s

        # --- State Tracking ---
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.have_home = False
        
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.yaw_accumulator = 0.0  # Tracks total rotation for the 360 test
        
        self.stream_counter = 0
        self.offboard_ready = False

        # --- QoS Setup ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, "fmu/in/offboard_control_mode", qos)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, "fmu/in/trajectory_setpoint", qos)

        # Subscribers
        self.position_sub = self.create_subscription(
            VehicleLocalPosition, 
            "fmu/out/vehicle_local_position_v1", 
            self.position_callback, 
            qos
        )

        # Main Control Loop Timer
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        
        self.get_logger().info(f"Task Test Node Initialized. Selected Task: {self.task.upper()}")

    def position_callback(self, msg):
        # 1. Capture the initial home position as soon as data arrives
        if not self.have_home:
            self.home_x = msg.x
            self.home_y = msg.y
            self.home_z = msg.z
            self.have_home = True
            self.get_logger().info(f"Home position established: X={self.home_x:.2f}, Y={self.home_y:.2f}")

        # 2. Track current orientation (PX4 uses heading/yaw in radians)
        self.current_yaw = msg.heading

    def publish_offboard_heartbeat(self):
        """Tells PX4 that this node wants to control position."""
        msg = OffboardControlMode()
        msg.timestamp = self.get_timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_mode_pub.publish(msg)

    def control_loop(self):
        # Wait until we actually know where the drone is globally/locally
        if not self.have_home:
            return

        # Always stream heartbeat first to satisfy PX4 safety checks
        self.publish_offboard_heartbeat()

        # Initialize the setpoint structure
        sp = TrajectorySetpoint()
        sp.timestamp = self.get_timestamp_us()

        # DEFAULT SAFE SETPOINT: Hover at home position, 2m up
        sp.position[0] = self.home_x
        sp.position[1] = self.home_y
        sp.position[2] = -self.target_altitude  # NED Frame: Negative Z is up
        sp.yaw = 0.0

        # Wait for 20 iterations (1 second at 20Hz) of valid streaming before executing actions
        if not self.offboard_ready:
            self.stream_counter += 1
            if self.stream_counter >= 20:
                self.offboard_ready = True
                self.get_logger().info("Setpoint stream stable. Ready to switch to Offboard mode via QGC/RC.")
            self.trajectory_pub.publish(sp)
            return

        # --- EXECUTE SELECTED TASK ---
        if self.task == "forward":
            # Move 2m forward along the local X-axis
            sp.position[0] = self.home_x + self.forward_distance
            sp.position[1] = self.home_y
            sp.yaw = 0.0

        elif self.task == "yaw_360":
            # Hold current position position
            sp.position[0] = self.home_x
            sp.position[1] = self.home_y
            
            # Increment yaw over time if we haven't reached a full 360 rotation yet
            if self.yaw_accumulator < (2 * math.pi):
                step = self.yaw_speed * self.timer_period
                self.target_yaw += step
                self.yaw_accumulator += step
            else:
                # Lock it at exactly 360 degrees (0 rad) once finished
                self.target_yaw = 0.0

            sp.yaw = self.target_yaw

        # Publish the active task setpoint
        self.trajectory_pub.publish(sp)

    def get_timestamp_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)


def main(args=None):
    rclpy.init(args=args)
    node = PX4TaskTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()