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
        self.forward_speed = 1.0    # 1 m/s forward
        self.yaw_speed = math.radians(45)  # Rotate at 45 deg/s
        self.forward_duration = 2.0  # Move forward for 2 seconds

        # --- State Tracking ---
        self.have_position = False
        self.have_heading = False
        
        self.current_altitude = 0.0
        self.current_yaw = 0.0
        
        # Task timing
        self.task_start_time = 0.0
        self.task_active = False
        self.yaw_accumulator = 0.0
        
        self.stream_counter = 0
        self.offboard_ready = False

        # --- QoS Setup ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, 
            "/fmu/in/offboard_control_mode", 
            qos
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, 
            "/fmu/in/trajectory_setpoint", 
            qos
        )

        # Subscribers
        self.position_sub = self.create_subscription(
            VehicleLocalPosition, 
            "/fmu/out/vehicle_local_position", 
            self.position_callback, 
            qos
        )

        # Main Control Loop Timer
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        
        self.get_logger().info(f"Task Test Node Initialized (BODY FRAME CONTROL). Selected Task: {self.task.upper()}")

    def position_callback(self, msg):
        """Receive PX4 position/heading data."""
        # Check if position is valid
        if hasattr(msg, 'x_valid') and msg.x_valid:
            self.current_altitude = -msg.z  # NED -> positive up
            self.have_position = True
            
        # Check if heading is valid
        if hasattr(msg, 'heading_valid') and msg.heading_valid:
            self.current_yaw = msg.heading
            self.have_heading = True
        elif hasattr(msg, 'heading'):
            self.current_yaw = msg.heading
            self.have_heading = True

    def publish_offboard_heartbeat(self):
        """Tells PX4 that this node wants to control the drone."""
        msg = OffboardControlMode()
        msg.timestamp = self.get_timestamp_us()
        msg.position = False      # Not using position control
        msg.velocity = True       # Using velocity control (body frame)
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_mode_pub.publish(msg)

    def publish_setpoint_body(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Publish velocity setpoint in body frame.
        This is the simplest mode - PX4 handles everything.
        
        Args:
            vx: Forward velocity in body frame (m/s)
            vy: Lateral velocity in body frame (m/s)  
            vz: Vertical velocity in body frame (m/s, positive up)
            yaw_rate: Yaw rate in body frame (rad/s)
        """
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_timestamp_us()
        msg.position = [nan, nan, nan]           # Not used (velocity control)
        msg.velocity = [float(vx), float(vy), float(vz)]  # Body frame velocities
        msg.acceleration = [nan, nan, nan]       # Not used
        msg.yaw = nan                            # Not used (using yawspeed)
        msg.yawspeed = float(yaw_rate)           # Yaw rate in body frame
        self.trajectory_pub.publish(msg)

    def control_loop(self):
        # Wait until we have position and heading data
        if not self.have_position or not self.have_heading:
            return

        # Always stream heartbeat first to satisfy PX4 safety checks
        self.publish_offboard_heartbeat()

        # DEFAULT SAFE SETPOINT: Hover (zero velocity)
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0

        # Wait for 20 iterations (1 second at 20Hz) of valid streaming before executing actions
        if not self.offboard_ready:
            self.stream_counter += 1
            if self.stream_counter >= 20:
                self.offboard_ready = True
                self.get_logger().info("Setpoint stream stable. Ready for Offboard mode.")
            self.publish_setpoint_body(vx, vy, vz, yaw_rate)
            return

        # --- EXECUTE SELECTED TASK (BODY FRAME) ---
        
        if self.task == "forward":
            # Move forward in body frame at 1 m/s
            vx = self.forward_speed
            vy = 0.0
            vz = 0.0
            yaw_rate = 0.0
            
            # Optional: Add altitude hold
            alt_error = self.target_altitude - self.current_altitude
            vz = np.clip(alt_error * 0.5, -0.5, 0.5)  # Simple altitude P-controller

        elif self.task == "yaw_360":
            # Hold position (zero velocity)
            vx = 0.0
            vy = 0.0
            
            # Simple altitude hold
            alt_error = self.target_altitude - self.current_altitude
            vz = np.clip(alt_error * 0.5, -0.5, 0.5)
            
            # Rotate at constant yaw rate until 360 degrees
            if self.yaw_accumulator < (2 * math.pi):
                yaw_rate = self.yaw_speed
                self.yaw_accumulator += yaw_rate * self.timer_period
            else:
                yaw_rate = 0.0  # Stop rotating once complete

        # Publish the setpoint in body frame
        self.publish_setpoint_body(vx, vy, vz, yaw_rate)

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