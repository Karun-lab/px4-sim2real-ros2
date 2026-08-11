#!/usr/bin/env python3
"""
px4_joystick_commander.py
=========================
Simple joystick commander for PX4 guidance node.
Sends normalized commands to /uav/action_cmd.

Controls:
    Right Stick Y: Forward/Backward velocity (vx) - Up=Forward, Down=Backward
    Left Stick X: Yaw rate (yaw) - Left=Yaw Left, Right=Yaw Right
    A Button: Emergency stop (zero everything)
    B Button: Quit

Dependencies:
    pip install pygame
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame
import threading

HELP = """
PX4 ICM Joystick Commander
===========================
Controls:
  Right Stick Y: Forward/Backward velocity (vx)
  Left Stick X: Yaw rate (yaw)
  A Button: Emergency stop (zero everything)
  B Button: Quit

Range: vx [-1.0, 1.0]  |  yaw [-1.0, 1.0]
Deadzone: 0.15
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

class PX4JoystickCommander(Node):
    def __init__(self):
        super().__init__('px4_joystick_commander')

        # ── Parameters ──────────────────────────────────────────────────────────
        self.declare_parameter("deadzone", 0.15)
        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_yaw", 1.0)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("joystick_id", 0)

        self.deadzone = self.get_parameter("deadzone").value
        self.max_vx = self.get_parameter("max_vx").value
        self.max_yaw = self.get_parameter("max_yaw").value
        rate = self.get_parameter("publish_rate").value
        joystick_id = self.get_parameter("joystick_id").value

        # ── State ──────────────────────────────────────────────────────────────
        self.vx = 0.0
        self.yaw = 0.0
        self.running = True

        # ── Publisher ──────────────────────────────────────────────────────────
        self.pub = self.create_publisher(Twist, '/uav/action_cmd', 10)
        self.create_timer(1.0/rate, self.publish)

        # ── Initialize Joystick ────────────────────────────────────────────────
        if not self.init_joystick(joystick_id):
            return

        # ── Joystick Thread ────────────────────────────────────────────────────
        self.joystick_thread = threading.Thread(target=self.joystick_loop, daemon=True)
        self.joystick_thread.start()

        print(HELP)
        print(f"  Joystick: {self.joystick.get_name()}")
        print(f"  Axes: {self.joystick.get_numaxes()}")
        print(f"  Buttons: {self.joystick.get_numbuttons()}")
        print(f"  Deadzone: {self.deadzone:.2f}")
        print("=" * 60)

    def init_joystick(self, joystick_id):
        """Initialize pygame and connect to joystick."""
        try:
            pygame.init()
            pygame.joystick.init()
            
            num_joysticks = pygame.joystick.get_count()
            if num_joysticks == 0:
                print("\n  ERROR: No joystick detected.")
                print("  Please connect a gamepad and try again.")
                return False
            
            if joystick_id >= num_joysticks:
                print(f"\n  ERROR: Joystick {joystick_id} not found. {num_joysticks} available.")
                return False
            
            self.joystick = pygame.joystick.Joystick(joystick_id)
            self.joystick.init()
            
            # Print joystick info
            print(f"\n  Joystick connected:")
            print(f"    Name: {self.joystick.get_name()}")
            print(f"    Axes: {self.joystick.get_numaxes()}")
            print(f"    Buttons: {self.joystick.get_numbuttons()}")
            
            # Print axis mappings for debugging
            print("\n  Axis mapping (move sticks to see values):")
            for i in range(self.joystick.get_numaxes()):
                print(f"    Axis {i}: {self.joystick.get_axis(i):+.2f}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Joystick initialization failed: {e}")
            return False

    def apply_deadzone(self, value):
        """Apply deadzone to joystick axis value."""
        if abs(value) < self.deadzone:
            return 0.0
        # Rescale the remaining range
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def get_axis_safe(self, axis_index):
        """Safely get axis value, return 0 if axis doesn't exist."""
        if axis_index < self.joystick.get_numaxes():
            return self.joystick.get_axis(axis_index)
        return 0.0

    def joystick_loop(self):
        """Joystick input handling loop."""
        clock = pygame.time.Clock()
        
        # Detect which axes are actually sticks by checking which ones move
        # For ShanWan PC/PS3/Android with 4 axes:
        # Typical mapping: Axis 0 = Left X, Axis 1 = Left Y, Axis 2 = Right X, Axis 3 = Right Y
        
        # Let's try common mappings
        # Map 1: Axis 0=Left X, Axis 1=Left Y, Axis 2=Right X, Axis 3=Right Y
        # Map 2: Axis 0=Left X, Axis 1=Left Y, Axis 3=Right X, Axis 4=Right Y (not possible with 4 axes)
        
        # For your controller with 4 axes, try:
        left_x_axis = 0
        left_y_axis = 1
        right_x_axis = 2
        right_y_axis = 3
        
        print(f"\n  Using axis mapping:")
        print(f"    Left X: axis {left_x_axis}")
        print(f"    Left Y: axis {left_y_axis}")
        print(f"    Right X: axis {right_x_axis}")
        print(f"    Right Y: axis {right_y_axis}")
        print("\n  Controls active. Press A to stop, B to quit.\n")
        
        while self.running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    rclpy.shutdown()
                    return
                
                # ── Button Presses ──────────────────────────────────────────────
                if event.type == pygame.JOYBUTTONDOWN:
                    # A Button (0) - Emergency stop
                    if event.button == 0:
                        self.vx = 0.0
                        self.yaw = 0.0
                        print("\n  EMERGENCY STOP: Zeroed all commands")
                    
                    # B Button (1) - Quit
                    elif event.button == 1:
                        self.vx = 0.0
                        self.yaw = 0.0
                        self.running = False
                        rclpy.shutdown()
                        return
            
            # ── Read Stick Axes ──────────────────────────────────────────────────
            # Right Stick Y (axis 3) - Forward/Backward
            # Up = -1.0 (forward), Down = +1.0 (backward)
            # Invert so up = positive vx (forward)
            right_y = self.get_axis_safe(right_y_axis)
            self.vx = -self.apply_deadzone(right_y)
            
            # Left Stick X (axis 0) - Yaw rate
            # Left = -1.0, Right = +1.0
            left_x = self.get_axis_safe(left_x_axis)
            self.yaw = self.apply_deadzone(left_x)
            
            # ── Clamp values ──────────────────────────────────────────────────
            self.vx = max(-self.max_vx, min(self.max_vx, self.vx))
            self.yaw = max(-self.max_yaw, min(self.max_yaw, self.yaw))
            
            # ── Display current values (every 10 cycles) ──────────────────────
            if not hasattr(self, '_display_counter'):
                self._display_counter = 0
            
            self._display_counter += 1
            if self._display_counter % 10 == 0:
                self.get_logger().info(
                    f'  vx: {self.vx:+.2f}  |  yaw: {self.yaw:+.2f}'
                )
            
            # Cap the loop rate
            clock.tick(50)

    def publish(self):
        """Publish current command to /uav/action_cmd."""
        msg = Twist()
        msg.linear.x = float(self.vx)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(self.yaw)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PX4JoystickCommander()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency stop on exit
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        node.pub.publish(msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()