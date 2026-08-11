#!/usr/bin/env python3
"""
px4_keyboard_commander.py
=========================
Keyboard commander for PX4 guidance node.
Sends normalized commands to /uav/action_cmd with 0.1 increments.

Controls:
    w - Increase forward velocity (+0.1)
    s - Decrease forward velocity (-0.1)
    a - Increase yaw rate (left turn, +0.1)
    d - Decrease yaw rate (right turn, -0.1)
    Space - Emergency stop (zero everything)
    x - Center/Stop (zero everything)
    q - Quit
    +/- - Change step size

Current values are displayed in real-time.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import tty
import termios
import threading
import math

HELP = """
PX4 ICM Teleop - Keyboard Commander
-----------------------------------
w: increase forward velocity (+0.1)
s: decrease forward velocity (-0.1)
a: increase yaw rate (left turn, +0.1)
d: decrease yaw rate (right turn, -0.1)
Space: emergency stop (zero everything)
x: center/stop (zero everything)
+: increase step size
-: decrease step size
q: quit

Current step size: 0.1

Range: vx [-1.0, 1.0]  |  yaw [-1.0, 1.0]
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

class PX4KeyboardCommander(Node):
    def __init__(self):
        super().__init__('px4_keyboard_commander')

        # ── Parameters ──────────────────────────────────────────────────────────
        self.declare_parameter("step", 0.1)
        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_yaw", 1.0)
        self.declare_parameter("publish_rate", 20.0)

        self.step = self.get_parameter("step").value
        self.max_vx = self.get_parameter("max_vx").value
        self.max_yaw = self.get_parameter("max_yaw").value
        rate = self.get_parameter("publish_rate").value

        # ── State ──────────────────────────────────────────────────────────────
        self.vx = 0.0
        self.yaw = 0.0
        self.running = True

        # ── Publisher ──────────────────────────────────────────────────────────
        self.pub = self.create_publisher(Twist, '/uav/action_cmd', 10)
        self.create_timer(1.0/rate, self.publish)

        # ── Keyboard Thread ──────────────────────────────────────────────────
        self.kb_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.kb_thread.start()

        print(HELP)
        print(f"  Current step size: {self.step:.2f}")
        print(f"  vx: {self.vx:+.2f}  |  yaw: {self.yaw:+.2f}")
        print("=" * 60)

    def get_key(self):
        """Get a single keypress without Enter key."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def keyboard_loop(self):
        """Keyboard input handling loop."""
        while self.running:
            key = self.get_key().lower()

            # ── Forward/Backward ──────────────────────────────────────────────
            if key == 'w':
                self.vx = min(self.vx + self.step, self.max_vx)
            elif key == 's':
                self.vx = max(self.vx - self.step, -self.max_vx)
            
            # ── Yaw Left/Right ──────────────────────────────────────────────
            elif key == 'a':
                self.yaw = min(self.yaw + self.step, self.max_yaw)
            elif key == 'd':
                self.yaw = max(self.yaw - self.step, -self.max_yaw)

            # ── Emergency Stop ──────────────────────────────────────────────
            elif key == ' ' or key == 'x':
                self.vx = 0.0
                self.yaw = 0.0
                print("\n  EMERGENCY STOP: Zeroed all commands")

            # ── Change Step Size ──────────────────────────────────────────────
            elif key == '+':
                self.step = min(0.5, self.step + 0.05)
                print(f"\n  Step size increased to: {self.step:.2f}")
            elif key == '-':
                self.step = max(0.05, self.step - 0.05)
                print(f"\n  Step size decreased to: {self.step:.2f}")

            # ── Quit ──────────────────────────────────────────────────────────
            elif key == 'q':
                self.vx = 0.0
                self.yaw = 0.0
                self.running = False
                rclpy.shutdown()
                return

            # ── Display Current Values ────────────────────────────────────────
            self.get_logger().info(
                f'  vx: {self.vx:+.2f}  |  yaw: {self.yaw:+.2f}  '
                f'[step: {self.step:.2f}]'
            )

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
    node = PX4KeyboardCommander()
    
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