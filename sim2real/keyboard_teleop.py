#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import sys
import tty
import termios
import threading

HELP = """
RL Teleop - PX4 Offboard Test
-----------------------------

w : increase forward velocity
s : decrease forward velocity
a : yaw left
d : yaw right
space : brake (reduce vx)
x : reset (vx=0, yaw=0)
q : quit

vx step      : 0.1 (normalized)
yaw step     : 0.1 (normalized)

Output:
[vx, yaw_rate] in range [-1, 1]

++++++++++++++++++++++++++++++
"""

class RLTeleopNode(Node):
    def __init__(self):
        super().__init__('rl_teleop')

        # Parameters
        self.declare_parameter("vx_step", 0.1)
        self.declare_parameter("yaw_step", 0.1)
        self.declare_parameter("publish_rate", 20.0)

        self.vx_step = self.get_parameter("vx_step").value
        self.yaw_step = self.get_parameter("yaw_step").value
        rate = self.get_parameter("publish_rate").value

        self.brake_step = 2 * self.vx_step

        # RL-style normalized values
        self.vx = 0.0
        self.yaw = 0.0

        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, '/rl_action', 10)
        self.create_timer(1.0 / rate, self.publish)

        # Keyboard thread
        self.running = True
        self.kb_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.kb_thread.start()

        print(HELP)

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def keyboard_loop(self):
        while self.running:
            key = self.get_key().lower()

            if key == "w":
                self.vx = min(self.vx + self.vx_step, 1.0)

            elif key == "s":
                self.vx = max(self.vx - self.vx_step, -1.0)

            elif key == "d":
                self.yaw = min(self.yaw + self.yaw_step, 1.0)

            elif key == "a":
                self.yaw = max(self.yaw - self.yaw_step, -1.0)

            elif key == " ":
                # Brake only affects vx
                if self.vx > 0:
                    self.vx = max(0.0, self.vx - self.brake_step)
                elif self.vx < 0:
                    self.vx = min(0.0, self.vx + self.brake_step)

            elif key == "x":
                self.vx = 0.0
                self.yaw = 0.0

            elif key == "q":
                self.vx = 0.0
                self.yaw = 0.0
                self.running = False
                rclpy.shutdown()
                return

            self.get_logger().info(
                f'vx: {self.vx:+.2f} | yaw_rate: {self.yaw:+.2f}'
            )

    def publish(self):
        msg = Float32MultiArray()
        msg.data = [self.vx, self.yaw]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RLTeleopNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()