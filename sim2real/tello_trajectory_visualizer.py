#!/usr/bin/env python3
"""
tello_trajectory_visualizer.py
===============================
ROS 2 Node: Subscribes to /tello/imu_like for drone yaw and /uav/action_cmd for
forward commands. Dead-reckons 2D trajectory and plots path + coverage heatmap.

Plots:
    Vertical Axis (Y)   = Forward Motion (Meters Ahead)
    Horizontal Axis (X) = Lateral Motion (Meters Left/Right)
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, Vector3


class TelloTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__("tello_trajectory_visualizer")

        # --- Parameters ---
        self.declare_parameter("action_topic", "/uav/action_cmd")
        self.declare_parameter("imu_like_topic", "/tello/imu_like")
        self.declare_parameter("max_forward_m_s", 0.40)     # 40 cm/s
        self.declare_parameter("grid_resolution_m", 0.1)    # 10 cm
        self.declare_parameter("update_fps", 5.0)

        action_topic   = self.get_parameter("action_topic").value
        imu_like_topic = self.get_parameter("imu_like_topic").value
        self.max_v     = float(self.get_parameter("max_forward_m_s").value)
        self.res       = float(self.get_parameter("grid_resolution_m").value)
        fps            = float(self.get_parameter("update_fps").value)

        # --- Tracking State (World Coordinates) ---
        # World X_map = Lateral (Left/Right)
        # World Y_map = Forward (Ahead)
        self.x_map = 0.0
        self.y_map = 0.0
        self.current_yaw_rad = 0.0
        self.raw_yaw_deg = 0.0
        
        self.last_time = time.time()
        self.vx_norm = 0.0

        self.path_x = [0.0]
        self.path_y = [0.0]

        # --- Subscribers ---
        self.sub_action = self.create_subscription(
            Twist, action_topic, self._action_callback, qos_profile_sensor_data
        )
        self.sub_imu_like = self.create_subscription(
            Vector3, imu_like_topic, self._imu_like_callback, qos_profile_sensor_data
        )

        # --- Matplotlib Setup ---
        plt.ion()
        self.fig, (self.ax_path, self.ax_heat) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title("Tello Trajectory Visualizer")

        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self._update_and_plot)

        self.get_logger().info(f"Visualizer listening to {imu_like_topic} and {action_topic} at {fps} Hz...")

    def _action_callback(self, msg: Twist):
        self.vx_norm = float(np.clip(msg.linear.x, -0.2, 1.0))

    def _imu_like_callback(self, msg: Vector3):
        # msg.z contains Yaw in degrees (-180 to 180) from Tello
        self.raw_yaw_deg = msg.z
        # FIX: Invert the sign so left yaw turns left on the 2D Cartesian map
        self.current_yaw_rad = math.radians(-msg.z)

    def _update_and_plot(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # Convert normalized forward command to physical speed (m/s)
        v_fwd = self.vx_norm * self.max_v

        # Update position with swapped orientation axes:
        # Initial forward heading (0 deg) points UP along Y_map (sin=1, cos=0)
        # Yawing turns vector toward X_map (Horizontal)
        self.x_map += v_fwd * math.sin(self.current_yaw_rad) * dt   # Lateral step
        self.y_map += v_fwd * math.cos(self.current_yaw_rad) * dt   # Forward step

        self.path_x.append(self.x_map)
        self.path_y.append(self.y_map)

        # --- Render Plots ---
        self.ax_path.clear()
        self.ax_heat.clear()

        # 1. Path Plot
        self.ax_path.plot(self.path_x, self.path_y, 'b-', label='Path', linewidth=1.5)
        self.ax_path.plot(0, 0, 'go', label='Start (0,0)', markersize=8)
        self.ax_path.plot(self.x_map, self.y_map, 'ro', label='Current', markersize=8)

        # Yaw Arrow pointing in heading direction (Upward when starting)
        arrow_len = 0.25
        dx = arrow_len * math.sin(self.current_yaw_rad)
        dy = arrow_len * math.cos(self.current_yaw_rad)
        self.ax_path.arrow(self.x_map, self.y_map, dx, dy, head_width=0.05, head_length=0.05, fc='r', ec='r')

        yaw_deg_str = f"Yaw: {self.raw_yaw_deg:.1f}°"
        self.ax_path.set_title(f"Trajectory ({yaw_deg_str})")
        self.ax_path.set_xlabel("Lateral / Left-Right (Meters)")
        self.ax_path.set_ylabel("Forward / Ahead (Meters)")
        self.ax_path.grid(True)
        self.ax_path.legend(loc="upper left")
        self.ax_path.axis('equal')

        # 2. Coverage Heatmap
        if len(self.path_x) > 5:
            x_min, x_max = min(self.path_x) - 0.5, max(self.path_x) + 0.5
            y_min, y_max = min(self.path_y) - 0.5, max(self.path_y) + 0.5

            x_bins = max(int((x_max - x_min) / self.res), 5)
            y_bins = max(int((y_max - y_min) / self.res), 5)

            heatmap, xedges, yedges = np.histogram2d(
                self.path_x, self.path_y, bins=[x_bins, y_bins], range=[[x_min, x_max], [y_min, y_max]]
            )

            self.ax_heat.imshow(
                heatmap.T,
                origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='hot',
                interpolation='gaussian'
            )
            self.ax_heat.set_title("Spatial Dwelling Heatmap")
            self.ax_heat.set_xlabel("Lateral / Left-Right (Meters)")
            self.ax_heat.set_ylabel("Forward / Ahead (Meters)")
            self.ax_heat.axis('equal')

        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = TelloTrajectoryVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()