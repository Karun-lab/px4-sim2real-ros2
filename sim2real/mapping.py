#!/usr/bin/env python3
"""
mapping.py
======================================================================
Simple standalone mapping module for PX4 drones.

Features:
    - Start mapping when drone arms, stop when disarms
    - Save PNG only on disarm (no periodic autosave)
    - Clean occupancy grid with trajectory overlay
    - Safe to run alongside manual flight

Subscribes:
    /fmu/out/vehicle_local_position_v1   (px4_msgs/VehicleLocalPosition)
    /fmu/out/vehicle_status_v4           (px4_msgs/VehicleStatus) - for arming state

Publishes:
    /exploration/heatmap                 (nav_msgs/OccupancyGrid)

Run:
    ros2 run sim2real mapping
"""

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid

try:
    from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
except ImportError as e:
    raise ImportError(
        "px4_msgs not found. Build px4_ros_com / px4_msgs and source the workspace."
    ) from e


# =============================================================================
# CONSTANTS
# =============================================================================

GRID_CELL_M = 0.25
GRID_EXTENT_M = 30.0
KALMAN_Q_POS = 0.01
KALMAN_Q_VEL = 0.5
KALMAN_R_POS = 0.05
KALMAN_R_VEL = 0.05
KALMAN_MAX_JUMP_M = 1.0


# =============================================================================
# SIMPLE KALMAN FILTER
# =============================================================================

class SimpleKalman:
    """Minimal 2D position/velocity Kalman filter."""
    
    def __init__(self):
        self.x = np.zeros(4, dtype=np.float64)  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float64)
        self.initialised = False

    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.x = np.array([x, y, vx, vy], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self.initialised = True

    def predict(self, dt: float):
        if not self.initialised or dt <= 0.0:
            return
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = np.diag([KALMAN_Q_POS * dt, KALMAN_Q_POS * dt,
                     KALMAN_Q_VEL * dt, KALMAN_Q_VEL * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, x: float, y: float, vx: float, vy: float) -> bool:
        if not self.initialised:
            self.init(x, y, vx, vy)
            return False

        # Detect jumps (VIO resets)
        jump = math.hypot(x - self.x[0], y - self.x[1])
        if jump > KALMAN_MAX_JUMP_M:
            self.init(x, y, vx, vy)
            return True

        H = np.eye(4)
        z = np.array([x, y, vx, vy])
        R = np.diag([KALMAN_R_POS, KALMAN_R_POS, KALMAN_R_VEL, KALMAN_R_VEL])
        
        y_res = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_res
        self.P = (np.eye(4) - K @ H) @ self.P
        return False

    @property
    def position(self):
        return float(self.x[0]), float(self.x[1])


# =============================================================================
# VISIT GRID
# =============================================================================

class VisitGrid:
    def __init__(self):
        self.cell_m = GRID_CELL_M
        self.n = int(GRID_EXTENT_M / GRID_CELL_M) | 1
        self.half = self.n // 2
        
        self.visit_count = np.zeros((self.n, self.n), dtype=np.float32)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_set = False
        self.trajectory = []
        self.armed = False
        self.recording = False

    def set_origin(self, x: float, y: float):
        if not self.origin_set:
            self.origin_x = x
            self.origin_y = y
            self.origin_set = True

    def pos_to_cell(self, x: float, y: float):
        if not self.origin_set:
            return 0, 0
        lx = x - self.origin_x
        ly = y - self.origin_y
        col = int(round(lx / self.cell_m + self.half))
        row = int(round(-ly / self.cell_m + self.half))
        return max(0, min(col, self.n - 1)), max(0, min(row, self.n - 1))

    def visit(self, x: float, y: float):
        row, col = self.pos_to_cell(x, y)
        self.visit_count[row, col] += 1.0
        self.trajectory.append((x, y))

    def get_heatmap(self) -> np.ndarray:
        return 1.0 / (self.visit_count + 1.0)

    def covered_area(self) -> float:
        return float(np.sum(self.visit_count > 0)) * (self.cell_m ** 2)

    def total_distance(self) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        arr = np.array(self.trajectory)
        return float(np.sum(np.hypot(*np.diff(arr, axis=0).T)))

    def reset(self):
        """Reset grid for new flight."""
        self.visit_count.fill(0.0)
        self.trajectory.clear()
        self.origin_set = False
        self.recording = False


# =============================================================================
# ROS 2 NODE
# =============================================================================

class MappingNode(Node):
    def __init__(self):
        super().__init__("mapping_node")

        # Parameters
        self.declare_parameter("save_path", "/tmp/exploration_map.png")
        self.save_path = self.get_parameter("save_path").value

        # State
        self.grid = VisitGrid()
        self.kf = SimpleKalman()
        self.armed = False
        self.was_armed = False
        self.heatmap_pub_count = 0

        # Subscribers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_vio = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1",
            self._on_vio, qos)
        self.sub_status = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1",
            self._on_status, qos)

        # Publishers
        self.pub_heatmap = self.create_publisher(OccupancyGrid, "/exploration/heatmap", 10)

        # Timer (1Hz for heatmap, 2Hz for status)
        self.timer = self.create_timer(0.5, self._tick)

        self.get_logger().info(
            "Mapping node ready.\n"
            "  ARM to start mapping, DISARM to stop and save map.\n"
            f"  Map saved to: {self.save_path}"
        )

    # ──────────────────────────────────────────────────────────────────────
    def _on_status(self, msg):
        """Track arming state."""
        self.armed = (msg.arming_state == 2)  # ARMING_STATE_ARMED

        # Start recording when armed
        if self.armed and not self.was_armed:
            self.grid.reset()
            self.kf = SimpleKalman()
            self.get_logger().info("ARMED — Starting new map")

        # Stop and save when disarmed
        if not self.armed and self.was_armed:
            self.get_logger().info("DISARMED — Saving map")
            self._save_png()

        self.was_armed = self.armed

    # ──────────────────────────────────────────────────────────────────────
    def _on_vio(self, msg):
        """Update position when recording."""
        if not self.armed or not getattr(msg, "xy_valid", True):
            return

        # Kalman filter
        dt = 0.05  # ~20Hz
        self.kf.predict(dt)
        self.kf.update(float(msg.x), float(msg.y),
                       float(getattr(msg, "vx", 0.0)),
                       float(getattr(msg, "vy", 0.0)))

        # Update grid
        x, y = self.kf.position
        self.grid.set_origin(x, y)
        self.grid.visit(x, y)
        self.grid.recording = True

    # ──────────────────────────────────────────────────────────────────────
    def _tick(self):
        """Publish heatmap at 2Hz while armed."""
        if not self.armed or not self.grid.recording:
            return

        self._publish_heatmap()
        self.heatmap_pub_count += 1

    # ──────────────────────────────────────────────────────────────────────
    def _publish_heatmap(self):
        """Publish occupancy grid for RViz."""
        if not self.grid.origin_set:
            return

        novelty = self.grid.get_heatmap()
        occupancy = ((1.0 - novelty) * 100).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.width = self.grid.n
        msg.info.height = self.grid.n
        msg.info.resolution = self.grid.cell_m
        msg.info.origin.position.x = self.grid.origin_x - self.grid.half * self.grid.cell_m
        msg.info.origin.position.y = self.grid.origin_y - self.grid.half * self.grid.cell_m
        msg.data = occupancy.flatten().tolist()
        self.pub_heatmap.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    def _save_png(self):
        """Save final map as PNG."""
        if not self.grid.origin_set or len(self.grid.trajectory) < 2:
            self.get_logger().warn("No trajectory data to save.")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.get_logger().warn("matplotlib not available.")
            return

        try:
            g = self.grid
            novelty = g.get_heatmap()
            
            # Crop to visited area
            visited = g.visit_count > 0
            if visited.any():
                rows, cols = np.where(visited)
                margin = 5
                r0 = max(0, rows.min() - margin)
                r1 = min(g.n - 1, rows.max() + margin)
                c0 = max(0, cols.min() - margin)
                c1 = min(g.n - 1, cols.max() + margin)
            else:
                r0, r1, c0, c1 = 0, g.n - 1, 0, g.n - 1

            crop = novelty[r0:r1+1, c0:c1+1]
            
            # Extent
            lx0 = (c0 - g.half) * g.cell_m
            lx1 = (c1 - g.half) * g.cell_m
            ly0 = (r0 - g.half) * g.cell_m
            ly1 = (r1 - g.half) * g.cell_m

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(crop, origin="lower", cmap="hot", 
                     extent=[lx0, lx1, ly0, ly1], vmin=0, vmax=1)

            # Trajectory
            traj = np.array(g.trajectory)
            tx = traj[:, 0] - g.origin_x
            ty = traj[:, 1] - g.origin_y
            ax.plot(tx, ty, "c-", lw=1.5, alpha=0.8)
            ax.scatter(tx[0], ty[0], c="lime", s=100, marker="o", label="Start")
            ax.scatter(tx[-1], ty[-1], c="red", s=100, marker="x", label="End")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(
                f"Exploration Map\n"
                f"Distance: {g.total_distance():.1f}m | "
                f"Area: {g.covered_area():.1f}m² | "
                f"Points: {len(g.trajectory)}"
            )
            ax.set_aspect("equal")
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            self.get_logger().info(f"Map saved → {self.save_path}")

        except Exception as e:
            self.get_logger().warn(f"Failed to save map: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()