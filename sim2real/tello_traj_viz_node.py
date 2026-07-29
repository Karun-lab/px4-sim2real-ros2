#!/usr/bin/env python3
"""
tello_traj_viz_node.py
======================
Standalone ROS 2 node that displays a live two-panel matplotlib trajectory
plot. Runs in a completely separate process from the Tello bridge so the
plot's event loop never interferes with RC timing.

Subscribes:
    /tello/rc_command      (geometry_msgs/Twist)
        Physical RC values published by tello_icm_lstm_node.
            linear.x  = forward cm/s
            angular.z = yaw     deg/s
        Used for dead-reckoning integration.

    /uav/lstm_pose_est     (geometry_msgs/Point)
        XY displacement estimate from the LSTM pose head (metres from start).
        Published by icm_lstm_inference_node.

Plots:
    Left  — Dead reckoning (integrated from RC commands)
    Right — LSTM pose head estimate

Parameters (--ros-args -p)
    rc_topic        str    default /tello/rc_command
    pose_topic      str    default /uav/lstm_pose_est
    update_hz       float  default 5.0   (plot refresh rate)
    max_points      int    default 3000  (trajectory history length)
    dt_rc           float  default 0.05  (RC command period = 1/RC_HZ)
    window_title    str    default "Tello ICM+LSTM — Live Trajectory"

Usage:
    # In a new terminal, any machine on the same ROS 2 network:
    ros2 run sim2real tello_traj_viz_node

    # Override topics if needed:
    ros2 run sim2real tello_traj_viz_node \\
        --ros-args -p rc_topic:=/tello/rc_command \\
                   -p pose_topic:=/uav/lstm_pose_est

Dependencies:
    pip install matplotlib
"""

import sys
sys.path.insert(0, "/home/karun/venvs/rl/lib/python3.12/site-packages")

import math
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, Point


# =============================================================================
# NODE
# =============================================================================

class TelloTrajVizNode(Node):

    def __init__(self):
        super().__init__("tello_traj_viz_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("rc_topic",    "/tello/rc_command")
        self.declare_parameter("pose_topic",  "/uav/lstm_pose_est")
        self.declare_parameter("update_hz",   5.0)
        self.declare_parameter("max_points",  3000)
        self.declare_parameter("dt_rc",       0.05)
        self.declare_parameter("window_title","Tello ICM+LSTM — Live Trajectory")

        rc_topic    = self.get_parameter("rc_topic").value
        pose_topic  = self.get_parameter("pose_topic").value
        update_hz   = float(self.get_parameter("update_hz").value)
        max_points  = int(  self.get_parameter("max_points").value)
        self._dt_rc = float(self.get_parameter("dt_rc").value)
        win_title   = self.get_parameter("window_title").value

        # ── Dead reckoning state ───────────────────────────────────────────────
        self._lock    = threading.Lock()
        self._dr_x    = 0.0
        self._dr_y    = 0.0
        self._dr_yaw  = 0.0   # radians, integrated from yaw_deg_s

        self._dr_xs   = deque(maxlen=max_points)
        self._dr_ys   = deque(maxlen=max_points)
        self._lstm_xs = deque(maxlen=max_points)
        self._lstm_ys = deque(maxlen=max_points)
        for q in (self._dr_xs, self._dr_ys, self._lstm_xs, self._lstm_ys):
            q.append(0.0)

        # Flight stats
        self._total_dist_dr   = 0.0
        self._total_dist_lstm = 0.0
        self._step_count      = 0

        # ── ROS subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Twist, rc_topic, self._on_rc, qos_profile_sensor_data)
        self.create_subscription(
            Point, pose_topic, self._on_pose, qos_profile_sensor_data)

        # ── Matplotlib in main thread ─────────────────────────────────────────
        # The ROS spin runs in a background thread; matplotlib stays in main.
        self._update_hz  = update_hz
        self._win_title  = win_title
        self._spin_thread = threading.Thread(
            target=self._spin_ros, daemon=True, name="ROSSpin")
        self._spin_thread.start()

        self.get_logger().info(
            f"Trajectory visualiser\n"
            f"  RC commands  ← {rc_topic}\n"
            f"  LSTM pose    ← {pose_topic}\n"
            f"  Plot rate    : {update_hz} Hz\n"
            f"  Close window or Ctrl-C to exit."
        )

    # ── ROS callbacks (run in spin thread) ────────────────────────────────────

    def _on_rc(self, msg: Twist):
        """
        Integrate RC command into dead reckoning position.
        msg.linear.x  = forward cm/s (from tello_icm_lstm_node)
        msg.angular.z = yaw     deg/s
        """
        fwd_m_s   = msg.linear.x  / 100.0          # cm/s → m/s
        yaw_rad_s = math.radians(msg.angular.z)     # deg/s → rad/s

        with self._lock:
            self._dr_yaw  += yaw_rad_s * self._dt_rc
            dx             = fwd_m_s * math.cos(self._dr_yaw) * self._dt_rc
            dy             = fwd_m_s * math.sin(self._dr_yaw) * self._dt_rc
            self._dr_x    += dx
            self._dr_y    += dy
            self._dr_xs.append(self._dr_x)
            self._dr_ys.append(self._dr_y)
            self._total_dist_dr += math.hypot(dx, dy)
            self._step_count    += 1

    def _on_pose(self, msg: Point):
        """Receive LSTM pose head estimate."""
        with self._lock:
            prev_x = self._lstm_xs[-1] if self._lstm_xs else 0.0
            prev_y = self._lstm_ys[-1] if self._lstm_ys else 0.0
            self._lstm_xs.append(float(msg.x))
            self._lstm_ys.append(float(msg.y))
            self._total_dist_lstm += math.hypot(
                float(msg.x) - prev_x, float(msg.y) - prev_y)

    def _spin_ros(self):
        try:
            rclpy.spin(self)
        except Exception:
            pass

    # ── Matplotlib (main thread) ───────────────────────────────────────────────

    def run_plot(self):
        """
        Blocking call — runs the matplotlib event loop in the main thread.
        Returns when the window is closed.
        """
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("[viz] matplotlib not found. Install: pip install matplotlib")
            return

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(self._win_title, fontsize=13, fontweight="bold")
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        ax_dr   = fig.add_subplot(gs[0])
        ax_lstm = fig.add_subplot(gs[1])

        for ax, title, col in [
            (ax_dr,   "Dead Reckoning\n(integrated from RC commands)", "steelblue"),
            (ax_lstm, "LSTM Pose Head\n(model trajectory belief)",     "mediumpurple"),
        ]:
            ax.set_aspect("equal")
            ax.set_xlabel("East  X (m)", fontsize=9)
            ax.set_ylabel("North Y (m)", fontsize=9)
            ax.set_title(title, fontsize=10, pad=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=8)
            ax.plot(0, 0, "go", ms=10, label="start", zorder=6)
            ax.legend(fontsize=8, loc="upper left")

        line_dr,  = ax_dr.plot(  [], [], "-",  color="steelblue",    lw=1.3, alpha=0.85, label="path")
        head_dr,  = ax_dr.plot(  [], [], "r^", ms=10, zorder=7, label="current")
        trail_dr, = ax_dr.plot(  [], [], ".",  color="steelblue",    ms=2, alpha=0.3)

        line_lstm,  = ax_lstm.plot([], [], "-",  color="mediumpurple", lw=1.3, alpha=0.85, label="path")
        head_lstm,  = ax_lstm.plot([], [], "r^", ms=10, zorder=7, label="current")
        trail_lstm, = ax_lstm.plot([], [], ".",  color="mediumpurple", ms=2, alpha=0.3)

        stat_dr   = ax_dr.text(  0.02, 0.97, "", transform=ax_dr.transAxes,
                                 fontsize=7, va="top", family="monospace",
                                 bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="white", alpha=0.7))
        stat_lstm = ax_lstm.text(0.02, 0.97, "", transform=ax_lstm.transAxes,
                                 fontsize=7, va="top", family="monospace",
                                 bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="white", alpha=0.7))

        plt.ion()
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        period = 1.0 / self._update_hz

        while plt.fignum_exists(fig.number):
            t0 = time.time()

            with self._lock:
                drx  = list(self._dr_xs)
                dry  = list(self._dr_ys)
                lstx = list(self._lstm_xs)
                lsty = list(self._lstm_ys)
                dist_dr   = self._total_dist_dr
                dist_lstm = self._total_dist_lstm
                steps     = self._step_count

            # ── Dead reckoning ─────────────────────────────────────────────────
            if len(drx) > 1:
                line_dr.set_data(drx, dry)
                trail_dr.set_data(drx[:-1], dry[:-1])
                head_dr.set_data([drx[-1]], [dry[-1]])
                _autoscale(ax_dr, drx, dry)
                displacement = math.hypot(drx[-1], dry[-1])
                stat_dr.set_text(
                    f"pos : ({drx[-1]:+.2f}, {dry[-1]:+.2f}) m\n"
                    f"disp: {displacement:.2f} m from start\n"
                    f"dist: {dist_dr:.2f} m total path\n"
                    f"pts : {steps}")

            # ── LSTM ───────────────────────────────────────────────────────────
            if len(lstx) > 1:
                line_lstm.set_data(lstx, lsty)
                trail_lstm.set_data(lstx[:-1], lsty[:-1])
                head_lstm.set_data([lstx[-1]], [lsty[-1]])
                _autoscale(ax_lstm, lstx, lsty)
                displacement = math.hypot(lstx[-1], lsty[-1])
                stat_lstm.set_text(
                    f"pos : ({lstx[-1]:+.2f}, {lsty[-1]:+.2f}) m\n"
                    f"disp: {displacement:.2f} m from start\n"
                    f"dist: {dist_lstm:.2f} m total path\n"
                    f"pts : {len(lstx)}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            elapsed = time.time() - t0
            sleep_t = max(0.0, period - elapsed)
            time.sleep(sleep_t)

        plt.ioff()
        plt.close("all")


def _autoscale(ax, xs, ys, margin: float = 1.5):
    """Equal-aspect autoscale centred on trajectory bounding box."""
    if len(xs) < 2:
        return
    x_mid = (min(xs) + max(xs)) / 2.0
    y_mid = (min(ys) + max(ys)) / 2.0
    half  = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 + margin
    half  = max(half, 1.0)   # minimum ±1 m view even if drone hasn't moved
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)


# =============================================================================
# MAIN
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = TelloTrajVizNode()
    try:
        node.run_plot()   # blocks in main thread until window closed
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()