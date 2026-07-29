#!/usr/bin/env python3
"""
tello_supervisor_node.py
=========================
Lightweight ROS 2 supervisory controller for the PPO+ICM Tello exploration
policy. Sits *between* the policy and `tello_icm_bridge_node`, arbitrating
commands. It never modifies the policy and passes its commands through
unchanged the vast majority of the time.

Why a separate node instead of editing the policy or the bridge
------------------------------------------------------------------
The policy is a learned artifact — patching its behaviour in-place is fragile
and hides the real bias from evaluation. The bridge is a thin hardware driver
and should stay that way. A supervisor that only *arbitrates* the command
stream is the minimal-footprint place to add exploration-level logic
(loop/coverage detection, escape maneuvers, mission termination) without
retraining or touching either.

Wiring (no code changes to the policy needed — just a topic remap)
------------------------------------------------------------------
    policy  --publishes-->  /uav/policy_cmd   (remap the policy's normal
                                                /uav/action_cmd output here
                                                with a launch-file remap,
                                                e.g. -r /uav/action_cmd:=/uav/policy_cmd)
    this node --subscribes--> /uav/policy_cmd
    this node --publishes-->  /uav/action_cmd  (bridge's normal input topic,
                                                 unchanged)
    this node --publishes--> /uav/land_request (std_msgs/Empty; bridge patch
                                                 grounds the drone on receipt)

Everything downstream (bridge) is untouched in its normal-operation code
path; only a small additive land-request hook was patched into the bridge
(see tello_icm_bridge_node_patched.py), plus a pre-existing yaw-deadzone bug
fix — without that fix, this supervisor's own corrective yaw commands would
have been silently zeroed by the bridge too. See PATCH NOTES in that file.

What this node does
------------------------------------------------------------------
1. Dead-reckons a 2D (x, y, theta) trajectory from the commands it forwards,
   replicating the bridge's own command-shaping math so the estimate tracks
   what the Tello actually receives (not the raw policy output).
2. Maintains a sparse (dict-based) occupancy/visit grid — O(cells actually
   visited), not O(map size) — for coverage tracking and frontier scoring.
3. Publishes:
     - nav_msgs/OccupancyGrid  /supervisor/coverage   (heatmap, ~1 Hz)
     - nav_msgs/Path           /supervisor/trajectory (live XY path, rviz)
     - a periodic matplotlib PNG snapshot to disk (low rate; off the hot path)
4. Detects "room sufficiently explored": coverage growth (unique cells / time)
   plateaus after a minimum area has been covered -> publishes a land request
   and stops forwarding further motion commands.
5. Detects "stuck in a small loop / corridor oscillation": low path
   efficiency (net displacement / path length) over a sliding window ->
   injects a short corrective yaw maneuver, then hands control straight back
   to the policy.
6. The escape heading is NOT a fixed "always turn right" heuristic — it's
   chosen by scoring candidate headings against the *unvisited* cells in the
   occupancy grid (frontier score) so interventions push the drone toward
   unexplored space. Because the known failure mode is a left-yaw bias, ties
   / near-ties in frontier score are broken in favour of a right turn, but a
   strong unexplored-frontier signal on the left will still win.
7. Designed to be robust to dead-reckoning drift: all decisions are LOCAL /
   relative (path efficiency over a short window, revisits of nearby cells,
   short-horizon frontier scoring) rather than relying on long-horizon
   absolute position accuracy, which is exactly what accumulates drift.

Parameters (all tunable via --ros-args -p)
------------------------------------------------------------------
    policy_cmd_topic      str    default /uav/policy_cmd
    action_cmd_topic      str    default /uav/action_cmd
    land_topic            str    default /uav/land_request
    max_forward_cm_s      int    default 40     (must match bridge)
    max_yaw_deg_s         int    default 40     (must match bridge)
    cmd_timeout_s         float  default 0.5    (must match bridge)
    loop_hz               float  default 10.0   supervisor tick rate
    cell_size_m           float  default 0.25   occupancy grid resolution
    min_area_explored_m2  float  default 4.0    floor before "explored" can fire
    coverage_window_s     float  default 6.0    plateau-check window
    coverage_growth_eps   float  default 1.0    min new cells/window to count as "still growing"
    stall_confirm_windows int    default 3      consecutive plateaued windows before landing
    osc_window_s          float  default 8.0    oscillation sliding window
    osc_min_path_m        float  default 1.5    ignore near-stationary windows
    osc_efficiency_thresh float  default 0.22   net_disp/path_len below this = stuck
    intervention_cooldown_s float default 5.0   min gap between interventions
    intervention_duration_s float default 1.4   how long an escape yaw lasts
    intervention_yaw_norm  float default 0.9    |yaw_norm| commanded during escape
    frontier_lookahead_m  float  default 2.0    how far ahead frontier scoring looks
    frontier_num_headings int    default 12     candidate headings sampled
    plot_path             str    default /tmp/tello_supervisor_trajectory.png
    plot_period_s         float  default 15.0   PNG snapshot period (0 disables)

Dependencies:
    pip install numpy matplotlib
"""

import math
import time
import bisect
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Empty
from nav_msgs.msg import OccupancyGrid, Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ── Helpers ────────────────────────────────────────────────────────────────

def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def shape_command(vx_norm: float, yaw_norm: float,
                   max_fwd_cm_s: float, max_yaw_deg_s: float):
    """Replicates tello_icm_bridge_node._control_loop's shaping so the
    dead-reckoner integrates what the Tello actually receives, not the raw
    normalised command. Mirrors the bridge's (post-bugfix) logic exactly —
    keep in sync if the bridge shaping changes.

    Returns (fwd_cm_s, yaw_deg_s).
    """
    vx_n = float(np.clip(vx_norm * 1.4, -0.2, 1.0))
    raw_yaw_n = float(np.clip(yaw_norm, -1.0, 1.0))
    yaw_n = float(np.clip(raw_yaw_n * 0.07, -1.0, 1.0))

    if vx_n > 0.7:
        yaw_n = 0.0
    if abs(raw_yaw_n) < 0.4:
        yaw_n = 0.0
    if abs(raw_yaw_n) < 0.7 and vx_n < 0.2:
        vx_n = 0.2

    fwd = float(np.clip(vx_n * max_fwd_cm_s, -100, 100))
    yaw = float(np.clip(yaw_n * max_yaw_deg_s, -100, 100))
    return fwd, yaw


# ── Sparse occupancy / visit grid ────────────────────────────────────────────

class SparseVisitGrid:
    """Dict-keyed grid: O(cells actually visited) memory, not O(map area).
    Robust to unknown/unbounded exploration extent and cheap to update."""

    def __init__(self, cell_size_m: float):
        self.cell = cell_size_m
        self.visits: dict[tuple[int, int], int] = {}

    def cell_of(self, x: float, y: float):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def mark(self, x: float, y: float):
        k = self.cell_of(x, y)
        self.visits[k] = self.visits.get(k, 0) + 1
        return k

    def unique_count(self) -> int:
        return len(self.visits)

    def area_m2(self) -> float:
        return self.unique_count() * (self.cell ** 2)

    def frontier_score(self, x: float, y: float, heading: float,
                        lookahead_m: float) -> float:
        """Score a heading by how much UNVISITED area lies ahead of it.
        Cheap ray-march in grid-cell steps; unique unvisited cells only."""
        steps = max(1, int(lookahead_m / self.cell))
        seen = set()
        score = 0.0
        for s in range(1, steps + 1):
            d = s * self.cell
            px = x + d * math.cos(heading)
            py = y + d * math.sin(heading)
            k = self.cell_of(px, py)
            if k in seen:
                continue
            seen.add(k)
            if k not in self.visits:
                score += 1.0
            # slightly de-weight cells we've visited a lot (well-explored)
            else:
                score -= 0.15 * min(self.visits[k], 3)
        return score

    def to_grid_array(self, margin_cells: int = 2):
        """Dense array for OccupancyGrid publishing — built on demand only,
        at whatever bounding box the trajectory currently occupies."""
        if not self.visits:
            return None, 0, 0, (0, 0)
        xs = [k[0] for k in self.visits]
        ys = [k[1] for k in self.visits]
        min_x, max_x = min(xs) - margin_cells, max(xs) + margin_cells
        min_y, max_y = min(ys) - margin_cells, max(ys) + margin_cells
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        arr = np.zeros((h, w), dtype=np.int8)
        max_v = max(self.visits.values())
        for (gx, gy), v in self.visits.items():
            arr[gy - min_y, gx - min_x] = int(np.clip(100.0 * v / max_v, 0, 100))
        return arr, w, h, (min_x, min_y)


# ── Node ──────────────────────────────────────────────────────────────────

class TelloSupervisorNode(Node):
    def __init__(self):
        super().__init__("tello_supervisor_node")

        # ── Parameters ──────────────────────────────────────────────────
        p = self.declare_parameter
        p("policy_cmd_topic", "/uav/policy_cmd")
        p("action_cmd_topic", "/uav/action_cmd")
        p("land_topic", "/uav/land_request")
        p("max_forward_cm_s", 40)
        p("max_yaw_deg_s", 40)
        p("cmd_timeout_s", 0.5)
        p("loop_hz", 10.0)
        p("cell_size_m", 0.25)
        p("min_area_explored_m2", 4.0)
        p("coverage_window_s", 6.0)
        p("coverage_growth_eps", 1.0)
        p("stall_confirm_windows", 3)
        p("osc_window_s", 8.0)
        p("osc_min_path_m", 1.5)
        p("osc_efficiency_thresh", 0.22)
        p("intervention_cooldown_s", 5.0)
        p("intervention_duration_s", 1.4)
        p("intervention_yaw_norm", 0.9)
        p("frontier_lookahead_m", 2.0)
        p("frontier_num_headings", 12)
        p("plot_path", "/tmp/tello_supervisor_trajectory.png")
        p("plot_period_s", 15.0)

        g = lambda n: self.get_parameter(n).value
        self._policy_topic = g("policy_cmd_topic")
        self._action_topic = g("action_cmd_topic")
        self._land_topic = g("land_topic")
        self._max_fwd = float(g("max_forward_cm_s"))
        self._max_yaw = float(g("max_yaw_deg_s"))
        self._cmd_to = float(g("cmd_timeout_s"))
        self._hz = float(g("loop_hz"))
        self._min_area = float(g("min_area_explored_m2"))
        self._cov_window_s = float(g("coverage_window_s"))
        self._cov_growth_eps = float(g("coverage_growth_eps"))
        self._stall_confirm = int(g("stall_confirm_windows"))
        self._osc_window_s = float(g("osc_window_s"))
        self._osc_min_path = float(g("osc_min_path_m"))
        self._osc_eff_thresh = float(g("osc_efficiency_thresh"))
        self._interv_cooldown = float(g("intervention_cooldown_s"))
        self._interv_duration = float(g("intervention_duration_s"))
        self._interv_yaw_norm = float(g("intervention_yaw_norm"))
        self._frontier_lookahead = float(g("frontier_lookahead_m"))
        self._frontier_n = int(g("frontier_num_headings"))
        self._plot_path = g("plot_path")
        self._plot_period = float(g("plot_period_s"))

        # ── Dead reckoning state ───────────────────────────────────────
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

        # ── Trajectory buffer (memory-bounded; lightweight) ─────────────
        # (t, x, y) samples, capped so memory stays flat over a long mission
        self._traj: deque = deque(maxlen=20000)

        # ── Occupancy / coverage ────────────────────────────────────────
        self._grid = SparseVisitGrid(float(g("cell_size_m")))
        self._coverage_hist: deque = deque()  # (t, unique_cell_count)
        self._stall_count = 0

        # ── Command state ────────────────────────────────────────────────
        self._last_policy_twist = Twist()
        self._last_policy_time = 0.0

        # ── Intervention / mission state ─────────────────────────────────
        self._intervention_end_t = 0.0
        self._intervention_yaw_sign = 1.0
        self._last_intervention_t = -1e9
        self._mission_complete = False
        self._last_plot_t = 0.0

        # ── ROS interfaces ──────────────────────────────────────────────
        self.sub_policy = self.create_subscription(
            Twist, self._policy_topic, self._on_policy_cmd, qos_profile_sensor_data)

        self.pub_action = self.create_publisher(Twist, self._action_topic, 10)
        self.pub_land = self.create_publisher(Empty, self._land_topic, 10)
        self.pub_coverage = self.create_publisher(OccupancyGrid, "/supervisor/coverage", 5)
        self.pub_path = self.create_publisher(Path, "/supervisor/trajectory", 5)

        self._t0 = time.time()
        self._last_tick_t = self._t0

        self.timer = self.create_timer(1.0 / self._hz, self._tick)
        self.viz_timer = self.create_timer(1.0, self._publish_viz)

        self.get_logger().info(
            f"Supervisor ready. policy:{self._policy_topic} -> action:{self._action_topic} "
            f"@ {self._hz:.0f}Hz, land on: {self._land_topic}"
        )

    # ── Policy command intake ────────────────────────────────────────────
    def _on_policy_cmd(self, msg: Twist):
        self._last_policy_twist = msg
        self._last_policy_time = time.time()

    # ── Main supervisory loop ────────────────────────────────────────────
    def _tick(self):
        if self._mission_complete:
            # Keep publishing zero so the drone stays grounded even if the
            # bridge patch / land_request hook wasn't applied for some reason.
            self.pub_action.publish(Twist())
            return

        now = time.time()
        dt = max(1e-3, now - self._last_tick_t)
        self._last_tick_t = now

        # 1) Decide which command to actually forward this tick.
        in_intervention = now < self._intervention_end_t
        if in_intervention:
            cmd = Twist()
            cmd.linear.x = 0.15  # slow crawl forward while turning, avoids a dead hover
            cmd.angular.z = self._intervention_yaw_sign * self._interv_yaw_norm
        else:
            policy_fresh = (now - self._last_policy_time) < self._cmd_to
            cmd = self._last_policy_twist if policy_fresh else Twist()

        self.pub_action.publish(cmd)

        # 2) Dead-reckon using the command actually sent (matches bridge shaping).
        fwd_cm_s, yaw_deg_s = shape_command(cmd.linear.x, cmd.angular.z,
                                             self._max_fwd, self._max_yaw)
        v_m_s = (fwd_cm_s / 100.0)
        w_rad_s = math.radians(yaw_deg_s)

        self._theta = wrap_angle(self._theta + w_rad_s * dt)
        self._x += v_m_s * math.cos(self._theta) * dt
        self._y += v_m_s * math.sin(self._theta) * dt

        self._traj.append((now, self._x, self._y))
        self._grid.mark(self._x, self._y)

        # 3) Run detectors.
        self._update_coverage_and_check_explored(now)
        if not self._mission_complete:
            self._check_oscillation_and_maybe_intervene(now)

    # ── Detector 1: room-explored via coverage plateau ───────────────────
    def _update_coverage_and_check_explored(self, now: float):
        self._coverage_hist.append((now, self._grid.unique_count()))
        cutoff = now - self._cov_window_s * (self._stall_confirm + 1)
        while self._coverage_hist and self._coverage_hist[0][0] < cutoff:
            self._coverage_hist.popleft()

        # Only evaluate once we have enough history for a full window.
        if not self._coverage_hist or (now - self._coverage_hist[0][0]) < self._cov_window_s:
            return
        if self._grid.area_m2() < self._min_area:
            return

        # growth over the most recent window
        times = [t for t, _ in self._coverage_hist]
        window_start_idx = bisect.bisect_left(times, now - self._cov_window_s)
        if window_start_idx >= len(self._coverage_hist):
            return
        count_then = self._coverage_hist[window_start_idx][1]
        count_now = self._coverage_hist[-1][1]
        growth = count_now - count_then

        if growth <= self._cov_growth_eps:
            self._stall_count += 1
        else:
            self._stall_count = 0

        if self._stall_count >= self._stall_confirm:
            self.get_logger().info(
                f"Coverage plateaued ({self._grid.area_m2():.1f} m^2 explored, "
                f"+{growth} cells over last {self._cov_window_s:.0f}s) — "
                f"treating room as sufficiently explored. Landing."
            )
            self._terminate_mission()

    # ── Detector 2: corridor oscillation / small repeated loop ───────────
    def _check_oscillation_and_maybe_intervene(self, now: float):
        if now < self._intervention_end_t:
            return  # already correcting
        if (now - self._last_intervention_t) < self._interv_cooldown:
            return  # cooling down

        window = [(t, x, y) for (t, x, y) in self._traj if t >= now - self._osc_window_s]
        if len(window) < 4:
            return

        path_len = 0.0
        for i in range(1, len(window)):
            _, x0, y0 = window[i - 1]
            _, x1, y1 = window[i]
            path_len += math.hypot(x1 - x0, y1 - y0)

        if path_len < self._osc_min_path:
            return  # basically stationary — not an oscillation, don't intervene

        _, xs, ys = window[0]
        _, xe, ye = window[-1]
        net_disp = math.hypot(xe - xs, ye - ys)
        efficiency = net_disp / (path_len + 1e-6)

        if efficiency >= self._osc_eff_thresh:
            return  # making real progress, leave the policy alone

        self.get_logger().info(
            f"Low path efficiency ({efficiency:.2f} over {path_len:.1f} m traveled) — "
            f"corridor oscillation / small loop suspected. Injecting escape yaw."
        )
        self._start_intervention(now)

    def _start_intervention(self, now: float):
        best_heading, best_score, right_score, left_score = self._pick_frontier_heading()
        turn = wrap_angle(best_heading - self._theta)

        # Tie-break bias: if candidates are within a small margin of each
        # other, prefer the turn direction that counters the known left-yaw
        # bias (right). A clearly better unexplored frontier on the left
        # still wins — this is not a fixed "always turn right" rule.
        margin = 0.75  # frontier-score units
        if abs(right_score - left_score) < margin:
            sign = 1.0  # right
        else:
            sign = 1.0 if turn >= 0 else -1.0

        self._intervention_yaw_sign = sign
        self._intervention_end_t = now + self._interv_duration
        self._last_intervention_t = now

    def _pick_frontier_heading(self):
        """Score candidate headings by unexplored area ahead of them.
        Returns (best_heading, best_score, aggregate_right_score, aggregate_left_score)."""
        n = self._frontier_n
        best_h, best_s = self._theta, -1e9
        right_s, left_s = 0.0, 0.0
        for i in range(n):
            h = wrap_angle(self._theta + 2.0 * math.pi * i / n)
            s = self._grid.frontier_score(self._x, self._y, h, self._frontier_lookahead)
            if s > best_s:
                best_h, best_s = h, s
            rel = wrap_angle(h - self._theta)
            if rel > 0:
                right_s = max(right_s, s)  # positive angular.z convention: CCW+; see note below
            else:
                left_s = max(left_s, s)
        return best_h, best_s, right_s, left_s

    # ── Mission termination ──────────────────────────────────────────────
    def _terminate_mission(self):
        self._mission_complete = True
        self.pub_action.publish(Twist())
        for _ in range(3):
            self.pub_land.publish(Empty())
        self._save_trajectory_plot(final=True)
        self.get_logger().info("Mission complete: land command issued.")

    # ── Visualization ─────────────────────────────────────────────────────
    def _publish_viz(self):
        self._publish_coverage_grid()
        self._publish_path()
        if self._plot_period > 0 and (time.time() - self._last_plot_t) >= self._plot_period:
            self._save_trajectory_plot(final=False)

    def _publish_coverage_grid(self):
        arr, w, h, origin_cells = self._grid.to_grid_array()
        if arr is None:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self._grid.cell
        msg.info.width = w
        msg.info.height = h
        msg.info.origin.position.x = origin_cells[0] * self._grid.cell
        msg.info.origin.position.y = origin_cells[1] * self._grid.cell
        msg.data = arr.flatten().tolist()
        self.pub_coverage.publish(msg)

    def _publish_path(self):
        if not self._traj:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        # Downsample for publishing cost; keep it cheap.
        stride = max(1, len(self._traj) // 500)
        for i, (_, x, y) in enumerate(self._traj):
            if i % stride != 0:
                continue
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def _save_trajectory_plot(self, final: bool):
        self._last_plot_t = time.time()
        if not _HAVE_MPL or not self._traj:
            return
        try:
            xs = [x for _, x, _ in self._traj]
            ys = [y for _, _, y in self._traj]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(xs, ys, linewidth=1.0, color="tab:blue", label="dead-reckoned path")
            ax.scatter([xs[0]], [ys[0]], color="green", zorder=5, label="start")
            ax.scatter([xs[-1]], [ys[-1]], color="red", zorder=5, label="current/end")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            title = "Tello dead-reckoned trajectory"
            if final:
                title += " (mission complete)"
            ax.set_title(title)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self._plot_path, dpi=120)
            plt.close(fig)
        except Exception as e:
            self.get_logger().warn(f"Trajectory plot save failed: {e}")

    def shutdown(self):
        self._save_trajectory_plot(final=True)


def main(args=None):
    rclpy.init(args=args)
    node = TelloSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()