#!/usr/bin/env python3
"""
icm_command_monitor.py
========================
ROS 2 GUI tool for monitoring ICM commands and PX4 trajectory setpoints.
Provides real-time visualization with sliders and plots for debugging.

Subscribes:
    /uav/action_cmd        (geometry_msgs/Twist) - ICM commands
    /fmu/in/trajectory_setpoint (px4_msgs/TrajectorySetpoint) - PX4 setpoints

Features:
    - Real-time sliders showing command values
    - Time-series plots for velocity and yaw
    - Command history tracking
    - Status indicators for OFFBOARD/ARMED
"""

import sys
import threading
import time
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist
try:
    from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleStatus
except ImportError:
    TrajectorySetpoint = None
    OffboardControlMode = None
    VehicleStatus = None

# PyQt5 imports
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QSlider, QGroupBox, QGridLayout, QPushButton,
        QCheckBox, QTabWidget, QFrame
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QFont, QColor, QPalette
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"Error importing PyQt5: {e}")
    print("Install with: pip install PyQt5 matplotlib")
    sys.exit(1)


# ── QoS for PX4 topics ────────────────────────────────────────────────────
PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


# ── ROS2 Node Wrapper for Qt ─────────────────────────────────────────────
class ROSNodeThread(QThread):
    """Run ROS2 node in separate thread."""
    
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.running = True
        
    def run(self):
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            self.msleep(10)  # 10ms sleep to prevent CPU hogging
            
    def stop(self):
        self.running = False


# ── Data Storage ──────────────────────────────────────────────────────────
class CommandData:
    """Thread-safe data storage for commands and setpoints."""
    
    def __init__(self, history_length=500):
        self.history_length = history_length
        self.lock = threading.Lock()
        
        # Current values
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.cmd_yaw_rate = 0.0
        
        self.setpoint_vx = 0.0
        self.setpoint_vy = 0.0
        self.setpoint_vz = 0.0
        self.setpoint_yaw_rate = 0.0
        
        # History for plotting
        self.timestamps = deque(maxlen=history_length)
        self.cmd_vx_history = deque(maxlen=history_length)
        self.cmd_yaw_history = deque(maxlen=history_length)
        self.sp_vx_history = deque(maxlen=history_length)
        self.sp_yaw_history = deque(maxlen=history_length)
        self.sp_vz_history = deque(maxlen=history_length)
        
        # Status
        self.offboard = False
        self.armed = False
        self.cmd_age = 0.0
        self.last_cmd_time = time.time()
        
    def update_cmd(self, msg: Twist):
        with self.lock:
            self.cmd_vx = msg.linear.x
            self.cmd_vy = msg.linear.y
            self.cmd_vz = msg.linear.z
            self.cmd_yaw_rate = msg.angular.z
            self.last_cmd_time = time.time()
            
            # Add to history
            now = time.time()
            self.timestamps.append(now)
            self.cmd_vx_history.append(msg.linear.x)
            self.cmd_yaw_history.append(msg.angular.z)
            
    def update_setpoint(self, msg):
        with self.lock:
            if hasattr(msg, 'velocity') and len(msg.velocity) >= 3:
                self.setpoint_vx = msg.velocity[0]
                self.setpoint_vy = msg.velocity[1]
                self.setpoint_vz = msg.velocity[2]
            if hasattr(msg, 'yawspeed'):
                self.setpoint_yaw_rate = msg.yawspeed
                
            # Add to history (if we have commands to align with)
            if len(self.timestamps) > 0:
                now = time.time()
                # Use the latest timestamp or current time
                ts = self.timestamps[-1] if len(self.timestamps) > 0 else now
                self.sp_vx_history.append(self.setpoint_vx)
                self.sp_yaw_history.append(self.setpoint_yaw_rate)
                self.sp_vz_history.append(self.setpoint_vz)
                
    def update_status(self, offboard: bool, armed: bool):
        with self.lock:
            self.offboard = offboard
            self.armed = armed
            
    def get_data(self):
        with self.lock:
            # Calculate command age
            self.cmd_age = time.time() - self.last_cmd_time
            return {
                'cmd_vx': self.cmd_vx,
                'cmd_vy': self.cmd_vy,
                'cmd_vz': self.cmd_vz,
                'cmd_yaw': self.cmd_yaw_rate,
                'sp_vx': self.setpoint_vx,
                'sp_vy': self.setpoint_vy,
                'sp_vz': self.setpoint_vz,
                'sp_yaw': self.setpoint_yaw_rate,
                'offboard': self.offboard,
                'armed': self.armed,
                'cmd_age': self.cmd_age,
                'timestamps': list(self.timestamps),
                'cmd_vx_hist': list(self.cmd_vx_history),
                'cmd_yaw_hist': list(self.cmd_yaw_history),
                'sp_vx_hist': list(self.sp_vx_history),
                'sp_yaw_hist': list(self.sp_yaw_history),
                'sp_vz_hist': list(self.sp_vz_history),
            }


# ── Monitor Node ──────────────────────────────────────────────────────────
class ICMCommandMonitorNode(Node):
    """ROS2 node that subscribes to command topics."""
    
    def __init__(self, data: CommandData):
        super().__init__("icm_command_monitor")
        self.data = data
        
        # Subscribers
        self.create_subscription(
            Twist, "/uav/action_cmd",
            self._on_cmd, 10)
            
        self.create_subscription(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint",
            self._on_setpoint, PX4_QOS)
            
        # Status subscribers if available
        if VehicleStatus is not None:
            self.create_subscription(
                VehicleStatus, "/fmu/out/vehicle_status",
                self._on_status, PX4_QOS)
                
        self.get_logger().info("ICM Command Monitor node started")
        
    def _on_cmd(self, msg: Twist):
        self.data.update_cmd(msg)
        
    def _on_setpoint(self, msg: TrajectorySetpoint):
        self.data.update_setpoint(msg)
        
    def _on_status(self, msg: VehicleStatus):
        offboard = (msg.nav_state == 14)  # NAV_STATE_OFFBOARD
        armed = (msg.arming_state == 2)   # ARMING_STATE_ARMED
        self.data.update_status(offboard, armed)


# ── Main GUI Window ──────────────────────────────────────────────────────
class MonitorWindow(QMainWindow):
    """Main GUI window for command monitoring."""
    
    def __init__(self, data: CommandData):
        super().__init__()
        self.data = data
        self.setWindowTitle("ICM Command Monitor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Setup UI
        self._setup_ui()
        
        # Timer for updating UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(50)  # 20 Hz
        
        # Plot timer for smoother updates
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plots)
        self.plot_timer.start(100)  # 10 Hz
        
    def _setup_ui(self):
        """Create the GUI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ── Status Bar ──────────────────────────────────────────────────────
        status_layout = QHBoxLayout()
        self.status_label = QLabel("⏳ Waiting for data...")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.status_label)
        
        self.offboard_label = QLabel("OFFBOARD: ❌")
        self.offboard_label.setFont(QFont("Arial", 11))
        status_layout.addWidget(self.offboard_label)
        
        self.armed_label = QLabel("ARMED: ❌")
        self.armed_label.setFont(QFont("Arial", 11))
        status_layout.addWidget(self.armed_label)
        
        self.cmd_age_label = QLabel("CMD Age: 0.00s")
        self.cmd_age_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.cmd_age_label)
        
        status_layout.addStretch()
        main_layout.addLayout(status_layout)
        
        # ── Tab Widget ──────────────────────────────────────────────────────
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Tab 1: Sliders Dashboard
        sliders_tab = QWidget()
        tabs.addTab(sliders_tab, "Sliders")
        self._setup_sliders_tab(sliders_tab)
        
        # Tab 2: Plots
        plots_tab = QWidget()
        tabs.addTab(plots_tab, "Plots")
        self._setup_plots_tab(plots_tab)
        
        # Tab 3: Numeric Values
        numeric_tab = QWidget()
        tabs.addTab(numeric_tab, "Values")
        self._setup_numeric_tab(numeric_tab)
        
    def _setup_sliders_tab(self, parent):
        """Setup the sliders dashboard tab."""
        layout = QGridLayout(parent)
        
        # ── Command Group ──────────────────────────────────────────────────
        cmd_group = QGroupBox("ICM Commands (/uav/action_cmd)")
        cmd_layout = QGridLayout(cmd_group)
        
        # VX Slider
        cmd_layout.addWidget(QLabel("Forward (vx):"), 0, 0)
        self.cmd_vx_slider = QSlider(Qt.Horizontal)
        self.cmd_vx_slider.setRange(-100, 100)
        self.cmd_vx_slider.setEnabled(False)
        cmd_layout.addWidget(self.cmd_vx_slider, 0, 1)
        self.cmd_vx_value = QLabel("0.00")
        self.cmd_vx_value.setFont(QFont("Arial", 10, QFont.Bold))
        cmd_layout.addWidget(self.cmd_vx_value, 0, 2)
        
        # Yaw Slider
        cmd_layout.addWidget(QLabel("Yaw Rate:"), 1, 0)
        self.cmd_yaw_slider = QSlider(Qt.Horizontal)
        self.cmd_yaw_slider.setRange(-100, 100)
        self.cmd_yaw_slider.setEnabled(False)
        cmd_layout.addWidget(self.cmd_yaw_slider, 1, 1)
        self.cmd_yaw_value = QLabel("0.00")
        self.cmd_yaw_value.setFont(QFont("Arial", 10, QFont.Bold))
        cmd_layout.addWidget(self.cmd_yaw_value, 1, 2)
        
        # VZ Slider (altitude)
        cmd_layout.addWidget(QLabel("Altitude (vz):"), 2, 0)
        self.cmd_vz_slider = QSlider(Qt.Horizontal)
        self.cmd_vz_slider.setRange(-100, 100)
        self.cmd_vz_slider.setEnabled(False)
        cmd_layout.addWidget(self.cmd_vz_slider, 2, 1)
        self.cmd_vz_value = QLabel("0.00")
        self.cmd_vz_value.setFont(QFont("Arial", 10, QFont.Bold))
        cmd_layout.addWidget(self.cmd_vz_value, 2, 2)
        
        cmd_layout.setColumnStretch(1, 1)
        layout.addWidget(cmd_group, 0, 0, 2, 1)
        
        # ── PX4 Setpoints Group ──────────────────────────────────────────
        sp_group = QGroupBox("PX4 Setpoints (/fmu/in/trajectory_setpoint)")
        sp_layout = QGridLayout(sp_group)
        
        # VX Setpoint
        sp_layout.addWidget(QLabel("Forward (vx):"), 0, 0)
        self.sp_vx_slider = QSlider(Qt.Horizontal)
        self.sp_vx_slider.setRange(-100, 100)
        self.sp_vx_slider.setEnabled(False)
        sp_layout.addWidget(self.sp_vx_slider, 0, 1)
        self.sp_vx_value = QLabel("0.00")
        self.sp_vx_value.setFont(QFont("Arial", 10, QFont.Bold))
        sp_layout.addWidget(self.sp_vx_value, 0, 2)
        
        # Yaw Setpoint
        sp_layout.addWidget(QLabel("Yaw Rate:"), 1, 0)
        self.sp_yaw_slider = QSlider(Qt.Horizontal)
        self.sp_yaw_slider.setRange(-100, 100)
        self.sp_yaw_slider.setEnabled(False)
        sp_layout.addWidget(self.sp_yaw_slider, 1, 1)
        self.sp_yaw_value = QLabel("0.00")
        self.sp_yaw_value.setFont(QFont("Arial", 10, QFont.Bold))
        sp_layout.addWidget(self.sp_yaw_value, 1, 2)
        
        # VZ Setpoint (altitude)
        sp_layout.addWidget(QLabel("Altitude (vz):"), 2, 0)
        self.sp_vz_slider = QSlider(Qt.Horizontal)
        self.sp_vz_slider.setRange(-100, 100)
        self.sp_vz_slider.setEnabled(False)
        sp_layout.addWidget(self.sp_vz_slider, 2, 1)
        self.sp_vz_value = QLabel("0.00")
        self.sp_vz_value.setFont(QFont("Arial", 10, QFont.Bold))
        sp_layout.addWidget(self.sp_vz_value, 2, 2)
        
        sp_layout.setColumnStretch(1, 1)
        layout.addWidget(sp_group, 0, 1, 2, 1)
        
        # ── Comparison Group ───────────────────────────────────────────────
        comp_group = QGroupBox("Command vs Setpoint Comparison")
        comp_layout = QVBoxLayout(comp_group)
        
        self.comp_labels = []
        for label in ["Forward (vx):", "Yaw Rate:", "Altitude (vz):"]:
            lbl = QLabel(f"{label}  CMD: 0.00  →  SP: 0.00")
            lbl.setFont(QFont("Arial", 10))
            comp_layout.addWidget(lbl)
            self.comp_labels.append(lbl)
        
        layout.addWidget(comp_group, 2, 0, 1, 2)
        
        # Set column/row stretches
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        
    def _setup_plots_tab(self, parent):
        """Setup the plots tab."""
        layout = QVBoxLayout(parent)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Plot controls
        controls_layout = QHBoxLayout()
        
        self.pause_checkbox = QCheckBox("Pause")
        controls_layout.addWidget(self.pause_checkbox)
        
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self._clear_history)
        controls_layout.addWidget(self.clear_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Create subplots
        self.ax1 = self.figure.add_subplot(3, 1, 1)
        self.ax2 = self.figure.add_subplot(3, 1, 2)
        self.ax3 = self.figure.add_subplot(3, 1, 3)
        
        self.figure.tight_layout(pad=2.0)
        
        # Plot lines
        self.line_cmd_vx = None
        self.line_sp_vx = None
        self.line_cmd_yaw = None
        self.line_sp_yaw = None
        self.line_sp_vz = None
        
    def _setup_numeric_tab(self, parent):
        """Setup the numeric values tab."""
        layout = QGridLayout(parent)
        
        # Create a table-like display
        headers = ["Parameter", "Command", "Setpoint", "Difference"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(label, 0, col)
        
        # Value rows
        self.numeric_labels = {}
        row = 1
        for name, key_cmd, key_sp in [
            ("Forward (vx)", "cmd_vx", "sp_vx"),
            ("Lateral (vy)", "cmd_vy", "sp_vy"),
            ("Vertical (vz)", "cmd_vz", "sp_vz"),
            ("Yaw Rate", "cmd_yaw", "sp_yaw"),
        ]:
            layout.addWidget(QLabel(name), row, 0)
            
            cmd_lbl = QLabel("0.000")
            cmd_lbl.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(cmd_lbl, row, 1)
            
            sp_lbl = QLabel("0.000")
            sp_lbl.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(sp_lbl, row, 2)
            
            diff_lbl = QLabel("0.000")
            diff_lbl.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(diff_lbl, row, 3)
            
            self.numeric_labels[key_cmd] = cmd_lbl
            self.numeric_labels[key_sp] = sp_lbl
            self.numeric_labels[f"{key_cmd}_diff"] = diff_lbl
            row += 1
            
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        
    # ── UI Update Methods ──────────────────────────────────────────────────
    def _update_ui(self):
        """Update the UI elements with latest data."""
        data = self.data.get_data()
        
        # Update status
        offboard = data['offboard']
        armed = data['armed']
        cmd_age = data['cmd_age']
        
        status_text = "🟢 RECEIVING" if cmd_age < 0.5 else "🟡 STALE" if cmd_age < 2.0 else "🔴 TIMEOUT"
        self.status_label.setText(f"Status: {status_text}")
        
        self.offboard_label.setText(f"OFFBOARD: {'✅' if offboard else '❌'}")
        self.offboard_label.setStyleSheet(f"color: {'green' if offboard else 'red'}")
        
        self.armed_label.setText(f"ARMED: {'✅' if armed else '❌'}")
        self.armed_label.setStyleSheet(f"color: {'green' if armed else 'red'}")
        
        self.cmd_age_label.setText(f"CMD Age: {cmd_age:.2f}s")
        self.cmd_age_label.setStyleSheet(
            f"color: {'green' if cmd_age < 0.5 else 'orange' if cmd_age < 2.0 else 'red'}"
        )
        
        # Update sliders and values
        def update_slider(slider, label, value, scale=100):
            slider.setValue(int(value * scale))
            label.setText(f"{value:.2f}")
            
        update_slider(self.cmd_vx_slider, self.cmd_vx_value, data['cmd_vx'])
        update_slider(self.cmd_yaw_slider, self.cmd_yaw_value, data['cmd_yaw'])
        update_slider(self.cmd_vz_slider, self.cmd_vz_value, data['cmd_vz'])
        
        update_slider(self.sp_vx_slider, self.sp_vx_value, data['sp_vx'])
        update_slider(self.sp_yaw_slider, self.sp_yaw_value, data['sp_yaw'])
        update_slider(self.sp_vz_slider, self.sp_vz_value, data['sp_vz'])
        
        # Update comparison labels
        self.comp_labels[0].setText(
            f"Forward (vx):  CMD: {data['cmd_vx']:+.2f}  →  SP: {data['sp_vx']:+.2f}"
        )
        self.comp_labels[1].setText(
            f"Yaw Rate:      CMD: {data['cmd_yaw']:+.2f}  →  SP: {data['sp_yaw']:+.2f}"
        )
        self.comp_labels[2].setText(
            f"Altitude (vz): CMD: {data['cmd_vz']:+.2f}  →  SP: {data['sp_vz']:+.2f}"
        )
        
        # Update numeric values
        for key, lbl in self.numeric_labels.items():
            if key.startswith('cmd_'):
                val = data.get(key, 0.0)
                lbl.setText(f"{val:+.3f}")
            elif key.startswith('sp_'):
                val = data.get(key, 0.0)
                lbl.setText(f"{val:+.3f}")
            elif key.endswith('_diff'):
                cmd_key = key.replace('_diff', '')
                if cmd_key.startswith('cmd_'):
                    sp_key = cmd_key.replace('cmd_', 'sp_')
                    val = data.get(cmd_key, 0.0) - data.get(sp_key, 0.0)
                    lbl.setText(f"{val:+.3f}")
                    if abs(val) > 0.1:
                        lbl.setStyleSheet("color: orange")
                    else:
                        lbl.setStyleSheet("color: green")
        
    def _update_plots(self):
        """Update the matplotlib plots."""
        if self.pause_checkbox.isChecked():
            return
            
        data = self.data.get_data()
        timestamps = data['timestamps']
        
        if len(timestamps) < 2:
            return
            
        # Convert to numpy arrays for plotting
        t = np.array(timestamps) - timestamps[0]
        cmd_vx = np.array(data['cmd_vx_hist'])
        cmd_yaw = np.array(data['cmd_yaw_hist'])
        sp_vx = np.array(data['sp_vx_hist'])
        sp_yaw = np.array(data['sp_yaw_hist'])
        sp_vz = np.array(data['sp_vz_hist'])
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: Forward Velocity
        self.ax1.plot(t, cmd_vx, 'b-', linewidth=2, label='CMD vx')
        self.ax1.plot(t, sp_vx, 'r--', linewidth=2, label='SP vx')
        self.ax1.set_ylabel('Forward (m/s)')
        self.ax1.set_title('Forward Velocity')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(-1.5, 1.5)
        
        # Plot 2: Yaw Rate
        self.ax2.plot(t, cmd_yaw, 'b-', linewidth=2, label='CMD yaw')
        self.ax2.plot(t, sp_yaw, 'r--', linewidth=2, label='SP yaw')
        self.ax2.set_ylabel('Yaw Rate (rad/s)')
        self.ax2.set_title('Yaw Rate')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(-1.5, 1.5)
        
        # Plot 3: Altitude
        self.ax3.plot(t, sp_vz, 'g-', linewidth=2, label='SP vz')
        self.ax3.set_ylabel('Vertical (m/s)')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_title('Altitude Velocity')
        self.ax3.legend(loc='upper right')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(-1.0, 1.0)
        
        # Update plot limits
        if len(t) > 0:
            t_min = 0
            t_max = max(t[-1], 10.0)  # At least 10 seconds window
            self.ax1.set_xlim(t_min, t_max)
            self.ax2.set_xlim(t_min, t_max)
            self.ax3.set_xlim(t_min, t_max)
        
        self.canvas.draw()
        
    def _clear_history(self):
        """Clear the history data."""
        self.data.timestamps.clear()
        self.data.cmd_vx_history.clear()
        self.data.cmd_yaw_history.clear()
        self.data.sp_vx_history.clear()
        self.data.sp_yaw_history.clear()
        self.data.sp_vz_history.clear()


# ── Main Application ──────────────────────────────────────────────────────
def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create data storage
    data = CommandData(history_length=500)
    
    # Create ROS node
    node = ICMCommandMonitorNode(data)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show window
    window = MonitorWindow(data)
    window.show()
    
    # Start ROS spinning in separate thread
    ros_thread = ROSNodeThread(node)
    ros_thread.start()
    
    # Run Qt application
    try:
        sys.exit(app.exec_())
    finally:
        ros_thread.stop()
        ros_thread.wait()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()