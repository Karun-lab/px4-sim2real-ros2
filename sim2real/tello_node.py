#Flight data check code.

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3

from djitellopy import Tello


class TelloDataNode(Node):

    def __init__(self):
        super().__init__('tello_data_node')

        # ---------------- DRONE INIT ----------------
        self.drone = Tello()
        self.drone.connect()

        self.get_logger().info(f"Battery: {self.drone.get_battery()}%")

        self.drone.streamon()
        self.frame_reader = self.drone.get_frame_read()

        # ---------------- ROS PUBLISHERS ----------------
        self.image_pub = self.create_publisher(Image, '/tello/image', 10)
        self.status_pub = self.create_publisher(Float32MultiArray, '/tello/status', 10)
        self.imu_pub = self.create_publisher(Vector3, '/tello/imu_like', 10)

        self.bridge = CvBridge()

        # ---------------- LOOP ----------------
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 FPS

        self.frame_count = 0

    # ---------------- STATUS ----------------
    def get_status_data(self):

        return [
            float(self.drone.get_flight_time()),
            float(self.drone.get_battery()),
            float(self.drone.get_height()),
            float(self.drone.get_distance_tof()),
            float(self.drone.get_lowest_temperature()),
            float(self.drone.get_highest_temperature()),
            float(self.drone.get_temperature()),
            float(self.drone.get_barometer())
        ]

    # ---------------- IMU-LIKE DATA ----------------
    def get_orientation_data(self):

        imu = Vector3()
        imu.x = float(self.drone.get_pitch())
        imu.y = float(self.drone.get_roll())
        imu.z = float(self.drone.get_yaw())
        return imu

    # ---------------- MAIN LOOP ----------------
    def timer_callback(self):

        frame = self.frame_reader.frame

        if frame is None:
            return

        self.frame_count += 1

        # -------- IMAGE --------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tello_camera"

        self.image_pub.publish(msg)

        # -------- STATUS --------
        status = Float32MultiArray()
        status.data = self.get_status_data()
        self.status_pub.publish(status)

        # -------- IMU --------
        if self.frame_count % 30 == 0:
            imu = self.get_orientation_data()
            self.imu_pub.publish(imu)

            self.get_logger().info(
                f"Battery: {self.drone.get_battery()}% | "
                f"Height: {self.drone.get_height()} cm"
            )

    def destroy_node(self):
        self.drone.streamoff()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TelloDataNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()