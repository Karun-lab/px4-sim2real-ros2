#!/usr/bin/env python3


import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def build_gst_pipeline(port: int) -> str:
    return (
        f"udpsrc port={port} "
        f'caps="application/x-rtp, media=video, encoding-name=H264, payload=96" '
        f"! rtph264depay "
        f"! avdec_h264 "
        f"! videoconvert "
        f"! video/x-raw, format=BGR "
        f"! appsink drop=true max-buffers=1"
    )


class GstCameraNode(Node):

    def __init__(self):
        super().__init__("gst_camera_node")

        # Parameters
        self.declare_parameter("udp_port", 5000)
        self.port = self.get_parameter("udp_port").value

        # ROS2 publisher
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "/camera/rgb/image_raw", 10)

        # Open GStreamer stream
        pipeline = build_gst_pipeline(self.port)
        self.get_logger().info(f"Opening pipeline:\n{pipeline}")

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open GStreamer stream")
        else:
            self.get_logger().info("Stream opened successfully ✓")

        # Timer (~30 FPS)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)


    def timer_callback(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("No frame received", throttle_duration_sec=3.0)
            return

        # Convert OpenCV image → ROS2 Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        # Publish
        self.pub.publish(msg)


    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GstCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()