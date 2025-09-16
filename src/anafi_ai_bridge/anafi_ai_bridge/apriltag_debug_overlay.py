#!/usr/bin/env python3
# apriltag_debug_overlay.py
# 카메라 이미지 + /tag_detections + /camera_info → /apriltag/debug (오버레이 영상)

import math
import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from apriltag_msgs.msg import AprilTagDetectionArray

def quat_to_R(x, y, z, w):
    # 표준 쿼터니언 → 회전행렬
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float32)
    return R

class AprilTagOverlay(Node):
    def __init__(self):
        super().__init__("apriltag_debug_overlay")
        # 파라미터
        self.declare_parameter("image_topic", "/anafi/camera/image")
        self.declare_parameter("camera_info_topic", "/anafi/camera/camera_info")
        self.declare_parameter("tag_topic", "/tag_detections")
        self.declare_parameter("out_topic", "/apriltag/debug")
        self.declare_parameter("axis_len", 0.05)  # 5cm 축

        # QoS
        qos_cam = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # I/O
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.on_image, qos_cam)
        self.sub_info = self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value, self.on_info, qos_cam)
        self.sub_tag = self.create_subscription(
            AprilTagDetectionArray, self.get_parameter("tag_topic").value, self.on_tags, qos_cam)
        self.pub_dbg = self.create_publisher(
            Image, self.get_parameter("out_topic").value, qos_cam)

        # 내부상태
        self.K = None
        self.D = None
        self.detections = []  # (R, t) 리스트

    def on_info(self, msg: CameraInfo):
        # 카메라 내부행렬/왜곡계수
        self.K = np.array(msg.k, dtype=np.float32).reshape(3,3)
        self.D = np.array(msg.d, dtype=np.float32).reshape(-1,)

    def on_tags(self, msg: AprilTagDetectionArray):
        dets = []
        for d in msg.detections:
            # 다양한 메시지 변형을 호환하려 시도
            try:
                p = d.pose.pose.pose  # PoseWithCovarianceStamped
            except Exception:
                try:
                    p = d.pose.pose      # PoseStamped
                except Exception:
                    p = d.pose           # Pose
            t = np.array([[p.position.x],
                          [p.position.y],
                          [p.position.z]], dtype=np.float32)
            q = p.orientation
            R = quat_to_R(q.x, q.y, q.z, q.w)
            dets.append((R, t))
        self.detections = dets

    def on_image(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.K is None or self.D is None or not self.detections:
            # 정보 없으면 원본만 내보내기
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
            return

        dbg = img.copy()
        axis_len = float(self.get_parameter("axis_len").value)

        # 3D 축 (원점, x, y, z)
        obj_pts = np.float32([[0,0,0],
                              [axis_len,0,0],
                              [0,axis_len,0],
                              [0,0,axis_len]]).reshape(-1,3)

        for R, t in self.detections:
            # OpenCV에 맞는 rvec
            rvec, _ = cv2.Rodrigues(R)
            # 투영
            img_pts, _ = cv2.projectPoints(obj_pts, rvec, t, self.K, self.D)
            p0 = tuple(img_pts[0].ravel().astype(int))
            px = tuple(img_pts[1].ravel().astype(int))
            py = tuple(img_pts[2].ravel().astype(int))
            pz = tuple(img_pts[3].ravel().astype(int))
            # 축 그리기 (X=R, Y=G, Z=B)
            cv2.line(dbg, p0, px, (0,0,255), 2)
            cv2.line(dbg, p0, py, (0,255,0), 2)
            cv2.line(dbg, p0, pz, (255,0,0), 2)
            cv2.circle(dbg, p0, 4, (255,255,255), -1)

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))

def main():
    rclpy.init()
    node = AprilTagOverlay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
