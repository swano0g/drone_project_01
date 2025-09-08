#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class FollowCrazyflie(Node):
    """
    - Subscribe:  /camera/image   (sensor_msgs/Image, bgr8)
    - Publish:    /control/pcmd   (geometry_msgs/Twist)
    - Detect Crazyflie via HSV color mask (BLUE LED default)
    """

    def __init__(self):
        super().__init__("creative_behavior_follow_cf")
        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----------------- Parameters -----------------
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("cmd_topic", "/control/pcmd")

        # 🔵 BLUE HSV range (OpenCV HSV: H:0–179)
        # - H: 100~130가 보통 파랑에 잘 맞음
        # - S: 120 이상(채도 낮은 파랑/난반사 방지)
        # - V: 70 이상(어두운 잡음 억제, 필요시 40~60까지 낮출 수 있음)
        self.declare_parameter("hsv_low",  [100, 120, 70])
        self.declare_parameter("hsv_high", [130, 255, 255])

        # 목표 면적(화면 대비 비율)과 데드밴드
        self.declare_parameter("target_area_ratio", 0.008)  # 파랑 LED가 작을 수 있어 0.8%로 살짝 낮춤
        self.declare_parameter("area_deadband", 0.002)

        # 제어 게인
        self.declare_parameter("kp_yaw",   60.0)
        self.declare_parameter("kp_pitch", 12.0)
        self.declare_parameter("kp_alt",    1.2)

        # 출력 상한
        self.declare_parameter("max_yaw_rate",       120.0)  # deg/s
        self.declare_parameter("max_pitch_deg",       12.0)  # deg
        self.declare_parameter("max_vertical_speed",   1.2)  # m/s

        # 데드밴드/잡음 제거
        self.declare_parameter("center_deadband_px", 24)
        self.declare_parameter("min_blob_area_px",   120)    # 파란 LED 점 잡음 방지 위해 약간 상향

        # 미검출 시 수색 회전
        self.declare_parameter("search_yaw_rate", 20.0)
        self.declare_parameter("lost_timeout", 0.6)

        # 읽기
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.cmd_topic   = self.get_parameter("cmd_topic").get_parameter_value().string_value

        self.hsv_low  = np.array(self.get_parameter("hsv_low").value, dtype=np.uint8)
        self.hsv_high = np.array(self.get_parameter("hsv_high").value, dtype=np.uint8)

        self.target_area_ratio = float(self.get_parameter("target_area_ratio").value)
        self.area_deadband = float(self.get_parameter("area_deadband").value)

        self.kp_yaw   = float(self.get_parameter("kp_yaw").value)
        self.kp_pitch = float(self.get_parameter("kp_pitch").value)
        self.kp_alt   = float(self.get_parameter("kp_alt").value)

        self.max_yaw  = float(self.get_parameter("max_yaw_rate").value)
        self.max_pitch= float(self.get_parameter("max_pitch_deg").value)
        self.max_vz   = float(self.get_parameter("max_vertical_speed").value)

        self.center_deadband_px = int(self.get_parameter("center_deadband_px").value)
        self.min_blob_area_px   = int(self.get_parameter("min_blob_area_px").value)

        self.search_yaw_rate = float(self.get_parameter("search_yaw_rate").value)
        self.lost_timeout    = float(self.get_parameter("lost_timeout").value)

        # ROS I/O
        self.bridge = CvBridge()
        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, qos_cmd)
        self.sub_img = self.create_subscription(Image, self.image_topic, self.image_cb, qos_image)

        # 내부 상태
        self._last_seen = 0.0
        self._last_cmd_time = 0.0
        self._last_cmd = Twist()

        # 20 Hz 보강 퍼블리시
        self._timer = self.create_timer(1.0/20.0, self._tick)

        self.get_logger().info(f"creative_behavior_follow_cf (BLUE) ready. HSV low={self.hsv_low.tolist()}, high={self.hsv_high.tolist()}")

    @staticmethod
    def _saturate(x, lo, hi):
        return max(lo, min(hi, x))

    def image_cb(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # --- HSV & Mask ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 조명 변화에 대비: 값 성분(V)의 상한 클리핑으로 과노출 bloom 완화
        hsv[...,2] = np.minimum(hsv[...,2], 250)

        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # 노이즈 정리(파란 점광원 특성상 작은 번짐/노이즈가 많음)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_found = False
        best = None
        best_area = 0

        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.min_blob_area_px:
                continue
            if a > best_area:
                best_area = a
                best = c

        cmd = Twist()

        if best is not None:
            M = cv2.moments(best)
            if M["m00"] > 1e-5:
                tx = int(M["m10"] / M["m00"])
                ty = int(M["m01"] / M["m00"])
                target_found = True

        if target_found:
            self._last_seen = now
            ex = (tx - cx)
            ey = (ty - cy)

            if abs(ex) < self.center_deadband_px:
                ex = 0
            if abs(ey) < self.center_deadband_px:
                ey = 0

            ex_n = float(ex) / max(1.0, w/2)
            ey_n = float(ey) / max(1.0, h/2)

            area_ratio = best_area / float(w*h)

            yaw_rate = - self.kp_yaw * ex_n
            yaw_rate = self._saturate(yaw_rate, -self.max_yaw, self.max_yaw)

            e_area = (self.target_area_ratio - area_ratio)
            if abs(e_area) < self.area_deadband:
                e_area = 0.0
            pitch = self.kp_pitch * e_area
            pitch = self._saturate(pitch, -self.max_pitch, self.max_pitch)

            gaz = - self.kp_alt * ey_n
            gaz = self._saturate(gaz, -self.max_vz, self.max_vz)

            cmd.linear.x = float(pitch)
            cmd.linear.y = 0.0
            cmd.linear.z = float(gaz)
            cmd.angular.z = float(yaw_rate)
        else:
            if (now - self._last_seen) > self.lost_timeout:
                cmd.angular.z = self.search_yaw_rate  # 느린 수색 회전

        self.pub_cmd.publish(cmd)
        self._last_cmd = cmd
        self._last_cmd_time = now

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_cmd_time > 0.2:
            self.pub_cmd.publish(self._last_cmd)


def main():
    rclpy.init()
    node = FollowCrazyflie()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
