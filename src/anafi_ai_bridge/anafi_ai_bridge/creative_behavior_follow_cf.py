#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# creative_behavior_follow_blue.py
import math
import time
import rclpy
import numpy as np
import cv2

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from anafi_ros_interfaces.msg import PilotingCommand


class CreativeBehaviorFollowBlue(Node):
    """
    파란 LED(HSV) 추적 → PilotingCommand 퍼블리시 + 디버그 영상 퍼블리시
    - 입력:  /<ns>/camera/image (sensor_msgs/Image, bgr8)
    - 출력1: /<ns>/drone/command (anafi_ros_interfaces/PilotingCommand)
    - 출력2: /follow_cf/debug (주석 영상, bgr8)
    - 출력3: /follow_cf/mask  (마스크, mono8)
    """
    def __init__(self):
        super().__init__('creative_behavior_follow_blue')

        # ---------------- Params ----------------
        self.declare_parameter('ns', 'anafi')
        ns = self.get_parameter('ns').get_parameter_value().string_value

        # 토픽
        self.declare_parameter('image_topic', f'/{ns}/camera/image')
        self.declare_parameter('cmd_topic',   f'/{ns}/drone/command')
        self.declare_parameter('state_topic', f'/{ns}/drone/state')

        # 디버그 토픽
        self.declare_parameter('debug_image_topic', '/follow_cf/debug')
        self.declare_parameter('debug_mask_topic',  '/follow_cf/mask')

        # HSV 파란 LED 범위 (BGR → HSV, 0~179/255/255)
        self.declare_parameter('h_lo',  95)   # 파란색 시작 Hue
        self.declare_parameter('h_hi', 130)   # 파란색 끝 Hue
        self.declare_parameter('s_lo', 120)
        self.declare_parameter('v_lo', 120)
        self.declare_parameter('morph_kernel', 3)

        # 컨트롤 게인/제한
        self.declare_parameter('kp_yaw',   0.30)   # [deg/s] per (norm error)
        self.declare_parameter('kp_pitch', 10.0)   # [deg]    forward tilt per distance error
        self.declare_parameter('vz_hover', 0.0)    # [m/s]    기본 고도 명령(필요시 0 유지)

        self.declare_parameter('pitch_deg_max', 12.0)   # 안전 제한 (±deg)
        self.declare_parameter('yaw_deg_s_max', 45.0)   # 안전 제한 (±deg/s)

        # 거리(=타겟 크기) 제어
        self.declare_parameter('target_px', 80.0)  # 원하는 LED blob 지름(px)
        self.declare_parameter('kp_dist',   0.06)  # pitch에 추가 가중치 (크기 오차→전진)
        self.declare_parameter('min_area',  30)    # 너무 작은 잡음 제거

        # 타겟 소실/탐색
        self.declare_parameter('lost_timeout', 0.8)      # s
        self.declare_parameter('search_yaw_deg_s', 15.0) # 타겟 소실 시 좌우로 훑기

        # 전송 속도
        self.declare_parameter('rate_hz', 20.0)

        # ---------------- Internals --------------
        self.bridge = CvBridge()
        self.last_seen_t = 0.0
        self.state = "UNKNOWN"
        self.frame_w = 640
        self.frame_h = 480

        # 마지막 측정값 (컨트롤용)
        self.last_cx_norm = None
        self.last_diam_px = None

        # QoS
        qos_cam = QoSProfile(
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

        # Subs/Pubs
        self.sub_img = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.on_image,
            qos_cam,
        )
        self.sub_state = self.create_subscription(
            String,
            self.get_parameter('state_topic').value,
            self.on_state,
            qos_cmd,
        )
        self.pub_cmd = self.create_publisher(
            PilotingCommand,
            self.get_parameter('cmd_topic').value,
            qos_cmd,
        )

        # 디버그 퍼블리셔
        self.pub_dbg  = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            qos_cam,
        )
        self.pub_mask = self.create_publisher(
            Image,
            self.get_parameter('debug_mask_topic').value,
            qos_cam,
        )

        # 메인 루프 타이머
        self.timer = self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self.control_loop)

        self.get_logger().info(f"[creative] image: {self.get_parameter('image_topic').value}")
        self.get_logger().info(f"[creative] cmd  : {self.get_parameter('cmd_topic').value}")
        self.get_logger().info(f"[creative] dbg  : {self.get_parameter('debug_image_topic').value}")
        self.get_logger().info(f"[creative] mask : {self.get_parameter('debug_mask_topic').value}")
        self.get_logger().info("[creative] ready.")

    # ------------ Callbacks ------------
    def on_state(self, msg: String):
        self.state = msg.data

    def on_image(self, msg: Image):
        # BGR 이미지 획득
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_h, self.frame_w = frame.shape[:2]
        cx_img, cy_img = self.frame_w // 2, self.frame_h // 2

        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 파라미터 읽기
        h_lo = int(self.get_parameter('h_lo').value)
        h_hi = int(self.get_parameter('h_hi').value)
        s_lo = int(self.get_parameter('s_lo').value)
        v_lo = int(self.get_parameter('v_lo').value)
        kernel_size = int(self.get_parameter('morph_kernel').value)
        min_area = float(self.get_parameter('min_area').value)

        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, 255, 255], dtype=np.uint8)

        # 마스크 생성
        mask = cv2.inRange(hsv, lower, upper)

        # 모폴로지(노이즈 제거)
        if kernel_size > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # 컨투어 최대값 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 디버그용 복사본
        dbg = frame.copy()

        if not contours:
            self.last_cx_norm = None
            self.last_diam_px = None
            # 디버그 오버레이
            cv2.drawMarker(dbg, (cx_img, cy_img), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
            cv2.putText(dbg, "SEARCHING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # 디버그 퍼블리시
            self._publish_debug(dbg, mask)
            return

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            self.last_cx_norm = None
            self.last_diam_px = None
            # 디버그 오버레이
            cv2.drawMarker(dbg, (cx_img, cy_img), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
            cv2.putText(dbg, "SEARCHING...(small blob)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._publish_debug(dbg, mask)
            return

        # 원형 외접/바운딩박스
        (x, y), radius = cv2.minEnclosingCircle(c)
        x, y, radius = float(x), float(y), float(radius)
        diam_px = 2.0 * radius

        # 중심 정규화: -1(왼) ~ +1(오)
        cx_norm = (x - self.frame_w * 0.5) / (self.frame_w * 0.5)

        # 내부 상태 업데이트
        self.last_cx_norm = cx_norm
        self.last_diam_px = diam_px
        self.last_seen_t = time.time()

        # 디버그 주석 그리기
        x_b, y_b, w_b, h_b = cv2.boundingRect(c)
        cv2.rectangle(dbg, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 255, 0), 2)    # 초록 박스
        cv2.circle(dbg, (int(x), int(y)), 4, (0, 0, 255), -1)                     # 타겟 중심
        cv2.drawMarker(dbg, (cx_img, cy_img), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)   # 화면 중심
        info = f"area={area:.0f} diam={diam_px:.1f}px cx_norm={cx_norm:+.3f}"
        cv2.putText(dbg, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 디버그 퍼블리시
        self._publish_debug(dbg, mask)

    # ------------ Control ------------
    def control_loop(self):
        # 비행중에만 제어(선택사항)
        if self.state not in ("HOVERING", "FLYING", "MOTOR_RAMPING", "TAKINGOFF"):
            return

        cmd = PilotingCommand()
        yaw_rate = 0.0
        pitch_deg = 0.0
        vz = float(self.get_parameter('vz_hover').value)

        now = time.time()
        lost = (now - self.last_seen_t) > float(self.get_parameter('lost_timeout').value)

        yaw_max = float(self.get_parameter('yaw_deg_s_max').value)
        pitch_max = float(self.get_parameter('pitch_deg_max').value)

        if lost or self.last_cx_norm is None or self.last_diam_px is None:
            # 타겟 소실 → 천천히 회전 탐색
            yaw_rate = float(self.get_parameter('search_yaw_deg_s').value)
            pitch_deg = 0.0
        else:
            # 1) 수평 중앙 정렬 → yaw rate
            kp_yaw = float(self.get_parameter('kp_yaw').value)
            yaw_rate = -kp_yaw * self.last_cx_norm  # 오른쪽(+cx_norm)면 왼쪽(-yaw) 회전

            # 2) 거리(=지름) 제어 → pitch
            target_px = float(self.get_parameter('target_px').value)
            kp_dist = float(self.get_parameter('kp_dist').value)
            dist_err = (target_px - self.last_diam_px) / max(target_px, 1.0)  # [+] 목표보다 작으면 전진
            pitch_deg = kp_dist * dist_err * pitch_max  # dist 비례로 전진각 생성

        # 안전 제한
        yaw_rate = float(np.clip(yaw_rate, -yaw_max, yaw_max))
        pitch_deg = float(np.clip(pitch_deg, -pitch_max, pitch_max))

        cmd.roll = 0.0
        cmd.pitch = pitch_deg
        cmd.yaw = yaw_rate
        cmd.gaz = vz

        self.pub_cmd.publish(cmd)

    # ------------ Debug Publish ------------
    def _publish_debug(self, dbg_bgr: np.ndarray, mask_gray: np.ndarray):
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg_bgr, encoding='bgr8'))
            self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask_gray, encoding='mono8'))
        except Exception as e:
            self.get_logger().warn(f"debug publish error: {e}")


def main():
    rclpy.init()
    node = CreativeBehaviorFollowBlue()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
