#!/usr/bin/env python3
import math
import time
from enum import Enum, auto
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Range
from std_srvs.srv import Trigger

# 이미지 처리
try:
    from cv_bridge import CvBridge
    import cv2
except Exception:
    CvBridge = None
    cv2 = None

# YOLO (있으면 사용)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class Phase(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    FORWARD1 = auto()
    DETECT = auto()
    GREET_DOWN = auto()
    GREET_UP = auto()
    GREET_SPIN = auto()   # ← 추가: 인사 후 제자리 회전
    AVOID = auto()
    FORWARD2 = auto()
    AVOID2 = auto()
    FORWARD3 = auto()
    LAND = auto()
    DONE = auto()
    ABORT = auto()


class CreativeBehaviorNode(Node):
    """
    고수준 행동 상태머신 노드.
    - DETECT 구간: 별도 타이머로 hover setpoint를 50Hz로 전송(추론 중 끊김 방지)
    - stop_after 파라미터로 중간 정지 시점 선택 가능(detect/greet/avoid/full)
    - "인사": 목표고도에서 greet_delta_z 만큼 내렸다가(GREET_DOWN) 잠시 대기 → 다시 목표고도로 복귀(GREET_UP)
    - GREET_UP 이후 GREET_SPIN 단계에서 제자리 회전(기본 360°) 수행
    """

    def __init__(self):
        super().__init__("creative_behavior")

        # ---------- QoS helpers ----------
        def _qos_ctrl():
            q = QoSProfile(depth=10)
            q.reliability = ReliabilityPolicy.RELIABLE
            q.history = HistoryPolicy.KEEP_LAST
            return q

        def _qos_sense_best():
            q = QoSProfile(depth=10)
            q.reliability = ReliabilityPolicy.BEST_EFFORT
            q.history = HistoryPolicy.KEEP_LAST
            return q

        # ---------- Emergency Stop ----------
        self._estop = False
        qos_estop = _qos_ctrl()
        self.create_subscription(Bool, '/cf/estop', self._on_estop, qos_estop)

        # ---------- Parameters ----------
        # 토픽
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("odom_topic", "/cf/odom")
        self.declare_parameter("front_range_topic", "/cf/range/front")

        # 시나리오 파라미터
        self.declare_parameter("takeoff_height_m", 0.4)
        self.declare_parameter("takeoff_timeout_s", 8.0)
        self.declare_parameter("forward_speed_mps", 0.3)
        self.declare_parameter("forward1_time_s", 1.0)
        self.declare_parameter("forward2_time_s", 3.0)
        self.declare_parameter("forward3_time_s", 3.0)
        self.declare_parameter("greet_delta_z_m", 0.15)
        self.declare_parameter("greet_pause_s", 0.8)

        # 인사 후 회전 관련 파라미터
        self.declare_parameter("greet_spin_enabled", True)
        self.declare_parameter("greet_spin_degrees", 360.0)  # 총 회전 각도
        self.declare_parameter("greet_spin_deg_s", 90.0)     # 회전 속도 (도/초)

        self.declare_parameter("avoid_lateral_speed_mps", 0.25)
        self.declare_parameter("avoid_forward_speed_mps", 0.15)
        self.declare_parameter("avoid_time_s", 2.0)
        self.declare_parameter("hover_cmd_rate_hz", 30.0)  # 상태머신 tick
        self.declare_parameter("safety_front_min_m", 0.0)

        # 탐지 파라미터
        self.declare_parameter("use_yolo", True)
        self.declare_parameter("yolo_model_path", "yolov8n.pt")
        self.declare_parameter("yolo_conf_th", 0.1)
        self.declare_parameter("hog_stride", 8)
        self.declare_parameter("detect_center_weight", 0.3)
        self.declare_parameter("detect_timeout_s", 12.0)

        # 테스트/정지 지점 선택: detect/greet/avoid/full
        self.declare_parameter("stop_after", "greet")

        # ---------- Load params ----------
        p = lambda n: self.get_parameter(n).get_parameter_value()
        self.image_topic = p("image_topic").string_value
        self.odom_topic = p("odom_topic").string_value
        self.front_range_topic = p("front_range_topic").string_value

        self.alt_target = float(p("takeoff_height_m").double_value or 0.4)
        self.takeoff_timeout_s = float(p("takeoff_timeout_s").double_value or 8.0)
        self.v_forward = float(p("forward_speed_mps").double_value or 0.3)
        self.forward1_time = float(p("forward1_time_s").double_value or 1.0)
        self.forward2_time = float(p("forward2_time_s").double_value or 3.0)
        self.forward3_time = float(p("forward3_time_s").double_value or 3.0)
        self.greet_dz = float(p("greet_delta_z_m").double_value or 0.15)
        self.greet_pause = float(p("greet_pause_s").double_value or 0.8)

        self.greet_spin_enabled = bool(p("greet_spin_enabled").bool_value)
        self.greet_spin_degrees = float(p("greet_spin_degrees").double_value or 360.0)
        self.greet_spin_deg_s = float(p("greet_spin_deg_s").double_value or 90.0)

        self.v_avoid_lat = float(p("avoid_lateral_speed_mps").double_value or 0.25)
        self.v_avoid_fwd = float(p("avoid_forward_speed_mps").double_value or 0.15)
        self.avoid_time = float(p("avoid_time_s").double_value or 2.0)
        self.cmd_rate = float(p("hover_cmd_rate_hz").double_value or 30.0)
        self.safety_front_min = float(p("safety_front_min_m").double_value or 0.00)

        self.use_yolo = bool(p("use_yolo").bool_value or True)
        self.yolo_model_path = p("yolo_model_path").string_value or "yolov8n.pt"
        self.yolo_conf_th = float(p("yolo_conf_th").double_value or 0.6)
        self.hog_stride = int(p("hog_stride").integer_value or 8)
        self.detect_center_weight = float(p("detect_center_weight").double_value or 0.3)
        self.detect_timeout_s = float(p("detect_timeout_s").double_value or 12.0)

        self.stop_after = (p("stop_after").string_value or "greet").lower()
        if self.stop_after not in ("detect", "greet", "avoid", "full"):
            self.get_logger().warn(f"stop_after 파라미터 값이 올바르지 않습니다: {self.stop_after}, 'greet'로 설정합니다.")
            self.stop_after = "greet"

        # ---------- QoS ----------
        qos_ctrl = _qos_ctrl()
        qos_sense = _qos_sense_best()

        # ---------- Publishers (control) ----------
        self.pub_takeoff = self.create_publisher(Float32, "/cf/hl/takeoff", qos_ctrl)
        self.pub_land = self.create_publisher(Float32, "/cf/hl/land", qos_ctrl)
        self.pub_goto = self.create_publisher(PoseStamped, "/cf/hl/goto", qos_ctrl)
        self.pub_hover = self.create_publisher(TwistStamped, "/cf/cmd_hover", qos_ctrl)

        # ---------- Service (safety/stop) ----------
        self.cli_stop = self.create_client(Trigger, "/cf/stop")

        # ---------- Subscribers (sensing) ----------
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos_sense)
        self.sub_front = self.create_subscription(Range, self.front_range_topic, self._on_front, qos_sense)

        # 카메라
        self.bridge = CvBridge() if CvBridge else None
        self.last_img = None
        if self.bridge is not None:
            self.sub_img = self.create_subscription(Image, self.image_topic, self._on_image, qos_sense)
        else:
            self.get_logger().warn("cv_bridge를 불러오지 못했습니다. 카메라 입력 없이 동작합니다.")

        # ---------- Detector init ----------
        self.detector_name = "none"
        self.yolo = None
        self.hog = None
        if self.use_yolo and YOLO is not None:
            try:
                self.yolo = YOLO(self.yolo_model_path)
                self.detector_name = "yolo"
                self.get_logger().info(f"YOLO 로드 성공: {self.yolo_model_path}")
            except Exception as e:
                self.get_logger().warn(f"YOLO 로드 실패: {e}")
        if self.yolo is None and cv2 is not None:
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.detector_name = "hog"
                self.get_logger().info("OpenCV HOG 사람검출로 폴백합니다.")
            except Exception as e:
                self.get_logger().warn(f"HOG 초기화 실패: {e}")

        # ---------- State machine ----------
        self.phase = Phase.TAKEOFF
        self.phase_t0 = self._now()
        self.odom: Optional[Odometry] = None
        self.front_m: Optional[float] = None
        self.alt_current = 0.0
        self.avoid_dir = -1

        # 착륙 원인 추적
        self._land_issued_by_me = False

        # DETECT 전용 호버 타이머
        self._detect_hover_timer = None
        self._detect_hover_rate_hz = 50.0

        # GREET_SPIN 지속시간(초) 계산 값 저장용
        self._greet_spin_duration = 0.0

        # 상태머신 tick
        self.create_timer(1.0 / max(1.0, self.cmd_rate), self._tick)

        self.get_logger().info("Creative behavior node 시작")
        self.get_logger().info(
            f"[PARAM] takeoff={self.alt_target:.2f}m, v_fwd={self.v_forward:.2f}m/s, "
            f"fwd1={self.forward1_time:.2f}s, fwd2={self.forward2_time:.2f}s, fwd3={self.forward3_time:.2f}s, "
            f"avoid(vx={self.v_avoid_fwd:.2f}, vy={self.v_avoid_lat:.2f}, t={self.avoid_time:.2f}s), "
            f"detect_to={self.detect_timeout_s:.2f}s, safety_front_min={self.safety_front_min:.2f}m, "
            f"stop_after={self.stop_after}, greet_spin_enabled={self.greet_spin_enabled}, "
            f"greet_spin={self.greet_spin_degrees:.1f}deg@{self.greet_spin_deg_s:.1f}deg/s"
        )

        # 외부 land 감시(우리 노드가 보낸 착륙과 구분)
        self.create_subscription(Float32, "/cf/hl/land", self._on_land_cmd, qos_ctrl)

    # ---- helpers ----
    def _hold_z(self) -> float:
        if self.odom is not None:
            return float(self.odom.pose.pose.position.z)
        return float(self.alt_target)

    # ---------- Emergency Stop ----------
    def _on_estop(self, msg: Bool):
        if msg.data and not self._estop:
            self._estop = True
            self.get_logger().warn('E-STOP latched → 동작 중지')
            self._stop_detect_hover()
            self.phase = Phase.ABORT

    # ---------- 외부 land 감시 ----------
    def _on_land_cmd(self, msg: Float32):
        if not getattr(self, "_land_issued_by_me", False):
            self.get_logger().warn(f"[LAND] 외부 land 수신(z={float(msg.data):.2f}). 우리 노드가 보낸 착륙 아님.")
        self._land_issued_by_me = False

    # ------------- Utils -------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry):
        self.odom = msg
        self.alt_current = float(msg.pose.pose.position.z)

    def _on_front(self, msg: Range):
        self.front_m = float(msg.range)

    def _on_image(self, msg: Image):
        if self.bridge is None:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_img = cv_img
        except Exception as e:
            self.get_logger().warn(f"이미지 변환 실패: {e}")

    def _publish_hover(self, vx: float, vy: float, z: float, yaw_rate_rad_s: float = 0.0):
        if self._estop:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(z)  # 절대 고도
        msg.twist.angular.z = float(yaw_rate_rad_s)
        self.pub_hover.publish(msg)

    def _publish_takeoff(self, z: float):
        if self._estop:
            return
        self.pub_takeoff.publish(Float32(data=float(z)))

    def _publish_land(self, z: float = 0.0, reason: str = ""):
        if self._estop:
            return
        self._land_issued_by_me = True
        if reason:
            self.get_logger().warn(f"[LAND] reason={reason}")
        self.pub_land.publish(Float32(data=float(z)))

    def _publish_goto_z(self, z: float):
        if self._estop or self.odom is None:
            return
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "map"
        ps.pose.position.x = float(self.odom.pose.pose.position.x)
        ps.pose.position.y = float(self.odom.pose.pose.position.y)
        ps.pose.position.z = float(z)
        ps.pose.orientation.w = 1.0
        self.pub_goto.publish(ps)

    # ---------- DETECT 전용 호버 타이머 ----------
    def _detect_hover_tick(self):
        if self._estop or self.phase != Phase.DETECT:
            return
        self._publish_hover(0.0, 0.0, self._hold_z(), 0.0)

    def _start_detect_hover(self):
        if self._detect_hover_timer is None:
            self._detect_hover_timer = self.create_timer(
                1.0 / max(1.0, self._detect_hover_rate_hz), self._detect_hover_tick
            )
            self.get_logger().info(f"[DETECT] hover hold start @ {self._detect_hover_rate_hz:.1f} Hz")

    def _stop_detect_hover(self):
        if self._detect_hover_timer is not None:
            try:
                self._detect_hover_timer.cancel()
            except Exception:
                pass
        self._detect_hover_timer = None
        self.get_logger().info("[DETECT] hover hold stop")

    # ------------- Detection -------------
    def _detect_person(self) -> Tuple[bool, Optional[float]]:
        if self.last_img is None:
            return False, None
        img = self.last_img

        if self.yolo is not None:
            try:
                res = self.yolo.predict(img, verbose=False, conf=self.yolo_conf_th, imgsz=640)[0]
                found = False
                best_x = None
                W = img.shape[1]
                score_best = -1.0
                for b in res.boxes:
                    cls = int(b.cls[0])
                    if cls != 0:
                        continue
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(float, b.xyxy[0])
                    cx = 0.5 * (x1 + x2)
                    center_off = abs((cx - W/2.0) / (W/2.0))
                    score = (1.0 - self.detect_center_weight) * conf + self.detect_center_weight * (1.0 - center_off)
                    if score > score_best:
                        score_best = score
                        best_x = cx
                        found = True
                if found and best_x is not None and W > 0:
                    x_off_norm = (best_x - W/2.0) / (W/2.0)
                    return True, float(x_off_norm)
                return False, None
            except Exception as e:
                self.get_logger().warn(f"YOLO 추론 실패: {e}")
                return False, None

        if self.hog is not None and cv2 is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects, _ = self.hog.detectMultiScale(gray, winStride=(self.hog_stride, self.hog_stride))
                if len(rects) > 0:
                    areas = [(w*h, (x + w/2.0)) for (x, y, w, h) in rects]
                    _, cx = max(areas, key=lambda t: t[0])
                    W = img.shape[1]
                    if W > 0:
                        x_off_norm = (cx - W/2.0) / (W/2.0)
                        return True, float(x_off_norm)
                return False, None
            except Exception as e:
                self.get_logger().warn(f"HOG 추론 실패: {e}")
                return False, None

        return False, None

    # ------------- Main tick -------------
    def _tick(self):
        if self._estop:
            return

        now = self._now()

        # 안전: 전방 초근접 시 비상 착륙
        if self.front_m is not None and self.front_m < self.safety_front_min and self.phase not in (Phase.LAND, Phase.DONE, Phase.ABORT):
            self.get_logger().warn(f"전방 장애물 감지({self.front_m:.2f} m). 비상 착륙.")
            self._publish_hover(0.0, 0.0, self.alt_current, 0.0)
            self._stop_detect_hover()
            self._publish_land(0.0, reason=f"front_safety front_m={self.front_m:.3f} < safety_min={self.safety_front_min}")
            self.phase = Phase.ABORT
            self.phase_t0 = now
            return

        # 상태별 처리
        if self.phase == Phase.TAKEOFF:
            if (now - self.phase_t0) < 0.2:
                self._publish_takeoff(self.alt_target)
            if self.odom is not None and self.odom.pose.pose.position.z >= (self.alt_target - 0.05):
                self.get_logger().info(f"이륙 완료 → 전진 (alt={self.odom.pose.pose.position.z:.2f}m, target={self.alt_target:.2f}m)")
                self.phase = Phase.FORWARD1
                self.phase_t0 = now
            elif (now - self.phase_t0) > self.takeoff_timeout_s:
                self.get_logger().warn("이륙 타임아웃. 착륙 시도")
                self._stop_detect_hover()
                self._publish_land(0.0, reason=f"takeoff_timeout elapsed={now - self.phase_t0:.2f}s > {self.takeoff_timeout_s}s")
                self.phase = Phase.ABORT
                self.phase_t0 = now

        elif self.phase == Phase.FORWARD1:
            self._publish_hover(self.v_forward, 0.0, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.forward1_time:
                self.get_logger().info(
                    f"전진1 완료 → 사람 탐지 (dt={now - self.phase_t0:.2f}s, plan={self.forward1_time:.2f}s, "
                    f"v={self.v_forward:.2f}m/s ≈ {self.v_forward*self.forward1_time:.2f}m)"
                )
                self.phase = Phase.DETECT
                self.phase_t0 = now
                self._start_detect_hover()

        elif self.phase == Phase.DETECT:
            detected, xoff = self._detect_person()
            if detected:
                if self.stop_after == "detect":
                    self.get_logger().info("사람 인식! (stop_after=detect) → 즉시 착륙")
                    self._stop_detect_hover()
                    self._publish_land(0.0, reason="stop_after=detect")
                    self.phase = Phase.LAND
                    self.phase_t0 = now
                    return

                if xoff is not None:
                    self.avoid_dir = -1 if xoff < 0.0 else 1
                self.get_logger().info(f"사람 인식! x_offset={xoff if xoff is not None else 'NA'} → 인사")
                self._publish_goto_z(max(0.1, self.alt_target - self.greet_dz))
                self._stop_detect_hover()
                self.phase = Phase.GREET_DOWN
                self.phase_t0 = now
            elif (now - self.phase_t0) > self.detect_timeout_s:
                self.get_logger().warn("사람 미탐지(타임아웃) → 착륙")
                self._stop_detect_hover()
                self._publish_land(0.0, reason=f"detect_timeout elapsed={now - self.phase_t0:.2f}s > {self.detect_timeout_s}s")
                self.phase = Phase.LAND
                self.phase_t0 = now
            else:
                # DETECT 중 호버는 별도 타이머가 전송 중
                pass

        elif self.phase == Phase.GREET_DOWN:
            # 내려간 뒤 잠깐 대기 → 다시 목표고도로
            if (now - self.phase_t0) > self.greet_pause:
                self._publish_goto_z(self.alt_target)
                self.phase = Phase.GREET_UP
                self.phase_t0 = now

        elif self.phase == Phase.GREET_UP:
            # 여기서는 hover를 보내지 않음(HL goto로 복귀 중)
            if (now - self.phase_t0) > self.greet_pause:
                # 인사 종료 → 회전 단계 또는 다음 단계
                if self.greet_spin_enabled:
                    self._greet_spin_duration = abs(self.greet_spin_degrees) / max(1e-3, abs(self.greet_spin_deg_s))
                    self.get_logger().info(
                        f"인사 완료 → 제자리 회전 시작 ({self.greet_spin_degrees:.1f}deg @ {self.greet_spin_deg_s:.1f}deg/s ≈ {self._greet_spin_duration:.2f}s)"
                    )
                    self.phase = Phase.GREET_SPIN
                    self.phase_t0 = now
                else:
                    if self.stop_after == "greet":
                        self.get_logger().info("인사 완료 (stop_after=greet) → 착륙")
                        self._publish_land(0.0, reason="stop_after=greet")
                        self.phase = Phase.LAND
                        self.phase_t0 = now
                    else:
                        self.get_logger().info("인사 완료 → 회피1(대각)")
                        self.phase = Phase.AVOID
                        self.phase_t0 = now

        elif self.phase == Phase.GREET_SPIN:
            # 제자리 회전: hover로 yaw_rate만 전달 (vx=vy=0, z=alt_target)
            yaw_rate_rad = math.radians(self.greet_spin_deg_s)
            self._publish_hover(0.0, 0.0, self.alt_target, yaw_rate_rad)
            if (now - self.phase_t0) >= self._greet_spin_duration:
                # 회전 종료 → hover 중지용 한 번 0 yaw 전송(옵션)
                self._publish_hover(0.0, 0.0, self.alt_target, 0.0)
                if self.stop_after == "greet":
                    self.get_logger().info("제자리 회전 완료 (stop_after=greet) → 착륙")
                    self._publish_land(0.0, reason="stop_after=greet_after_spin")
                    self.phase = Phase.LAND
                    self.phase_t0 = now
                else:
                    self.get_logger().info("제자리 회전 완료 → 회피1(대각)")
                    self.phase = Phase.AVOID
                    self.phase_t0 = now

        elif self.phase == Phase.AVOID:
            vx = self.v_avoid_fwd
            vy = self.v_avoid_lat * float(self.avoid_dir)
            self._publish_hover(vx, vy, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.avoid_time:
                if self.stop_after == "avoid":
                    self.get_logger().info("회피1 완료 (stop_after=avoid) → 착륙")
                    self._publish_land(0.0, reason="stop_after=avoid")
                    self.phase = Phase.LAND
                    self.phase_t0 = now
                else:
                    self.get_logger().info("회피1 완료 → 전진2")
                    self.phase = Phase.FORWARD2
                    self.phase_t0 = now

        elif self.phase == Phase.FORWARD2:
            self._publish_hover(self.v_forward, 0.0, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.forward2_time:
                self.get_logger().info("전진2 완료 → 회피2(반대방향 대각)")
                self.phase = Phase.AVOID2
                self.phase_t0 = now

        elif self.phase == Phase.AVOID2:
            vx = self.v_avoid_fwd
            vy = - self.v_avoid_lat * float(self.avoid_dir)
            self._publish_hover(vx, vy, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.avoid_time:
                self.get_logger().info("회피2 완료 → 전진3")
                self.phase = Phase.FORWARD3
                self.phase_t0 = now

        elif self.phase == Phase.FORWARD3:
            self._publish_hover(self.v_forward, 0.0, self.alt_target, 0.0)
            if (now - self.phase_t0) > self.forward3_time:
                self.get_logger().info("전진3 완료 → 착륙")
                self._publish_land(0.0, reason="sequence_end(full)")
                self.phase = Phase.LAND
                self.phase_t0 = now

        elif self.phase == Phase.LAND:
            # 이 상태에서는 hover 명령을 더 이상 보내지 않음(HL land가 우선)
            if self.odom is not None and self.odom.pose.pose.position.z <= 0.05:
                self.get_logger().info("착륙 완료. 동작 종료")
                self.phase = Phase.DONE
                self.phase_t0 = now

        elif self.phase in (Phase.DONE, Phase.ABORT):
            if (now - self.phase_t0) > 2.0:
                if self.cli_stop.service_is_ready():
                    try:
                        self.cli_stop.call_async(Trigger.Request())
                    except Exception:
                        pass
                self.phase_t0 = now


def main():
    rclpy.init()
    node = CreativeBehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
