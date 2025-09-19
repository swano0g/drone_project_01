#!/usr/bin/env python3
# creative_behavior_follow_blue.py
#
# Crazyflie 2.1+ 파란 LED 추적 + 키보드(이륙/착륙/비상/귀환/호버링/정지/오프보드/토글)
# - Follow ON 시 자동으로 Offboard ON (auto_offboard_on_follow)
# - 디버그 이미지 /<ns>/creative/debug_image 퍼블리시(마스크/검출 시각화)
# - 'd' 로 디버그 로그 토글, 'x' 로 테스트 전진 펄스(감지 없이도 명령 흐름 점검)
#
import math, time, sys, threading, termios, tty, select
import rclpy, numpy as np, cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, SetBool
from anafi_ros_interfaces.msg import PilotingCommand

HELP_TEXT = r"""
[creative] 키보드
  f : 추적 시작/정지 토글 (기본: 정지)
  t : 이륙(/<ns>/drone/takeoff)
  l : 착륙(/<ns>/drone/land)
  r : RTH(/<ns>/drone/rth)
  e : 비상(/<ns>/drone/emergency)
  o : 오프보드 토글(/<ns>/skycontroller/offboard)
  p : 즉시 호버링(명령 0 + /<ns>/drone/halt)
  space : 즉시 정지(명령 0 퍼블리시)
  d : 디버그 로그 토글
  x : 테스트 전진 펄스(감지 무시, pitch 약간 전진 1초)
  ? : 도움말
"""

class _Keyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd)
    def getch(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1) if dr else None
    def restore(self): termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

class CreativeBehaviorFollowBlue(Node):
    def __init__(self):
        super().__init__('creative_behavior_follow_blue')

        # ---------- Params ----------
        self.declare_parameter('ns', 'anafi')
        ns = self.get_parameter('ns').get_parameter_value().string_value

        # 토픽
        self.declare_parameter('image_topic', f'/{ns}/camera/image')
        self.declare_parameter('cmd_topic',   f'/{ns}/drone/command')
        self.declare_parameter('state_topic', f'/{ns}/drone/state')

        # HSV 파랑 + 점광원 튜닝(넓게 시작 가능)
        self.declare_parameter('h_lo',  95)
        self.declare_parameter('h_hi', 130)
        self.declare_parameter('s_lo',  60)
        self.declare_parameter('v_lo', 120)
        self.declare_parameter('use_bgr_blue_dominance', True)
        self.declare_parameter('b_min', 150)
        self.declare_parameter('b_over_g', 25)
        self.declare_parameter('b_over_r', 25)

        self.declare_parameter('morph_kernel', 3)
        self.declare_parameter('gaussian_blur_ksize', 3)

        # 제어(느리게)
        self.declare_parameter('kp_yaw',   0.16)
        self.declare_parameter('kp_dist',  0.045)
        self.declare_parameter('vz_hover', 0.0)
        self.declare_parameter('pitch_deg_max', 6.0)
        self.declare_parameter('yaw_deg_s_max', 22.0)

        # 거리/크기
        self.declare_parameter('target_px', 10.0)
        self.declare_parameter('target_distance_m', 0.0)  # >0이면 지름 기반 자동환산
        self.declare_parameter('target_marker_diameter_m', 0.0)  # 예: 0.004
        self.declare_parameter('camera_hfov_deg', 69.0)

        # 감지 안정화
        self.declare_parameter('min_area', 6)
        self.declare_parameter('min_radius_px', 2.0)
        self.declare_parameter('max_radius_px', 30.0)
        self.declare_parameter('diameter_ema_alpha', 0.4)

        # 탐색/루프/램프/스케일
        self.declare_parameter('lost_timeout', 0.6)
        self.declare_parameter('search_yaw_deg_s', 10.0)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('accel_pitch_degps', 50.0)
        self.declare_parameter('accel_yaw_dps2',    100.0)
        self.declare_parameter('speed_scale', 0.5)

        # 시작/오프보드/디버그
        self.declare_parameter('auto_start_follow', False)
        self.declare_parameter('auto_offboard_on_follow', True)
        self.declare_parameter('publish_debug_image', True)

        # ---------- Internals ----------
        self.bridge = CvBridge()
        self.state = "UNKNOWN"
        self.last_seen_t = 0.0
        self.frame_w, self.frame_h = 640, 480

        self.last_cx_norm = None
        self.last_diam_px = None
        self.ema_diam_px = None
        self.cur_cmd = PilotingCommand()
        self.follow_enabled = bool(self.get_parameter('auto_start_follow').value)
        self.debug_log = True

        # QoS
        qos_cam = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # Subs/Pubs
        self.sub_img   = self.create_subscription(Image,  self.get_parameter('image_topic').value, self.on_image,  qos_cam)
        self.sub_state = self.create_subscription(String, self.get_parameter('state_topic').value, self.on_state,  qos_cmd)
        self.pub_cmd   = self.create_publisher(PilotingCommand, self.get_parameter('cmd_topic').value, qos_cmd)
        self.pub_dbg   = self.create_publisher(Image, f'/{ns}/creative/debug_image', qos_cmd)

        # Services
        self.ns_prefix   = f'/{ns}'
        self.cli_halt      = self.create_client(Trigger, f'{self.ns_prefix}/drone/halt')
        self.cli_takeoff   = self.create_client(Trigger, f'{self.ns_prefix}/drone/takeoff')
        self.cli_land      = self.create_client(Trigger, f'{self.ns_prefix}/drone/land')
        self.cli_emergency = self.create_client(Trigger, f'{self.ns_prefix}/drone/emergency')
        self.cli_rth       = self.create_client(Trigger, f'{self.ns_prefix}/drone/rth')
        self.cli_offboard  = self.create_client(SetBool, f'{self.ns_prefix}/skycontroller/offboard')
        self._offboard_state = False

        # 타이머/키보드
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.kb = _Keyboard()
        self.kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True); self.kb_thread.start()

        self.get_logger().info(f"[creative] image: {self.get_parameter('image_topic').value}")
        self.get_logger().info(f"[creative] cmd  : {self.get_parameter('cmd_topic').value}")
        self.get_logger().info(f"[creative] follow_enabled={self.follow_enabled}")
        self.get_logger().info(HELP_TEXT)

    # -------- Callbacks --------
    def on_state(self, msg: String): self.state = msg.data

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_h, self.frame_w = frame.shape[:2]

        ksize = int(self.get_parameter('gaussian_blur_ksize').value)
        if ksize and ksize > 0 and ksize % 2 == 1:
            frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_lo, h_hi = int(self.get_parameter('h_lo').value), int(self.get_parameter('h_hi').value)
        s_lo, v_lo = int(self.get_parameter('s_lo').value), int(self.get_parameter('v_lo').value)
        mask_hsv = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo], np.uint8), np.array([h_hi, 255, 255], np.uint8))

        mask_bgr = np.zeros_like(mask_hsv)
        if bool(self.get_parameter('use_bgr_blue_dominance').value):
            b_min  = int(self.get_parameter('b_min').value)
            b_over_g = int(self.get_parameter('b_over_g').value)
            b_over_r = int(self.get_parameter('b_over_r').value)
            b, g, r = cv2.split(frame)
            cond = (b >= b_min) & ((b.astype(np.int16) - g.astype(np.int16)) >= b_over_g) & ((b.astype(np.int16) - r.astype(np.int16)) >= b_over_r)
            mask_bgr[cond] = 255

        mask = cv2.bitwise_or(mask_hsv, mask_bgr)

        kernel = int(self.get_parameter('morph_kernel').value)
        if kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 디버그용

        if not contours:
            self._mark_lost(); self._maybe_publish_debug(frame, dbg, None); return

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < float(self.get_parameter('min_area').value):
            self._mark_lost(); self._maybe_publish_debug(frame, dbg, None); return

        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius < float(self.get_parameter('min_radius_px').value) or radius > float(self.get_parameter('max_radius_px').value):
            self._mark_lost(); self._maybe_publish_debug(frame, dbg, None); return

        cx, cy = float(x), float(y)
        diam_px = float(2.0 * radius)

        alpha = float(self.get_parameter('diameter_ema_alpha').value)
        self.ema_diam_px = diam_px if self.ema_diam_px is None else alpha * diam_px + (1.0 - alpha) * self.ema_diam_px

        cx_norm = (cx - self.frame_w * 0.5) / (self.frame_w * 0.5)
        self.last_cx_norm = cx_norm
        self.last_diam_px = self.ema_diam_px
        self.last_seen_t = time.time()

        # 디버그 오버레이
        cv2.circle(dbg, (int(cx), int(cy)), int(radius), (0,255,0), 2)
        self._maybe_publish_debug(frame, dbg, (cx, cy))

    def _maybe_publish_debug(self, frame, dbg_mask_bgr, center):
        if not bool(self.get_parameter('publish_debug_image').value): return
        # 좌: 원본 / 우: 마스크+검출
        h, w = frame.shape[:2]
        vis = np.zeros((h, w*2, 3), dtype=np.uint8)
        vis[:, :w] = frame
        vis[:, w:] = dbg_mask_bgr
        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(vis, (cx, cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        self.pub_dbg.publish(msg)

    def _mark_lost(self):
        self.last_cx_norm = None; self.last_diam_px = None

    # -------- Keyboard --------
    def _keyboard_loop(self):
        kb = _Keyboard()
        try:
            while rclpy.ok():
                ch = kb.getch()
                if ch is None: time.sleep(0.01); continue
                if ch == ' ': self._publish_stop(); self.get_logger().warning("[creative] SPACE: STOP")
                elif ch == 'p': self._instant_hover()
                elif ch == 'o': self._toggle_offboard()
                elif ch == 't': self._call_trigger(self.cli_takeoff, "Takeoff")
                elif ch == 'l': self._call_trigger(self.cli_land, "Land")
                elif ch == 'e': self._call_trigger(self.cli_emergency, "Emergency")
                elif ch == 'r': self._call_trigger(self.cli_rth, "RTH")
                elif ch == 'f':
                    self.follow_enabled = not self.follow_enabled
                    self.get_logger().warning(f"[creative] Follow: {'ON' if self.follow_enabled else 'OFF'}")
                    if self.follow_enabled and bool(self.get_parameter('auto_offboard_on_follow').value):
                        # 자동 Offboard ON
                        self._set_offboard(True)
                    if not self.follow_enabled:
                        self._publish_stop()
                elif ch == 'd':
                    self.debug_log = not self.debug_log
                    self.get_logger().info(f"[creative] Debug log: {'ON' if self.debug_log else 'OFF'}")
                elif ch == 'x':
                    # 테스트 전진 펄스(감지 없이 명령 경로 확인)
                    self.get_logger().warning("[creative] TEST PULSE: pitch +2.0 deg for 1.0s")
                    cmd = PilotingCommand(); cmd.pitch = 2.0
                    t0 = time.time()
                    while time.time() - t0 < 1.0:
                        self.pub_cmd.publish(cmd); time.sleep(0.05)
                    self._publish_stop()
                elif ch == '?': self.get_logger().info(HELP_TEXT)
        finally:
            kb.restore()

    def _publish_stop(self):
        self.cur_cmd = PilotingCommand(); self.pub_cmd.publish(self.cur_cmd)

    def _instant_hover(self):
        self._publish_stop()
        if not self.cli_halt.service_is_ready(): self.cli_halt.wait_for_service(timeout_sec=2.0)
        fut = self.cli_halt.call_async(Trigger.Request()); fut.add_done_callback(lambda _: self._log_srv("HALT", fut))

    def _toggle_offboard(self):
        self._set_offboard(not self._offboard_state)

    def _set_offboard(self, state: bool):
        if not self.cli_offboard.service_is_ready(): self.cli_offboard.wait_for_service(timeout_sec=2.0)
        req = SetBool.Request(); req.data = state
        fut = self.cli_offboard.call_async(req)
        def _done(_):
            try:
                resp = fut.result()
                if resp.success:
                    self._offboard_state = state
                    self.get_logger().warning(f"[creative] Offboard: {'ON' if self._offboard_state else 'OFF'}")
                else:
                    self.get_logger().warn(f"[creative] Offboard 실패: {resp.message}")
            except Exception as e:
                self.get_logger().error(f"[creative] Offboard 오류: {e}")
        fut.add_done_callback(_done)

    def _call_trigger(self, client, name):
        if not client.service_is_ready(): client.wait_for_service(timeout_sec=2.0)
        fut = client.call_async(Trigger.Request()); fut.add_done_callback(lambda _: self._log_srv(name, fut))

    def _log_srv(self, name, fut):
        try:
            resp = fut.result()
            if resp.success: self.get_logger().warning(f"[creative] {name} OK")
            else: self.get_logger().warn(f"[creative] {name} 실패: {resp.message}")
        except Exception as e:
            self.get_logger().error(f"[creative] {name} 오류: {e}")

    # -------- Control --------
    def control_loop(self):
        if not self.follow_enabled: return
        if self.state not in ("HOVERING", "FLYING", "MOTOR_RAMPING", "TAKINGOFF"): return

        now = time.time()
        lost = (now - self.last_seen_t) > float(self.get_parameter('lost_timeout').value)

        yaw_max   = float(self.get_parameter('yaw_deg_s_max').value)
        pitch_max = float(self.get_parameter('pitch_deg_max').value)
        kp_yaw    = float(self.get_parameter('kp_yaw').value)
        kp_dist   = float(self.get_parameter('kp_dist').value)
        vz        = float(self.get_parameter('vz_hover').value)
        scale     = float(self.get_parameter('speed_scale').value)

        # 목표 픽셀 지름(거리 모드 지원)
        target_px = float(self.get_parameter('target_px').value)
        d_m = float(self.get_parameter('target_distance_m').value)
        diam_m = float(self.get_parameter('target_marker_diameter_m').value)
        hfov = float(self.get_parameter('camera_hfov_deg').value)
        if d_m > 0.0 and diam_m > 0.0 and self.frame_w > 0:
            f_px = (self.frame_w * 0.5) / math.tan(math.radians(hfov) * 0.5)
            target_px = (diam_m * f_px) / max(d_m, 1e-6)
            target_px = float(np.clip(target_px, 5.0, self.frame_w * 0.9))

        tgt = PilotingCommand()
        if lost or self.last_cx_norm is None or self.last_diam_px is None:
            tgt.yaw = float(np.clip(self.get_parameter('search_yaw_deg_s').value, -yaw_max, yaw_max)) * scale
            tgt.pitch = 0.0
            if self.debug_log:
                self.get_logger().debug("[creative] LOST: scanning yaw=%.2f" % tgt.yaw)
        else:
            yaw_rate = -kp_yaw * self.last_cx_norm
            dist_err = (target_px - self.last_diam_px) / max(target_px, 1.0)
            pitch = kp_dist * dist_err * pitch_max

            tgt.yaw   = float(np.clip(yaw_rate, -yaw_max, yaw_max)) * scale
            tgt.pitch = float(np.clip(pitch,   -pitch_max, pitch_max)) * scale
            if self.debug_log:
                self.get_logger().debug(f"[creative] cx_norm={self.last_cx_norm:.3f}, diam={self.last_diam_px:.1f}px, tgt_px={target_px:.1f}, pitch={tgt.pitch:.2f}, yaw={tgt.yaw:.2f}")

        tgt.roll = 0.0; tgt.gaz = vz * scale

        # 램프(가속 제한)
        max_dp = float(self.get_parameter('accel_pitch_degps').value) * self.dt
        max_dy = float(self.get_parameter('accel_yaw_dps2').value)    * self.dt
        self.cur_cmd.pitch = self._ramp(self.cur_cmd.pitch, tgt.pitch, max_dp)
        self.cur_cmd.yaw   = self._ramp(self.cur_cmd.yaw,   tgt.yaw,   max_dy)
        self.cur_cmd.roll  = self._ramp(self.cur_cmd.roll,  tgt.roll,  max_dp)
        self.cur_cmd.gaz   = self._ramp(self.cur_cmd.gaz,   tgt.gaz,   0.5 * self.dt)

        self.pub_cmd.publish(self.cur_cmd)

    @staticmethod
    def _ramp(cur, tgt, dmax):
        if tgt > cur + dmax: return cur + dmax
        if tgt < cur - dmax: return cur - dmax
        return tgt

def main():
    rclpy.init()
    node = CreativeBehaviorFollowBlue()
    try: rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
