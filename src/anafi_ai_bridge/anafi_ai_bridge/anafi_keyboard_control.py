#!/usr/bin/env python3
# anafi_keyboard_controller.py
#
# Keyboard teleop for Parrot Anafi (and Anafi AI) via andriyukr/anafi_ros (ROS 2)
# - Publishes anafi_ros_interfaces/PilotingCommand to 'drone/command'
# - Calls services (relative names):
#     'drone/takeoff', 'drone/land', 'drone/rth', 'drone/halt' (std_srvs/Trigger)
#     'skycontroller/offboard' (std_srvs/SetBool)
#
# 기본 네임스페이스: /anafi  (이 파일은 Node(namespace='/anafi')로 구동됨)
#
# 기본 키:
#   이동:   w/s = +pitch(전/후),  a/d = -/+roll(좌/우),  r/f = +/−gaz(상/하),  q/e = +/−yaw(CCW/CW)
#   속도:   z/x = 기본 스케일 ↓/↑,  대문자(WASDQERF) = 터보 증분
#   동작:   t=이륙, l=착륙, h=RTH, k=HALT(강제 정지), o=오프보드 토글,
#           space=즉시 정지(명령 0), **p=즉시 호버링(강제 정지 + 서비스 호출)**, g=부드러운 정지, ?=도움말
#
# 단위/한계:
#   roll/pitch [deg], yaw [deg/s], gaz [m/s]  (anafi.py의 rpyt_callback과 호환)

import sys
import termios
import tty
import select
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_srvs.srv import Trigger, SetBool
from anafi_ros_interfaces.msg import PilotingCommand

HELP_TEXT = r"""
Anafi Keyboard Controller (anafi_ros)

이동
  w/s : +pitch / -pitch  (전/후, deg)
  a/d : -roll  / +roll   (좌/우, deg)
  r/f : +gaz   / -gaz    (상/하,  m/s)
  q/e : +yaw   / -yaw    (좌/우 회전, deg/s)

속도/스케일
  z/x : base 증분 스케일 ↓ / ↑
  대문자 입력(WASDQERF): 터보 증분

동작
  t : 이륙 (drone/takeoff)
  l : 착륙 (drone/land)
  h : 귀환 (drone/rth)
  k : HALT (강제 정지, drone/halt)
  o : 오프보드 토글 (skycontroller/offboard SetBool)
  space : 즉시 정지(명령 0, 서비스 호출 없음)
  p : 즉시 호버링(강제 정지: 명령 0 + HALT 서비스 호출)
  g : 부드러운 정지(램프 다운)
  ? : 도움말

Ctrl+C 로 종료
"""

def _qos_ctrl() -> QoSProfile:
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.RELIABLE
    q.history = HistoryPolicy.KEEP_LAST
    return q

class _Keyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def getch(self) -> Optional[str]:
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            return sys.stdin.read(1)
        return None

    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

class AnafiKeyboard(Node):
    def __init__(self):
        # 기본 네임스페이스를 /anafi 로 고정
        super().__init__('anafi_keyboard_controller', namespace='/anafi')

        # 파라미터
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('accel_limit_degps', 120.0)     # roll/pitch 변화률 제한 [deg/s]
        self.declare_parameter('yaw_accel_limit_dps2', 360.0)  # yaw 변화률 제한 [deg/s^2] → 프레임당 delta 제한
        self.declare_parameter('gaz_accel_limit_mps2', 2.0)    # gaz 변화률 제한 [m/s^2]

        # 증분 스케일 (키 1회 입력당 누적 증분)
        self.declare_parameter('inc_base_roll_deg', 1.0)
        self.declare_parameter('inc_base_pitch_deg', 1.0)
        self.declare_parameter('inc_base_yaw_dps', 5.0)
        self.declare_parameter('inc_base_gaz_mps', 0.1)
        self.declare_parameter('turbo_multiplier', 3.0)

        # 하드 제한(anafi.py와 호환되는 범위)
        self.declare_parameter('limit_roll_deg',  5.0)
        self.declare_parameter('limit_pitch_deg', 5.0)
        self.declare_parameter('limit_yaw_dps',   200.0)
        self.declare_parameter('limit_gaz_mps',   4.0)

        # 시작 시 오프보드 ON 시도
        self.declare_parameter('enable_offboard_on_start', True)

        # 토픽/서비스 (상대 이름 → /anafi 네임스페이스에 걸림)
        self.cmd_topic = 'drone/command'          # PilotingCommand
        self.srv_takeoff = 'drone/takeoff'        # Trigger
        self.srv_land = 'drone/land'              # Trigger
        self.srv_rth = 'drone/rth'                # Trigger
        self.srv_halt = 'drone/halt'              # Trigger
        self.srv_offboard = 'skycontroller/offboard'  # SetBool

        # 퍼블리셔/서비스 클라이언트
        self.pub = self.create_publisher(PilotingCommand, self.cmd_topic, _qos_ctrl())
        self.cli_takeoff = self.create_client(Trigger, self.srv_takeoff)
        self.cli_land = self.create_client(Trigger, self.srv_land)
        self.cli_rth = self.create_client(Trigger, self.srv_rth)
        self.cli_halt = self.create_client(Trigger, self.srv_halt)
        self.cli_offboard = self.create_client(SetBool, self.srv_offboard)

        # 상태값
        self.dt = 1.0 / float(self.get_parameter('publish_rate_hz').value)

        self.acc_lim_degps = float(self.get_parameter('accel_limit_degps').value)
        self.yaw_acc_lim = float(self.get_parameter('yaw_accel_limit_dps2').value) * self.dt
        self.gaz_acc_lim = float(self.get_parameter('gaz_accel_limit_mps2').value) * self.dt

        self.inc_roll = float(self.get_parameter('inc_base_roll_deg').value)
        self.inc_pitch = float(self.get_parameter('inc_base_pitch_deg').value)
        self.inc_yaw = float(self.get_parameter('inc_base_yaw_dps').value)
        self.inc_gaz = float(self.get_parameter('inc_base_gaz_mps').value)
        self.turbo_mul = float(self.get_parameter('turbo_multiplier').value)

        self.lim_roll = float(self.get_parameter('limit_roll_deg').value)
        self.lim_pitch = float(self.get_parameter('limit_pitch_deg').value)
        self.lim_yaw = float(self.get_parameter('limit_yaw_dps').value)
        self.lim_gaz = float(self.get_parameter('limit_gaz_mps').value)

        # 명령 타깃/현재 (ramp 적용)
        self.tgt = PilotingCommand()
        self.cur = PilotingCommand()

        # 모드/상태
        self.gradual_stop = False
        self.offboard_enabled = False

        # 키보드 스레드 + 타이머
        self.kb = _Keyboard()
        self.kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.kb_thread.start()

        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(HELP_TEXT)

        # 시작 시 오프보드 ON
        if bool(self.get_parameter('enable_offboard_on_start').value):
            self._set_offboard(True)

    # 유틸
    @staticmethod
    def _clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def _ramp_to(self, cur, tgt, max_delta):
        if tgt > cur + max_delta:
            return cur + max_delta
        if tgt < cur - max_delta:
            return cur - max_delta
        return tgt

    def _call_trigger(self, client, name: str):
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f"{name} 서비스 대기 시간 초과")
                return
        fut = client.call_async(Trigger.Request())

        def _done(_):
            try:
                resp = fut.result()
                if resp.success:
                    self.get_logger().info(f"{name}: {resp.message}")
                else:
                    self.get_logger().warn(f"{name} 실패: {resp.message}")
            except Exception as e:
                self.get_logger().error(f"{name} 오류: {e}")

        fut.add_done_callback(_done)

    def _set_offboard(self, enable: bool):
        if not self.cli_offboard.service_is_ready():
            self.cli_offboard.wait_for_service(timeout_sec=3.0)
        req = SetBool.Request()
        req.data = enable
        fut = self.cli_offboard.call_async(req)

        def _done(_):
            ok = False
            try:
                resp = fut.result()
                ok = resp.success
            except Exception as e:
                self.get_logger().error(f"offboard 설정 오류: {e}")
            self.offboard_enabled = enable and ok
            state = "ON" if self.offboard_enabled else "OFF"
            self.get_logger().warning(f"Offboard: {state}")

        fut.add_done_callback(_done)

    def _instant_hover(self):
        """즉시 호버링: 명령 0 퍼블리시 + HALT 서비스 호출"""
        # 명령값 0으로 즉시 전송
        self.tgt = PilotingCommand()
        self.cur = PilotingCommand()
        self.pub.publish(self.cur)  # 즉시 한 번 송출
        # 강제 정지 서비스 호출 (anafi.py: PCMD zero, RTH/POI 등 정지)
        self._call_trigger(self.cli_halt, 'halt')

    # 키 입력 처리
    def _keyboard_loop(self):
        try:
            while rclpy.ok():
                ch = self.kb.getch()
                if ch is None:
                    time.sleep(0.005)
                    continue

                turbo = ch.isupper()
                inc_roll = self.inc_roll * (self.turbo_mul if turbo else 1.0)
                inc_pitch = self.inc_pitch * (self.turbo_mul if turbo else 1.0)
                inc_yaw = self.inc_yaw * (self.turbo_mul if turbo else 1.0)
                inc_gaz = self.inc_gaz * (self.turbo_mul if turbo else 1.0)

                # 이동 증분 (누적 타깃)
                if ch in ('w', 'W'):
                    self.tgt.pitch = self._clamp(self.tgt.pitch + inc_pitch, -self.lim_pitch, self.lim_pitch)
                elif ch in ('s', 'S'):
                    self.tgt.pitch = self._clamp(self.tgt.pitch - inc_pitch, -self.lim_pitch, self.lim_pitch)
                elif ch in ('a', 'A'):
                    self.tgt.roll = self._clamp(self.tgt.roll - inc_roll, -self.lim_roll, self.lim_roll)
                elif ch in ('d', 'D'):
                    self.tgt.roll = self._clamp(self.tgt.roll + inc_roll, -self.lim_roll, self.lim_roll)
                elif ch in ('r', 'R'):
                    self.tgt.gaz = self._clamp(self.tgt.gaz + inc_gaz, -self.lim_gaz, self.lim_gaz)
                elif ch in ('f', 'F'):
                    self.tgt.gaz = self._clamp(self.tgt.gaz - inc_gaz, -self.lim_gaz, self.lim_gaz)
                elif ch in ('q', 'Q'):
                    self.tgt.yaw = self._clamp(self.tgt.yaw + inc_yaw, -self.lim_yaw, self.lim_yaw)
                elif ch in ('e', 'E'):
                    self.tgt.yaw = self._clamp(self.tgt.yaw - inc_yaw, -self.lim_yaw, self.lim_yaw)

                # 동작
                elif ch == ' ':
                    # 즉시 정지(명령 0) - 서비스 호출 없음
                    self.tgt = PilotingCommand()
                    self.cur = PilotingCommand()
                    self.gradual_stop = False
                    self.pub.publish(self.cur)
                elif ch == 'p':
                    # 즉시 호버링(강제 정지)
                    self._instant_hover()
                elif ch == 'g':
                    self.gradual_stop = True
                elif ch == 't':
                    self._call_trigger(self.cli_takeoff, 'takeoff')
                elif ch == 'l':
                    self._call_trigger(self.cli_land, 'land')
                elif ch == 'h':
                    self._call_trigger(self.cli_rth, 'rth')
                elif ch == 'k':
                    self._call_trigger(self.cli_halt, 'halt')
                elif ch == 'o':
                    self._set_offboard(!self.offboard_enabled if False else (not self.offboard_enabled))
                elif ch == 'z':
                    self.inc_roll = max(0.1, self.inc_roll * 0.8)
                    self.inc_pitch = max(0.1, self.inc_pitch * 0.8)
                    self.inc_yaw = max(1.0, self.inc_yaw * 0.8)
                    self.inc_gaz = max(0.02, self.inc_gaz * 0.8)
                    self.get_logger().info(f"스케일↓ roll={self.inc_roll:.2f}°, pitch={self.inc_pitch:.2f}°, yaw={self.inc_yaw:.1f}°/s, gaz={self.inc_gaz:.2f} m/s")
                elif ch == 'x':
                    self.inc_roll = min(10.0, self.inc_roll * 1.25)
                    self.inc_pitch = min(10.0, self.inc_pitch * 1.25)
                    self.inc_yaw = min(45.0, self.inc_yaw * 1.25)
                    self.inc_gaz = min(1.0, self.inc_gaz * 1.25)
                    self.get_logger().info(f"스케일↑ roll={self.inc_roll:.2f}°, pitch={self.inc_pitch:.2f}°, yaw={self.inc_yaw:.1f}°/s, gaz={self.inc_gaz:.2f} m/s")
                elif ch == '?':
                    self.get_logger().info(HELP_TEXT)
                # 기타 키 무시
        finally:
            self.kb.restore()

    def _tick(self):
        # 부드러운 정지
        if self.gradual_stop:
            self.tgt.roll  *= 0.9
            self.tgt.pitch *= 0.9
            self.tgt.yaw   *= 0.9
            self.tgt.gaz   *= 0.9
            if (abs(self.tgt.roll) < 1e-2 and
                abs(self.tgt.pitch) < 1e-2 and
                abs(self.tgt.yaw) < 1e-2 and
                abs(self.tgt.gaz) < 1e-3):
                self.tgt = PilotingCommand()
                self.gradual_stop = False

        # 램프(가속 제한)
        max_delta_deg = self.acc_lim_degps * self.dt
        self.cur.roll  = self._ramp_to(self.cur.roll,  self.tgt.roll,  max_delta_deg)
        self.cur.pitch = self._ramp_to(self.cur.pitch, self.tgt.pitch, max_delta_deg)
        self.cur.yaw = self._ramp_to(self.cur.yaw, self.tgt.yaw, self.yaw_acc_lim)
        self.cur.gaz = self._ramp_to(self.cur.gaz, self.tgt.gaz, self.gaz_acc_lim)

        # 퍼블리시
        self.pub.publish(self.cur)

def main():
    rclpy.init()
    node = AnafiKeyboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
