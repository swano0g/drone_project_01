#!/usr/bin/env python3
# follow_apriltag_crazyflie.py
#
# Crazyflie 2.1+에 부착한 AprilTag를 카메라로 추적해 1 m 뒤에서 따라가기
# - 입력:
#     /<ns>/tag_detections (apriltag_msgs/AprilTagDetectionArray)
#     /<ns>/camera/image   (sensor_msgs/Image, bgr8)  [디버그 오버레이용]
# - 출력:
#     /<ns>/drone/command  (anafi_ros_interfaces/PilotingCommand)
#     /<ns>/follow/debug/* (rqt/rqt_image_view용 디버그 토픽)
# - 키보드: f/t/l/r/e/o/p/space/?  (도움말은 HELP_TEXT 참조)
#
import sys, time, math, threading, termios, tty, select
import rclpy
import numpy as np
import cv2

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, SetBool
from anafi_ros_interfaces.msg import PilotingCommand
from cv_bridge import CvBridge

# Foxy+ 대부분에서 apriltag_msgs 사용. 환경에 따라 apriltag_ros.msg일 수도 있음.
from apriltag_msgs.msg import AprilTagDetectionArray

HELP_TEXT = r"""
[follow_tag] 키보드
  f : 추적 시작/정지 토글 (기본: 정지)
  t : 이륙(/<ns>/drone/takeoff)
  l : 착륙(/<ns>/drone/land)
  r : RTH(/<ns>/drone/rth)
  e : 비상(/<ns>/drone/emergency)
  o : 오프보드 토글(/<ns>/skycontroller/offboard)
  p : 즉시 호버링(명령 0 + /<ns>/drone/halt)
  space : 즉시 정지(명령 0 퍼블리시)
  ? : 도움말
"""

class _Keyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
    def getch(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1) if dr else None
    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

class FollowAprilTagCrazyflie(Node):
    """
    AprilTag 포즈(z: 전방 거리, x: 우측, y: 하향)를 이용해
    - yaw  : atan2(x, z) 기반 정렬
    - pitch: (z - desired_distance)로 전/후진
    - (안전) 속도 스케일/가속 제한/탐색 동작 포함
    - (디버그) rqt_plot/rqt_image_view용 토픽 퍼블리시
    """
    def __init__(self):
        super().__init__('follow_apriltag_crazyflie')

        # ---------- Params ----------
        self.declare_parameter('ns', 'anafi')
        ns = self.get_parameter('ns').get_parameter_value().string_value

        # 토픽 이름
        self.declare_parameter('detections_topic', f'/{ns}/tag_detections')
        self.declare_parameter('image_topic',      f'/{ns}/camera/image')  # 디버그 오버레이용
        self.declare_parameter('cmd_topic',        f'/{ns}/drone/command')
        self.declare_parameter('state_topic',      f'/{ns}/drone/state')

        # 추적할 AprilTag ID 목록
        self.declare_parameter('tag_ids', [0])  # 예: [0, 1]

        # 원하는 스탠드오프 거리 [m]
        self.declare_parameter('desired_distance_m', 1.0)

        # 제어 게인/제한 (느리게, 안전하게)
        self.declare_parameter('kp_yaw',   35.0)  # [deg/s] per rad
        self.declare_parameter('kp_dist',   8.0)  # [deg] per m
        self.declare_parameter('vz_hover',  0.0)  # [m/s]

        self.declare_parameter('pitch_deg_max',   6.0)  # ±deg
        self.declare_parameter('yaw_deg_s_max',  22.0)  # ±deg/s

        # 스무딩/램프(가속 제한) 및 전체 스케일
        self.declare_parameter('rate_hz',  20.0)
        self.declare_parameter('accel_pitch_degps', 50.0)
        self.declare_parameter('accel_yaw_dps2',   100.0)
        self.declare_parameter('speed_scale',       0.5)  # 전체 속도 스케일

        # 탐색 동작 (타겟 소실 시)
        self.declare_parameter('lost_timeout',       0.6)
        self.declare_parameter('search_yaw_deg_s',  10.0)

        # 시작/오프보드
        self.declare_parameter('auto_start_follow', False)
        self.declare_parameter('auto_offboard_on_follow', True)

        # ---------- Internals ----------
        self.ns_prefix = f'/{ns}'
        self.follow_enabled = bool(self.get_parameter('auto_start_follow').value)
        self.last_seen_t = 0.0
        self.cur_cmd = PilotingCommand()
        self.state = "UNKNOWN"
        self._offboard_state = False

        # 마지막 관측 (카메라 좌표계 기준)
        self.last_x = None  # 오른쪽+
        self.last_y = None  # 아래+
        self.last_z = None  # 전방+  (거리)

        # 디버그용 영상 버퍼
        self.bridge = CvBridge()
        self.last_frame = None
        self.frame_w = 0
        self.frame_h = 0

        # QoS
        qos_det = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subs/Pubs
        self.sub_state = self.create_subscription(
            String, self.get_parameter('state_topic').value, self._on_state, qos_cmd
        )
        self.sub_det   = self.create_subscription(
            AprilTagDetectionArray, self.get_parameter('detections_topic').value, self._on_detections, qos_det
        )
        self.sub_img   = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_image, qos_img
        )
        self.pub_cmd   = self.create_publisher(PilotingCommand, self.get_parameter('cmd_topic').value, qos_cmd)

        # ---- Debug publishers (rqt/rqt_image_view 용) ----
        self.pub_dbg_state   = self.create_publisher(String,  f'{self.ns_prefix}/follow/debug/state',        10)
        self.pub_dbg_dist    = self.create_publisher(Float32, f'{self.ns_prefix}/follow/debug/distance',     10)
        self.pub_dbg_disterr = self.create_publisher(Float32, f'{self.ns_prefix}/follow/debug/dist_error',   10)
        self.pub_dbg_headdeg = self.create_publisher(Float32, f'{self.ns_prefix}/follow/debug/heading_error_deg', 10)
        self.pub_dbg_twist   = self.create_publisher(Twist,   f'{self.ns_prefix}/follow/debug/cmd_twist',    10)
        self.pub_dbg_image   = self.create_publisher(Image,   f'{self.ns_prefix}/follow/debug/image',        10)

        # Services
        self.cli_halt      = self.create_client(Trigger, f'{self.ns_prefix}/drone/halt')
        self.cli_takeoff   = self.create_client(Trigger, f'{self.ns_prefix}/drone/takeoff')
        self.cli_land      = self.create_client(Trigger, f'{self.ns_prefix}/drone/land')
        self.cli_emergency = self.create_client(Trigger, f'{self.ns_prefix}/drone/emergency')
        self.cli_rth       = self.create_client(Trigger, f'{self.ns_prefix}/drone/rth')
        self.cli_offboard  = self.create_client(SetBool, f'{self.ns_prefix}/skycontroller/offboard')

        # 타이머/키보드
        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)
        self.timer = self.create_timer(self.dt, self._control_loop)
        self.kb = _Keyboard()
        self.kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.kb_thread.start()

        self.get_logger().info(f"[follow_tag] det  : {self.get_parameter('detections_topic').value}")
        self.get_logger().info(f"[follow_tag] image: {self.get_parameter('image_topic').value}")
        self.get_logger().info(f"[follow_tag] cmd  : {self.get_parameter('cmd_topic').value}")
        self.get_logger().info(f"[follow_tag] follow_enabled={self.follow_enabled}")
        self.get_logger().info(HELP_TEXT)

    # ---------- Callbacks ----------
    def _on_state(self, msg: String):
        self.state = msg.data

    def _on_detections(self, msg: AprilTagDetectionArray):
        target_ids = set(self.get_parameter('tag_ids').value)
        best = None
        for det in msg.detections:
            if target_ids & set(det.id):
                best = det
                break

        if best is None:
            self.last_x = self.last_y = self.last_z = None
            return

        # 카메라 좌표: x=우(+), y=하(+), z=전(+)
        px = best.pose.pose.pose.position.x
        py = best.pose.pose.pose.position.y
        pz = best.pose.pose.pose.position.z

        self.last_x, self.last_y, self.last_z = float(px), float(py), float(pz)
        self.last_seen_t = time.time()

    def _on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = frame
            self.frame_h, self.frame_w = frame.shape[:2]
        except Exception as e:
            self.get_logger().warn(f"[follow_tag] 이미지 변환 실패: {e}")

    # ---------- Keyboard ----------
    def _keyboard_loop(self):
        try:
            while rclpy.ok():
                ch = self.kb.getch()
                if ch is None:
                    time.sleep(0.01); continue
                if ch == ' ':
                    self._publish_stop()
                    self._dbg_state("STOP (space)")
                elif ch == 'p':
                    self._instant_hover()
                elif ch == 'o':
                    self._set_offboard(not self._offboard_state)
                elif ch == 't':
                    self._call_trigger(self.cli_takeoff, "Takeoff")
                elif ch == 'l':
                    self._call_trigger(self.cli_land, "Land")
                elif ch == 'e':
                    self._call_trigger(self.cli_emergency, "Emergency")
                elif ch == 'r':
                    self._call_trigger(self.cli_rth, "RTH")
                elif ch == 'f':
                    self.follow_enabled = not self.follow_enabled
                    self._dbg_state(f"Follow: {'ON' if self.follow_enabled else 'OFF'}")
                    if self.follow_enabled and bool(self.get_parameter('auto_offboard_on_follow').value):
                        self._set_offboard(True)
                    if not self.follow_enabled:
                        self._publish_stop()
                elif ch == '?':
                    self.get_logger().info(HELP_TEXT)
        finally:
            self.kb.restore()

    def _publish_stop(self):
        self.cur_cmd = PilotingCommand()
        self.pub_cmd.publish(self.cur_cmd)
        self._publish_cmd_debug(self.cur_cmd)

    def _instant_hover(self):
        self._publish_stop()
        if not self.cli_halt.service_is_ready():
            self.cli_halt.wait_for_service(timeout_sec=2.0)
        fut = self.cli_halt.call_async(Trigger.Request())
        fut.add_done_callback(lambda _: self._log_srv("HALT", fut))

    def _set_offboard(self, state: bool):
        if not self.cli_offboard.service_is_ready():
            self.cli_offboard.wait_for_service(timeout_sec=2.0)
        req = SetBool.Request(); req.data = state
        fut = self.cli_offboard.call_async(req)
        def _done(_):
            try:
                resp = fut.result()
                if resp.success:
                    self._offboard_state = state
                    self._dbg_state(f"Offboard: {'ON' if self._offboard_state else 'OFF'}")
                else:
                    self.get_logger().warning(f"[follow_tag] Offboard 실패: {resp.message}")
            except Exception as e:
                self.get_logger().error(f"[follow_tag] Offboard 오류: {e}")
        fut.add_done_callback(_done)

    def _call_trigger(self, client, name):
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=2.0)
        fut = client.call_async(Trigger.Request())
        fut.add_done_callback(lambda _: self._log_srv(name, fut))

    def _log_srv(self, name, fut):
        try:
            resp = fut.result()
            if resp.success:
                self._dbg_state(f"{name} OK")
            else:
                self.get_logger().warning(f"[follow_tag] {name} 실패: {resp.message}")
        except Exception as e:
            self.get_logger().error(f"[follow_tag] {name} 오류: {e}")

    # ---------- Control ----------
    def _control_loop(self):
        # 상태 문자열 디버그
        dbg_state = f"state={self.state}, follow={'ON' if self.follow_enabled else 'OFF'}"

        # 디텍션/영상만으로도 rqt_plot/rqt_image_view 확인 가능
        # (follow_enabled가 False이면 명령만 0 유지)
        now = time.time()
        lost = (now - self.last_seen_t) > float(self.get_parameter('lost_timeout').value)

        yaw_max   = float(self.get_parameter('yaw_deg_s_max').value)
        pitch_max = float(self.get_parameter('pitch_deg_max').value)
        kp_yaw    = float(self.get_parameter('kp_yaw').value)
        kp_dist   = float(self.get_parameter('kp_dist').value)
        vz        = float(self.get_parameter('vz_hover').value)
        scale     = float(self.get_parameter('speed_scale').value)
        d_des     = float(self.get_parameter('desired_distance_m').value)

        tgt = PilotingCommand()
        heading_err = None  # rad
        dist_err = None     # m

        if lost or self.last_z is None:
            # 타겟 소실 → 느린 탐색 (추종 OFF면 퍼블리시만 0)
            tgt.yaw   = float(np.clip(self.get_parameter('search_yaw_deg_s').value, -yaw_max, yaw_max)) * scale
            tgt.pitch = 0.0
            self._dbg_state(dbg_state + " (lost→scan)")
        else:
            heading_err = math.atan2(self.last_x, self.last_z)  # rad
            yaw_cmd = kp_yaw * heading_err                      # deg/s
            dist_err = (self.last_z - d_des)                    # m
            pitch_cmd = kp_dist * dist_err                      # deg

            tgt.yaw   = float(np.clip(yaw_cmd,   -yaw_max,  yaw_max))  * scale
            tgt.pitch = float(np.clip(pitch_cmd, -pitch_max, pitch_max)) * scale

            # 디버그 수치 퍼블리시
            self.pub_dbg_dist.publish(Float32(data=self.last_z))
            self.pub_dbg_disterr.publish(Float32(data=dist_err))
            self.pub_dbg_headdeg.publish(Float32(data=math.degrees(heading_err)))

        tgt.roll = 0.0
        tgt.gaz  = vz * scale

        # 램프(가속 제한)
        max_dp = float(self.get_parameter('accel_pitch_degps').value) * self.dt
        max_dy = float(self.get_parameter('accel_yaw_dps2').value)    * self.dt
        self.cur_cmd.pitch = self._ramp(self.cur_cmd.pitch, tgt.pitch, max_dp)
        self.cur_cmd.yaw   = self._ramp(self.cur_cmd.yaw,   tgt.yaw,   max_dy)
        self.cur_cmd.roll  = self._ramp(self.cur_cmd.roll,  tgt.roll,  max_dp)
        self.cur_cmd.gaz   = self._ramp(self.cur_cmd.gaz,   tgt.gaz,   0.5 * self.dt)

        # follow_enabled일 때만 실제 명령 퍼블리시
        if self.follow_enabled and self.state in ("HOVERING", "FLYING", "MOTOR_RAMPING", "TAKINGOFF"):
            self.pub_cmd.publish(self.cur_cmd)
        else:
            # 안전: 명령 0 유지
            zero = PilotingCommand()
            self.pub_cmd.publish(zero)

        # 디버그 twist 및 영상 오버레이 퍼블리시
        self._publish_cmd_debug(self.cur_cmd if self.follow_enabled else PilotingCommand())
        self._publish_debug_image(heading_err, dist_err, self.cur_cmd if self.follow_enabled else PilotingCommand(), lost)

    def _publish_cmd_debug(self, cmd: PilotingCommand):
        tw = Twist()
        # 시각화 편의: pitch[deg] → linear.x, yaw[deg/s] → angular.z
        tw.linear.x = float(cmd.pitch)
        tw.linear.y = float(cmd.roll)
        tw.linear.z = float(cmd.gaz)
        tw.angular.z = float(cmd.yaw)
        self.pub_dbg_twist.publish(tw)

    def _publish_debug_image(self, heading_err_rad, dist_err_m, cmd: PilotingCommand, lost: bool):
        if self.last_frame is None:
            return
        frame = self.last_frame.copy()
        h, w = frame.shape[:2]
        cx, cy = int(w*0.5), int(h*0.5)

        # 중앙 기준 십자선
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # 헤딩오차 화살표 (화면 좌표에서 +x 오른쪽)
        if heading_err_rad is not None:
            # 화면상 x방향으로만 단순 표시(직관)
            scale_px = int(w * 0.15)  # 화살 길이
            dx = int(scale_px * math.tan(heading_err_rad))  # 소각도 근사
            end = (int(cx + dx), cy)
            cv2.arrowedLine(frame, (cx, cy), end, (0, 200, 0), 2, tipLength=0.2)

        # 텍스트 정보
        lines = []
        if self.last_z is not None:
            lines.append(f"Z: {self.last_z:.2f} m")
        if dist_err_m is not None:
            lines.append(f"dist_err: {dist_err_m:+.2f} m")
        if heading_err_rad is not None:
            lines.append(f"heading_err: {math.degrees(heading_err_rad):+.1f} deg")
        lines.append(f"cmd pitch: {cmd.pitch:+.2f} deg, yaw: {cmd.yaw:+.2f} deg/s")
        lines.append(f"follow: {'ON' if self.follow_enabled else 'OFF'}  lost: {lost}")

        y0 = 24
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (10, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, t, (10, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),   1, cv2.LINE_AA)

        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub_dbg_image.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"[follow_tag] 디버그 이미지 퍼블리시 실패: {e}")

    def _dbg_state(self, text: str):
        self.pub_dbg_state.publish(String(data=text))

    @staticmethod
    def _ramp(cur, tgt, dmax):
        if tgt > cur + dmax: return cur + dmax
        if tgt < cur - dmax: return cur - dmax
        return tgt

def main():
    rclpy.init()
    node = FollowAprilTagCrazyflie()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
