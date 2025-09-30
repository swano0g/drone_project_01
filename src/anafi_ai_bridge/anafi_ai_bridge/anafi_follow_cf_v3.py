#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger


def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def yaw_from_quat(w, x, y, z):
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)

def quat_from_yaw(yaw: float):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))  # x,y,z,w

def wrap_pi(a):
    while a > math.pi: a -= 2.0*math.pi
    while a < -math.pi: a += 2.0*math.pi
    return a


class AnafiPoseRelativeFollowerWithAvoid(Node):
    """
    초기 상대 위치/방향을 유지하며 추종 + CF 회피(측면 이동)를 Anafi가 동일 축/크기로 미러링.
    - 기본: 이전 '상대 추종'과 동일 (초기 rel_body, yaw_offset 유지)
    - 회피: CF 바디 프레임 측면 속도 |vy_body|가 임계값↑이면 '회피 시작'
            시작 시점의 (cf_pos, cf_yaw)를 래치하여 '회피 기준 프레임' 생성
            이후 dy_body(측면 변위)를 실시간 계산, Anafi 목표에 동일 dy를 추가
            dy≈0 & |vy_body|↓이면 '회피 종료'
    """

    def __init__(self):
        super().__init__('anafi_pose_relative_follower_with_avoid')

        # ---------- Parameters ----------
        # 토픽
        self.declare_parameter('cf_odom_topic', '/cf/odom')
        self.declare_parameter('cf_rpy_topic',  '/cf/rpy')
        self.declare_parameter('anafi_odom_topic', '/anafi/odom')
        self.declare_parameter('anafi_pose_topic', '/anafi/cmd_pose')

        # yaw 모드: 'keep_offset' | 'align_cf' | 'keep_anafi'
        self.declare_parameter('yaw_mode', 'keep_offset')

        # 안전/제어
        self.declare_parameter('min_alt', 0.3)
        self.declare_parameter('max_alt', 6.0)
        self.declare_parameter('cmd_rate_hz', 30.0)
        self.declare_parameter('target_timeout_s', 0.6)
        self.declare_parameter('pos_smoothing_tau_s', 0.12)

        # 자동 이륙(선택)
        self.declare_parameter('auto_takeoff', False)
        self.declare_parameter('takeoff_srv', '/anafi/takeoff')

        # 회피 미러링 파라미터
        self.declare_parameter('avoid_lat_vel_th', 0.18)     # 진입: |vy_body| ≥ th
        self.declare_parameter('avoid_lat_end_th', 0.08)     # 종료: |dy_body| ≤ th
        self.declare_parameter('avoid_vel_end_th', 0.10)     # 종료: |vy_body| ≤ th
        self.declare_parameter('avoid_min_duration_s', 0.30) # 최소 지속 시간
        self.declare_parameter('avoid_dom_ratio', 1.2)       # |vy_body| ≥ ratio*|vx_body| 시 측면 우세

        # ---------- Load ----------
        g = lambda n: self.get_parameter(n).get_parameter_value()
        self.cf_odom_topic = g('cf_odom_topic').string_value
        self.cf_rpy_topic  = g('cf_rpy_topic').string_value
        self.anafi_odom_topic = g('anafi_odom_topic').string_value
        self.anafi_pose_topic  = g('anafi_pose_topic').string_value

        self.yaw_mode = (g('yaw_mode').string_value or 'keep_offset').lower()

        self.min_alt = float(g('min_alt').double_value)
        self.max_alt = float(g('max_alt').double_value)
        self.cmd_rate = float(g('cmd_rate_hz').double_value)
        self.tgt_timeout = float(g('target_timeout_s').double_value)
        self.tau = float(g('pos_smoothing_tau_s').double_value)

        self.auto_takeoff = bool(g('auto_takeoff').bool_value)
        self.takeoff_srv_name = g('takeoff_srv').string_value

        self.avoid_lat_vel_th = float(g('avoid_lat_vel_th').double_value)
        self.avoid_lat_end_th = float(g('avoid_lat_end_th').double_value)
        self.avoid_vel_end_th = float(g('avoid_vel_end_th').double_value)
        self.avoid_min_duration_s = float(g('avoid_min_duration_s').double_value)
        self.avoid_dom_ratio = float(g('avoid_dom_ratio').double_value)

        # ---------- QoS ----------
        sense_qos = QoSProfile(depth=30)
        sense_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sense_qos.history = HistoryPolicy.KEEP_LAST

        ctrl_qos = QoSProfile(depth=10)
        ctrl_qos.reliability = ReliabilityPolicy.RELIABLE
        ctrl_qos.history = HistoryPolicy.KEEP_LAST

        # ---------- Subs/Pubs ----------
        self.sub_cf_odom = self.create_subscription(Odometry, self.cf_odom_topic, self._on_cf_odom, sense_qos)
        self.sub_cf_rpy  = self.create_subscription(Vector3Stamped, self.cf_rpy_topic, self._on_cf_rpy, sense_qos)
        self.sub_anafi_odom = self.create_subscription(Odometry, self.anafi_odom_topic, self._on_anafi_odom, sense_qos)
        self.pub_pose = self.create_publisher(PoseStamped, self.anafi_pose_topic, ctrl_qos)

        # 서비스
        self.cli_takeoff = self.create_client(Trigger, self.takeoff_srv_name)

        # ---------- States ----------
        # CF / Anafi 상태
        self.cf_pos: Optional[Tuple[float,float,float]] = None
        self.cf_vel: Optional[Tuple[float,float,float]] = None
        self.cf_yaw: Optional[float] = None
        self.cf_last_t: Optional[float] = None

        self.anafi_pos = [None, None, None]
        self.anafi_yaw: Optional[float] = None
        self.anafi_last_t: Optional[float] = None

        # 초기 상대(기본 추종)
        self._init_captured = False
        self._rel_body = (0.0, 0.0, 0.0)
        self._yaw_offset = 0.0
        self._anafi_yaw_init: Optional[float] = None

        # 스무딩 상태
        self._t_prev = None
        self._tx, self._ty, self._tz = None, None, None
        self._yaw_target = None

        # 회피 미러링 상태
        self._avoid_active = False
        self._avoid_t0 = 0.0
        self._avoid_cf_entry_pos = (0.0, 0.0, 0.0)
        self._avoid_cf_entry_yaw = 0.0
        self._avoid_extra_dy = 0.0  # Anafi 목표에 추가할 측면 오프셋(지도 프레임으로 변환하여 더함)

        # 타이머
        self.timer = self.create_timer(1.0/max(self.cmd_rate,1.0), self._step)
        self._tried_auto_takeoff = False

        self.get_logger().info("Started relative follower with avoid-mirroring")

    # ---------- Callbacks ----------
    def _on_cf_odom(self, msg: Odometry):
        self.cf_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        self.cf_vel = (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z)
        self.cf_last_t = self._now()

    def _on_cf_rpy(self, msg: Vector3Stamped):
        self.cf_yaw = math.radians(float(msg.vector.z))
        self.cf_last_t = self._now()

    def _on_anafi_odom(self, msg: Odometry):
        self.anafi_pos = [msg.pose.pose.position.x,
                          msg.pose.pose.position.y,
                          msg.pose.pose.position.z]
        q = msg.pose.pose.orientation
        self.anafi_yaw = yaw_from_quat(q.w, q.x, q.y, q.z)
        self.anafi_last_t = self._now()

    # ---------- Helpers ----------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ready(self) -> bool:
        now = self._now()
        if (self.cf_last_t is None) or (now - self.cf_last_t > self.tgt_timeout):
            return False
        if any(v is None for v in self.anafi_pos) or self.anafi_yaw is None:
            return False
        if self.cf_pos is None or self.cf_yaw is None:
            return False
        return True

    def _auto_takeoff(self):
        if not self.auto_takeoff or self._tried_auto_takeoff:
            return
        if self.cli_takeoff.service_is_ready():
            self.get_logger().info('Calling /anafi/takeoff ...')
            self.cli_takeoff.call_async(Trigger.Request())
            self._tried_auto_takeoff = True

    def _capture_initial(self):
        ax, ay, az = self.anafi_pos
        cx, cy, cz = self.cf_pos
        cf_yaw0 = float(self.cf_yaw)
        rel_map = (ax - cx, ay - cy, az - cz)
        c, s = math.cos(-cf_yaw0), math.sin(-cf_yaw0)
        rel_body_x =  c * rel_map[0] + s * rel_map[1]
        rel_body_y = -s * rel_map[0] + c * rel_map[1]
        rel_body_z =  rel_map[2]
        self._rel_body = (rel_body_x, rel_body_y, rel_body_z)
        self._yaw_offset = wrap_pi(float(self.anafi_yaw) - cf_yaw0)
        self._anafi_yaw_init = float(self.anafi_yaw)
        self._init_captured = True
        self.get_logger().info(f"[INIT] rel_body=({rel_body_x:.3f},{rel_body_y:.3f},{rel_body_z:.3f}), yaw_off={self._yaw_offset:.3f} rad")

    def _cf_body_vel(self) -> Tuple[float,float]:
        """CF 속도를 CF 바디 프레임으로 변환한 (vx_body, vy_body) 반환."""
        if self.cf_vel is None or self.cf_yaw is None:
            return (0.0, 0.0)
        c, s = math.cos(self.cf_yaw), math.sin(self.cf_yaw)
        vx, vy = self.cf_vel[0], self.cf_vel[1]
        vx_b =  c * vx + s * vy
        vy_b = -s * vx + c * vy
        return (vx_b, vy_b)

    # ---------- Avoid detect/replicate ----------
    def _avoid_update(self):
        """회피 상태 머신 업데이트 및 _avoid_extra_dy 계산."""
        if self.cf_pos is None or self.cf_yaw is None:
            return

        vx_b, vy_b = self._cf_body_vel()
        now = self._now()

        if not self._avoid_active:
            # 진입 조건: 측면 우세 + 임계 속도 초과
            if abs(vy_b) >= self.avoid_lat_vel_th and abs(vy_b) >= self.avoid_dom_ratio * abs(vx_b):
                self._avoid_active = True
                self._avoid_t0 = now
                self._avoid_cf_entry_pos = self.cf_pos
                self._avoid_cf_entry_yaw = float(self.cf_yaw)
                self._avoid_extra_dy = 0.0
                self.get_logger().info(f"[AVOID] enter: vy_body={vy_b:.2f} m/s")
            return
        else:
            # 현재 CF의 회피 시작 프레임 기준 측면 변위 dy_body 계산
            cx, cy, cz = self.cf_pos
            ex, ey, ez = self._avoid_cf_entry_pos
            yaw0 = self._avoid_cf_entry_yaw
            c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
            # 회피 시작 프레임으로 회전(map→body0)
            dx = cx - ex
            dy = cy - ey
            dx0 =  c0 * dx + s0 * dy
            dy0 = -s0 * dx + c0 * dy   # 측면 변위
            self._avoid_extra_dy = float(dy0)

            # 종료 조건: 최소 지속시간 이후, (변위≈0 & 측면속도 작음)
            if (now - self._avoid_t0) >= self.avoid_min_duration_s:
                if abs(self._avoid_extra_dy) <= self.avoid_lat_end_th and abs(vy_b) <= self.avoid_vel_end_th:
                    self._avoid_active = False
                    self._avoid_extra_dy = 0.0
                    self.get_logger().info("[AVOID] exit")
            return

    # ---------- Control ----------
    def _step(self):
        self._auto_takeoff()

        now = self._now()
        if not self._ready():
            if self.anafi_pos[0] is not None:
                self._publish_pose(self.anafi_pos[0], self.anafi_pos[1],
                                   clamp(self.anafi_pos[2] if self.anafi_pos[2] is not None else self.min_alt,
                                         self.min_alt, self.max_alt),
                                   self.anafi_yaw if self.anafi_yaw is not None else 0.0)
            self._t_prev = now
            return

        if not self._init_captured:
            self._capture_initial()

        # 1) 기본 '상대 추종' 목표 계산
        cx, cy, cz = self.cf_pos
        cf_yaw = float(self.cf_yaw)
        rbx, rby, rbz = self._rel_body
        c, s = math.cos(cf_yaw), math.sin(cf_yaw)
        rel_map_x =  c * rbx - s * rby
        rel_map_y =  s * rbx + c * rby
        rel_map_z =  rbz
        tx_raw = cx + rel_map_x
        ty_raw = cy + rel_map_y
        tz_raw = clamp(cz + rel_map_z, self.min_alt, self.max_alt)

        # 2) 회피 상태 업데이트 & 측면 오프셋 적용
        self._avoid_update()
        if self._avoid_active and abs(self._avoid_extra_dy) > 1e-4:
            # 회피 시작 프레임의 '측면 단위벡터'(지도 프레임) = Rz(yaw0)*[0,1]
            yaw0 = self._avoid_cf_entry_yaw
            lx, ly = -math.sin(yaw0), math.cos(yaw0)
            tx_raw += self._avoid_extra_dy * lx
            ty_raw += self._avoid_extra_dy * ly

        # 3) yaw 목표
        if self.yaw_mode == 'keep_offset':
            yaw_target = wrap_pi(cf_yaw + self._yaw_offset)
        elif self.yaw_mode == 'align_cf':
            yaw_target = cf_yaw
        else:  # keep_anafi
            yaw_target = self._anafi_yaw_init if self._anafi_yaw_init is not None else self.anafi_yaw

        # 4) 스무딩
        if self.tau > 1e-6 and self._t_prev is not None and self._tx is not None:
            dt = max(1e-3, now - self._t_prev)
            alpha = clamp(dt / max(self.tau, 1e-6), 0.0, 1.0)
            tx = (1.0 - alpha) * self._tx + alpha * tx_raw
            ty = (1.0 - alpha) * self._ty + alpha * ty_raw
            tz = (1.0 - alpha) * self._tz + alpha * tz_raw
            dy = wrap_pi(yaw_target - (self._yaw_target if self._yaw_target is not None else yaw_target))
            yaw_s = (self._yaw_target if self._yaw_target is not None else yaw_target) + alpha * dy
        else:
            tx, ty, tz = tx_raw, ty_raw, tz_raw
            yaw_s = yaw_target

        # 5) 발행 & 상태 갱신
        self._publish_pose(tx, ty, tz, yaw_s)
        self._tx, self._ty, self._tz = tx, ty, tz
        self._yaw_target = yaw_s
        self._t_prev = now

    def _publish_pose(self, x: float, y: float, z: float, yaw: float):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # 필요 시 변경
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        qx, qy, qz, qw = quat_from_yaw(float(yaw))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub_pose.publish(msg)


def main():
    rclpy.init()
    node = AnafiPoseRelativeFollowerWithAvoid()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
