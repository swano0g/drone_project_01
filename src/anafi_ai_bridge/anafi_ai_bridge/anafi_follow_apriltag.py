#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, time, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from apriltag_msgs.msg import AprilTagDetectionArray
from anafi_ros_interfaces.msg import PilotingCommand

class FollowAprilTag(Node):
    def __init__(self):
        super().__init__("follow_apriltag")
        self.declare_parameter("ns", "anafi")
        ns = self.get_parameter("ns").value
        self.declare_parameter("cmd_topic", f"/{ns}/drone/command")
        self.declare_parameter("state_topic", f"/{ns}/drone/state")
        self.declare_parameter("tag_topic", "/tag_detections")
        self.declare_parameter("target_id", 0)
        self.declare_parameter("z_target", 1.2)     # m: 유지할 거리
        self.declare_parameter("kp_yaw_rate", 120.0) # deg/s per rad
        self.declare_parameter("kp_z", 0.8)          # pitch(deg) per m
        self.declare_parameter("kp_y", 1.2)          # gaz(m/s) per m
        self.declare_parameter("yaw_deg_s_max", 90.0)
        self.declare_parameter("pitch_deg_max", 12.0)
        self.declare_parameter("vz_max", 1.2)
        self.last_seen = 0.0
        self.state = "UNKNOWN"

        qos_cam = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.sub_state = self.create_subscription(String, self.get_parameter("state_topic").value,
                                                  self._on_state, qos_cmd)
        self.sub_tag = self.create_subscription(AprilTagDetectionArray,
                                                self.get_parameter("tag_topic").value,
                                                self._on_tags, qos_cam)
        self.pub_cmd = self.create_publisher(PilotingCommand,
                                             self.get_parameter("cmd_topic").value, qos_cmd)

        self.timer = self.create_timer(1.0/20.0, self._tick)
        self.cmd_cached = PilotingCommand()

    def _on_state(self, msg: String):
        self.state = msg.data

    def _extract_pose(self, det) -> Pose:
        # apriltag_ros 구현별 필드 차이를 안전하게 처리
        try:    # PoseWithCovarianceStamped
            return det.pose.pose.pose
        except Exception:
            try: # PoseStamped
                return det.pose.pose
            except Exception:
                return det.pose

    def _on_tags(self, msg: AprilTagDetectionArray):
        tid = int(self.get_parameter("target_id").value)
        chosen = None
        for d in msg.detections:
            try:
                if hasattr(d, "id") and int(d.id) == tid:
                    chosen = d; break
                # id가 배열인 변형도 있음
                if hasattr(d, "id") and hasattr(d.id, "__iter__") and tid in list(d.id):
                    chosen = d; break
            except Exception:
                pass
        if chosen is None:
            return

        p = self._extract_pose(chosen)
        x = p.position.x  # 카메라 기준 +x: 오른쪽(일반적 관례)
        y = p.position.y  # +y: 아래쪽(또는 위쪽) — 실제 프레임에 따라 부호 조정 필요
        z = p.position.z  # +z: 전방(카메라 앞)

        # 간단한 컨트롤: yaw 오차(좌우), z 오차(앞뒤), y 오차(상하)
        yaw_err_rad = math.atan2(x, z)             # 좌우 각도
        z_err = (float(self.get_parameter("z_target").value) - z)
        y_err = -y  # 위로 양(프레임 부호 다르면 +y로)

        yaw_rate = float(self.get_parameter("kp_yaw_rate").value) * yaw_err_rad * 180.0/math.pi
        pitch_deg = float(self.get_parameter("kp_z").value) * z_err
        gaz = float(self.get_parameter("kp_y").value) * y_err

        # 포화
        yaw_max = float(self.get_parameter("yaw_deg_s_max").value)
        pitch_max = float(self.get_parameter("pitch_deg_max").value)
        vz_max = float(self.get_parameter("vz_max").value)
        yaw_rate = max(-yaw_max, min(yaw_max, yaw_rate))
        pitch_deg = max(-pitch_max, min(pitch_max, pitch_deg))
        gaz = max(-vz_max, min(vz_max, gaz))

        cmd = PilotingCommand()
        cmd.roll = 0.0
        cmd.pitch = pitch_deg
        cmd.yaw = yaw_rate
        cmd.gaz = gaz
        self.cmd_cached = cmd
        self.last_seen = time.time()

    def _tick(self):
        # 비행 상태일 때만 명령 (원하면 조건 제거)
        if self.state not in ("HOVERING","FLYING","MOTOR_RAMPING","TAKINGOFF"):
            return
        # 태그 소실 시 느린 수색
        if time.time() - self.last_seen > 0.6:
            cmd = PilotingCommand()
            cmd.pitch = 0.0
            cmd.yaw = 15.0
            cmd.gaz = 0.0
            self.pub_cmd.publish(cmd)
        else:
            self.pub_cmd.publish(self.cmd_cached)

def main():
    rclpy.init()
    node = FollowAprilTag()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
