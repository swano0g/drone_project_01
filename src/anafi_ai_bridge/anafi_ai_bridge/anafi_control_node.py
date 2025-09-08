#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import threading
import queue

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Vector3
from std_srvs.srv import Trigger

import olympe
from olympe.messages.ardrone3.Piloting import PCMD, TakeOff, Landing, Emergency, NavigateHome
from olympe.messages import mapper
from olympe.messages.skyctrl.CoPiloting import setPilotingSource
from olympe.messages.gimbal import set_target as gimbal_set_target


class AnafiControl(Node):
    """
    Minimal control node using Olympe only.
      - Connects to SkyController(192.168.53.1) or drone(192.168.42.1)
      - Services: /control/takeoff, /control/land, /control/emergency, /control/halt
      - Topics:
          /control/pcmd (geometry_msgs/Twist)
              linear.x = pitch (deg, +forward)
              linear.y = roll  (deg, +right)
              linear.z = gaz   (m/s, +up)
              angular.z = yaw rate (deg/s, +CCW)
          /control/move_by (Float32MultiArray)  -> data = [dx, dy, dz, dyaw(rad)]
          /control/gimbal (Vector3) -> yaw, pitch, roll (deg), relative frame
    """

    def __init__(self):
        super().__init__("anafi_control")

        # ---------- Parameters ----------
        self.declare_parameter("ip", "192.168.53.1")            # SkyController
        self.declare_parameter("use_skyctrl", True)             # True: SkyController, False: direct drone
        self.declare_parameter("model", "ai")                   # "ai"|"4k"|"thermal"|"usa" (여기선 로직 영향 없음)
        self.declare_parameter("max_tilt", 20.0)                # deg
        self.declare_parameter("max_vertical_speed", 2.0)       # m/s
        self.declare_parameter("max_yaw_rate", 180.0)           # deg/s
        self.declare_parameter("pcmd_hold_timeout", 0.5)        # s, 입력 없으면 호버
        self.declare_parameter("pcmd_rate_hz", 20.0)            # 명령 반복 전송 주기

        self.ip = self.get_parameter("ip").get_parameter_value().string_value
        self.use_sky = self.get_parameter("use_skyctrl").get_parameter_value().bool_value
        self.max_tilt = float(self.get_parameter("max_tilt").value)
        self.max_v = float(self.get_parameter("max_vertical_speed").value)
        self.max_yaw = float(self.get_parameter("max_yaw_rate").value)
        self.hold_timeout = float(self.get_parameter("pcmd_hold_timeout").value)
        self.cmd_rate = float(self.get_parameter("pcmd_rate_hz").value)

        # ---------- Connect ----------
        self.get_logger().info(f"Connecting to {'SkyController' if self.use_sky else 'Anafi'} at {self.ip} ...")
        self.drone = olympe.SkyController(self.ip) if self.use_sky else olympe.Drone(self.ip)
        if not self.drone.connect():
            raise RuntimeError("Failed to connect to device")
        self.get_logger().info("Connected.")

        # If SkyController: grab controls for offboard
        if self.use_sky:
            try:
                # buttons: 0=RTH, 1=TO/LAND, 2=back-left, 3=back-right
                # axes: 0=yaw, 1=throttle, 2=roll, 3=pitch, 4=camera, 5=zoom
                self.drone(mapper.grab(buttons=(1<<0 | 0<<1 | 1<<2 | 1<<3),
                                       axes=(1<<0 | 1<<1 | 1<<2 | 1<<3 | 1<<4 | 1<<5))).wait()
                self.drone(setPilotingSource(source="Controller")).wait()
                self.get_logger().info("Offboard control via SkyController enabled")
            except Exception as e:
                self.get_logger().warn(f"SkyController offboard setup failed: {e}")

        # ---------- State for PCMD loop ----------
        self._last_cmd_time = 0.0
        self._pcmd_lock = threading.Lock()
        # internal command storage (deg/deg/mps/degps)
        self._roll = 0.0
        self._pitch = 0.0
        self._gaz = 0.0
        self._yaw_rate = 0.0

        # ---------- Subscribers ----------
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(Twist, "control/pcmd", self._pcmd_cb, qos_cmd)
        self.create_subscription(Float32MultiArray, "control/move_by", self._move_by_cb, qos_cmd)
        self.create_subscription(Vector3, "control/gimbal", self._gimbal_cb, qos_cmd)

        # ---------- Services ----------
        self.create_service(Trigger, "control/takeoff", self._srv_takeoff)
        self.create_service(Trigger, "control/land", self._srv_land)
        self.create_service(Trigger, "control/emergency", self._srv_emergency)
        self.create_service(Trigger, "control/halt", self._srv_halt)

        # ---------- Timers ----------
        self._pcmd_timer = self.create_timer(1.0 / self.cmd_rate, self._pcmd_tick)

        self.get_logger().info("AnafiControl ready.")

    # ===================== Callbacks =====================

    def _pcmd_cb(self, msg: Twist):
        with self._pcmd_lock:
            self._pitch = float(msg.linear.x)     # deg
            self._roll = float(msg.linear.y)      # deg
            self._gaz = float(msg.linear.z)       # m/s
            self._yaw_rate = float(msg.angular.z) # deg/s
            self._last_cmd_time = time.time()

    def _move_by_cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) < 4:
            self.get_logger().warn("move_by expects [dx, dy, dz, dyaw(rad)]")
            return
        dx, dy, dz, dyaw = map(float, data[:4])

        def worker():
            from olympe.messages import move
            try:
                self.get_logger().info(f"extended_move_by dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, dpsi={dyaw:.2f}rad")
                self.drone(move.extended_move_by(
                    d_x=dx, d_y=dy, d_z=dz, d_psi=dyaw,
                    max_horizontal_speed=self._safe(self.max_v*2.0, 0.1, 15.0),  # 간단한 상한
                    max_vertical_speed=self.max_v,
                    max_yaw_rotation_speed=self.max_yaw
                )).wait()
            except Exception as e:
                self.get_logger().error(f"move_by error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _gimbal_cb(self, msg: Vector3):
        try:
            # relative frame, deg 입력
            self.drone(gimbal_set_target(
                gimbal_id=0,
                control_mode='position',
                yaw_frame_of_reference='relative',
                yaw=float(msg.x),
                pitch_frame_of_reference='relative',
                pitch=float(msg.y),
                roll_frame_of_reference='relative',
                roll=float(msg.z)
            )).wait()
        except Exception as e:
            self.get_logger().warn(f"gimbal command failed: {e}")

    # ===================== Services =====================

    def _srv_takeoff(self, request, response):
        try:
            self.get_logger().info("Takeoff")
            self.drone(TakeOff()).wait()
            response.success = True
            response.message = "OK"
        except Exception as e:
            response.success = False
            response.message = f"takeoff failed: {e}"
        return response

    def _srv_land(self, request, response):
        try:
            self.get_logger().info("Land")
            self.drone(Landing()).wait()
            response.success = True
            response.message = "OK"
        except Exception as e:
            response.success = False
            response.message = f"land failed: {e}"
        return response

    def _srv_emergency(self, request, response):
        try:
            self.get_logger().error("EMERGENCY")
            self.drone(Emergency()).wait()
            response.success = True
            response.message = "OK"
        except Exception as e:
            response.success = False
            response.message = f"emergency failed: {e}"
        return response

    def _srv_halt(self, request, response):
        # 즉시 호버/정지: PCMD 0 + NavigateHome stop
        try:
            with self._pcmd_lock:
                self._roll = self._pitch = self._gaz = self._yaw_rate = 0.0
                self._last_cmd_time = time.time()
            self.drone(PCMD(flag=1, roll=0, pitch=0, yaw=0, gaz=0, timestampAndSeqNum=0))
            self.drone(NavigateHome(start=0))
            response.success = True
            response.message = "halt ok"
        except Exception as e:
            response.success = False
            response.message = f"halt failed: {e}"
        return response

    # ===================== Periodic PCMD =====================

    def _pcmd_tick(self):
        # 입력이 오래 안 들어오면 자동 hover(0 cmd)
        with self._pcmd_lock:
            stale = (time.time() - self._last_cmd_time) > self.hold_timeout
            roll_deg = 0.0 if stale else self._roll
            pitch_deg = 0.0 if stale else self._pitch
            gaz_mps = 0.0 if stale else self._gaz
            yaw_degps = 0.0 if stale else self._yaw_rate

        # → % 변환
        def pct(x, max_abs):
            if max_abs <= 0.0:
                return 0
            val = int(max(-100.0, min(100.0, (x / max_abs) * 100.0)))
            return val

        roll_pct  = pct(roll_deg,  self.max_tilt)
        pitch_pct = pct(pitch_deg, self.max_tilt)
        yaw_pct   = pct(-yaw_degps, self.max_yaw)      # 드론 좌표계 보정(부호 반전)
        gaz_pct   = pct(gaz_mps,    self.max_v)

        try:
            self.drone(PCMD(flag=1,
                            roll=roll_pct,
                            pitch=pitch_pct,
                            yaw=yaw_pct,
                            gaz=gaz_pct,
                            timestampAndSeqNum=0))
        except Exception as e:
            self.get_logger().warn(f"PCMD send failed: {e}")

    # ===================== Helpers =====================

    @staticmethod
    def _safe(val, lo, hi):
        return max(lo, min(hi, float(val)))

    def destroy_node(self):
        try:
            self.get_logger().info("Disconnecting control...")
            try:
                self.drone(PCMD(flag=1, roll=0, pitch=0, yaw=0, gaz=0, timestampAndSeqNum=0))
            except Exception:
                pass
            self.drone.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = AnafiControl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
