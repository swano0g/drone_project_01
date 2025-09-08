#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import threading
import queue

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import UInt8, UInt16, Int8, Float32, String, Header, Bool
from geometry_msgs.msg import PointStamped, QuaternionStamped, Vector3Stamped
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time as RosTime

from cv_bridge import CvBridge
import cv2

import yaml
import olympe

# Olympe video formats
VDEF_I420 = olympe.VDEF_I420
VDEF_NV12 = olympe.VDEF_NV12

class AnafiSensing(Node):
    """
    Minimal sensing node using Parrot Olympe only.
      - Connects to SkyController (192.168.53.1) or drone (192.168.42.1)
      - Starts video streaming and publishes:
        /camera/image_raw (Image), /camera/camera_info (CameraInfo)
        /time (builtin_interfaces/Time), /drone/state (String)
        /drone/attitude (QuaternionStamped), /drone/rpy (Vector3Stamped)
        /drone/altitude (Float32), /drone/position (PointStamped)
        /battery/percentage (UInt8), /battery/health (UInt8)
        /link/quality (UInt8), /link/goodput (UInt16, if wifi link exists), /link/rssi (Int8, if wifi)
      - If frame vmeta is missing, it still publishes image.
    """

    def __init__(self):
        super().__init__("anafi_sensing")

        # ---------- Parameters ----------
        self.declare_parameter("ip", "192.168.53.1")         # SkyController IP (direct drone: 192.168.42.1)
        self.declare_parameter("use_skyctrl", True)          # True: SkyController, False: direct drone
        self.declare_parameter("model", "ai")                # "ai", "4k", "thermal", "usa"
        self.declare_parameter("camera_info_yaml", "")       # Optional path to camera_xxx.yaml
        self.declare_parameter("streaming_mode", "high_reliability")  # or "low_latency"
        self.declare_parameter("media_name_ai", "Front camera")       # AI media name
        self.declare_parameter("media_name_default", "DefaultVideo")  # other models
        self.declare_parameter("log_video_gaps", True)

        self.ip = self.get_parameter("ip").get_parameter_value().string_value
        self.use_sky = self.get_parameter("use_skyctrl").get_parameter_value().bool_value
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.streaming_mode = self.get_parameter("streaming_mode").get_parameter_value().string_value
        self.media_name_ai = self.get_parameter("media_name_ai").get_parameter_value().string_value
        self.media_name_default = self.get_parameter("media_name_default").get_parameter_value().string_value
        self.log_video_gaps = self.get_parameter("log_video_gaps").get_parameter_value().bool_value

        # ---------- QoS ----------
        qos_cam = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_fast = QoSProfile(depth=10)

        # ---------- Publishers ----------
        self.pub_image = self.create_publisher(Image, "camera/image_raw", qos_cam)
        self.pub_camera_info = self.create_publisher(CameraInfo, "camera/camera_info", qos_cam)
        self.pub_time = self.create_publisher(RosTime, "time", qos_fast)
        self.pub_state = self.create_publisher(String, "drone/state", qos_fast)
        self.pub_attitude = self.create_publisher(QuaternionStamped, "drone/attitude", qos_fast)
        self.pub_rpy = self.create_publisher(Vector3Stamped, "drone/rpy", qos_fast)
        self.pub_altitude = self.create_publisher(Float32, "drone/altitude", qos_fast)
        self.pub_position = self.create_publisher(PointStamped, "drone/position", qos_fast)
        self.pub_battery_percentage = self.create_publisher(UInt8, "battery/percentage", qos_fast)
        self.pub_battery_health = self.create_publisher(UInt8, "battery/health", qos_fast)
        self.pub_link_quality = self.create_publisher(UInt8, "link/quality", qos_fast)
        self.pub_link_goodput = self.create_publisher(UInt16, "link/goodput", qos_fast)
        self.pub_wifi_rssi = self.create_publisher(Int8, "link/rssi", qos_fast)

        # ---------- CameraInfo ----------
        self.msg_camera_info = CameraInfo()
        self.msg_camera_info.header.frame_id = "camera"
        self._load_camera_info()

        # ---------- Connect ----------
        self.get_logger().info(f"Connecting to {'SkyController' if self.use_sky else 'Anafi'} at {self.ip} ...")
        self.drone = olympe.SkyController(self.ip) if self.use_sky else olympe.Drone(self.ip)
        if not self.drone.connect():
            raise RuntimeError("Failed to connect to device")
        self.get_logger().info("Connected.")

        # Configure streaming mode
        try:
            from olympe.messages import camera
            self.drone(camera.set_streaming_mode(cam_id=0, value=self.streaming_mode)).wait()
        except Exception:
            pass  # older firmwares or AI sim may ignore

        # ---------- Streaming callbacks ----------
        self.frame_queue = queue.Queue(maxsize=1)
        self._last_frame_time = 0.0
        self._bridge = CvBridge()

        self.drone.streaming.set_callbacks(
            raw_cb=self._yuv_frame_cb,
            flush_raw_cb=self._flush_cb,
        )

        media_name = self.media_name_ai if self.model == "ai" else self.media_name_default
        self._streaming_start(media_name)

        # Worker thread for frame processing
        self._worker = threading.Thread(target=self._frame_worker, daemon=True)
        self._worker.start()

        # Watchdog timer: restart streaming if no frames for > 1.0s
        self._wd_timer = self.create_timer(1.0, self._streaming_watchdog)

        self.get_logger().info("AnafiSensing ready.")

    # ---------------- CameraInfo loader ----------------
    def _load_camera_info(self):
        path = self.get_parameter("camera_info_yaml").get_parameter_value().string_value
        if not path:
            # no file provided; publish only size after first frame
            self._caminfo_from_yaml = False
            return
        try:
            with open(os.path.expanduser(path), "r") as f:
                ci = yaml.safe_load(f)
            self.msg_camera_info.width = int(ci.get("image_width", 0))
            self.msg_camera_info.height = int(ci.get("image_height", 0))
            self.msg_camera_info.distortion_model = ci.get("distortion_model", "")
            self.msg_camera_info.k = ci.get("camera_matrix", {}).get("data", [0.0]*9)
            self.msg_camera_info.d = ci.get("distortion_coefficients", {}).get("data", [])
            self.msg_camera_info.r = ci.get("rectification_matrix", {}).get("data", [0.0]*9)
            self.msg_camera_info.p = ci.get("projection_matrix", {}).get("data", [0.0]*12)
            self._caminfo_from_yaml = True
            self.get_logger().info(f"Loaded CameraInfo from {path}")
        except Exception as e:
            self._caminfo_from_yaml = False
            self.get_logger().warn(f"CameraInfo YAML load failed: {e}")

    # ---------------- Streaming helpers ----------------
    def _streaming_start(self, media_name: str):
        try:
            self.drone.streaming.stop()
        except Exception:
            pass
        time.sleep(0.1)
        self.drone.streaming.start(media_name=media_name)
        self._last_frame_time = time.time()
        self.get_logger().info(f"Streaming started: {media_name}")

    def _streaming_watchdog(self):
        now = time.time()
        if self.log_video_gaps and self._last_frame_time > 0 and (now - self._last_frame_time) > 1.0:
            self.get_logger().warn("No frames for >1s: restarting streaming")
            media_name = self.media_name_ai if self.model == "ai" else self.media_name_default
            try:
                self._streaming_start(media_name)
            except Exception as e:
                self.get_logger().error(f"Streaming restart failed: {e}")

    def _yuv_frame_cb(self, yuv_frame):
        yuv_frame.ref()
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait().unref()
            except Exception:
                pass
        self.frame_queue.put_nowait(yuv_frame)

    def _flush_cb(self, stream):
        if stream["vdef_format"] not in (VDEF_I420, VDEF_NV12):
            return True
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait().unref()
            except Exception:
                pass
        return True

    # ---------------- Frame worker ----------------
    def _frame_worker(self):
        while rclpy.ok():
            try:
                yuv = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._last_frame_time = time.time()

            try:
                info = yuv.info()
            except Exception:
                info = {}

            try:
                # Convert YUV to BGR
                fmt = yuv.format()
                if fmt == VDEF_I420:
                    cvflag = cv2.COLOR_YUV2BGR_I420
                elif fmt == VDEF_NV12:
                    cvflag = cv2.COLOR_YUV2BGR_NV12
                else:
                    cvflag = None

                if cvflag is not None:
                    bgr = cv2.cvtColor(yuv.as_ndarray(), cvflag)
                else:
                    # Fallback: skip frame
                    yuv.unref()
                    continue

                # Publish Image
                img_msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
                # timestamp: use info ntp if available else now()
                stamp = self.get_clock().now().to_msg()
                try:
                    ts = info.get("ntp_raw_unskewed_timestamp", None)
                    if ts is not None:
                        # ts is in microseconds
                        sec = int(ts // 1e6)
                        nsec = int((ts % 1e6) * 1e3)
                        stamp.sec = sec
                        stamp.nanosec = nsec
                except Exception:
                    pass
                img_msg.header.stamp = stamp
                img_msg.header.frame_id = "camera"
                self.pub_image.publish(img_msg)

                # Publish CameraInfo (fill width/height if not from YAML)
                if not self._caminfo_from_yaml:
                    h, w = bgr.shape[:2]
                    if self.msg_camera_info.width != w or self.msg_camera_info.height != h:
                        self.msg_camera_info.width = w
                        self.msg_camera_info.height = h
                self.msg_camera_info.header.stamp = stamp
                self.msg_camera_info.header.frame_id = "camera"
                self.pub_camera_info.publish(self.msg_camera_info)
            except Exception as e:
                self.get_logger().warn(f"image publish error: {e}")

            # vmeta-derived telemetry
            try:
                vmeta = yuv.vmeta()
                # vmeta is typically (type, dict)
                if isinstance(vmeta, (list, tuple)) and len(vmeta) > 1 and vmeta[1]:
                    meta = vmeta[1]
                    # time (from raw timescale if present)
                    try:
                        timestamp = info["raw"]["frame"]["timestamp"]
                        timescale = info["raw"]["frame"]["timescale"]
                        tmsg = RosTime()
                        tmsg.sec = int(timestamp // timescale)
                        tmsg.nanosec = int((timestamp % timescale) * (1e9 / timescale))
                        self.pub_time.publish(tmsg)
                    except Exception:
                        pass

                    now_header = Header()
                    now_header.stamp = self.get_clock().now().to_msg()

                    # attitude (drone.quat)
                    try:
                        dq = meta["drone"]["quat"]
                        qmsg = QuaternionStamped()
                        qmsg.header = now_header
                        qmsg.header.frame_id = "world"
                        qmsg.quaternion.x = dq["x"]
                        qmsg.quaternion.y = -dq["y"]
                        qmsg.quaternion.z = -dq["z"]
                        qmsg.quaternion.w = dq["w"]
                        self.pub_attitude.publish(qmsg)

                        # RPY (rad->deg)
                        r, p, y = self._euler_from_quat(qmsg.quaternion)
                        vmsg = Vector3Stamped()
                        vmsg.header = now_header
                        vmsg.header.frame_id = "world"
                        vmsg.vector.x = math.degrees(r)
                        vmsg.vector.y = math.degrees(p)
                        vmsg.vector.z = math.degrees(y)
                        self.pub_rpy.publish(vmsg)
                    except Exception:
                        pass

                    # altitude (ground_distance)
                    try:
                        gd = meta["drone"]["ground_distance"]
                        amsg = Float32()
                        amsg.data = float(gd)
                        self.pub_altitude.publish(amsg)
                    except Exception:
                        pass

                    # position (north/east/down -> ENU-ish)
                    try:
                        pos = meta["drone"]["position"]
                        pmsg = PointStamped()
                        pmsg.header = now_header
                        pmsg.header.frame_id = "world"
                        pmsg.point.x = pos["north"]
                        pmsg.point.y = -pos["east"]
                        pmsg.point.z = -pos["down"]
                        self.pub_position.publish(pmsg)
                    except Exception:
                        pass

                    # battery
                    try:
                        bperc = meta["drone"]["battery_percentage"]
                        bmsg = UInt8()
                        bmsg.data = int(bperc)
                        self.pub_battery_percentage.publish(bmsg)
                    except Exception:
                        pass

                    # state
                    try:
                        state = meta["drone"]["flying_state"]
                        if state.startswith("FS_"):
                            state = state[3:]
                        smsg = String()
                        smsg.data = state
                        self.pub_state.publish(smsg)
                    except Exception:
                        pass

                    # link (wifi or starfish)
                    try:
                        links = meta.get("links", [])
                        if links:
                            link0 = links[0]
                            # wifi fields (4k/thermal/usa)
                            if "wifi" in link0:
                                w = link0["wifi"]
                                if "goodput" in w:
                                    gp = UInt16()
                                    gp.data = int(w["goodput"])
                                    self.pub_link_goodput.publish(gp)
                                if "quality" in w:
                                    ql = UInt8()
                                    ql.data = int(w["quality"])
                                    self.pub_link_quality.publish(ql)
                                if "rssi" in w:
                                    rssi = Int8()
                                    rssi.data = int(w["rssi"])
                                    self.pub_wifi_rssi.publish(rssi)
                            # starfish quality (Ai)
                            if "starfish" in link0 and "quality" in link0["starfish"]:
                                ql = UInt8()
                                ql.data = int(link0["starfish"]["quality"])
                                self.pub_link_quality.publish(ql)
                    except Exception:
                        pass
                else:
                    self.get_logger().warn("Frame metadata empty (publishing image only)")
            except Exception as e:
                self.get_logger().warn(f"vmeta parse error: {e}")

            # release frame
            try:
                yuv.unref()
            except Exception:
                pass

    # ---------------- Helpers ----------------
    @staticmethod
    def _euler_from_quat(q):
        # q = geometry_msgs/Quaternion
        x, y, z, w = q.x, q.y, q.z, q.w
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        # yaw (z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    # ---------------- Shutdown ----------------
    def destroy_node(self):
        try:
            self.get_logger().info("Stopping streaming & disconnecting...")
            try:
                self.drone.streaming.stop()
            except Exception:
                pass
            self.drone.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = AnafiSensing()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
