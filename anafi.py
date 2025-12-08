#!/usr/bin/env python3
"""
Combined Olympe + ROS2 control entry point for the Anafi drone.

This single file spawns two processes that communicate strictly through a
multiprocessing.Queue (for commands) and a multiprocessing.shared_memory buffer
for video frames.  The design follows the specification provided by the user:

Process A (Olympe core):
    * Creates and owns the olympe.Drone instance
    * Receives Pdraw frames and writes them to shared memory
    * Reads ROS commands from the Queue and applies them to the drone

Process B (ROS2):
    * Spins a ROS2 node that publishes frames read from shared memory
    * Receives std_msgs/String commands and pushes them onto the Queue

Both processes implement graceful shutdown and resource cleanup.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import queue
import signal
import sys
import threading
import time
import uuid
from multiprocessing import shared_memory
from typing import Dict, Tuple

import cv2
import numpy as np
import olympe
from olympe.messages import mapper
from olympe.messages.ardrone3.Piloting import (
    Landing,
    MoveBy,
    NavigateHome,
    PCMD,
    StopPilotedPOI,
    TakeOff,
)
from olympe.messages.common.Mavlink import Stop as MavlinkStop
from olympe.messages.follow_me import stop as FollowMeStop
from olympe.messages.rth import abort as RthAbort
from olympe.messages.rth import return_to_home
from olympe.messages.skyctrl.CoPiloting import setPilotingSource
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool

import rclpy


FRAME_CHANNELS = 3
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_CMD_QUEUE_SIZE = 64
DEFAULT_IMAGE_RATE = 15.0


def _frame_size_from_shape(shape: Tuple[int, int, int]) -> int:
    size = int(np.prod(shape))
    if size <= 0:
        raise ValueError("Shared memory frame size must be positive")
    return size


class DroneController:
    """Owns the Olympe drone and handles streaming + command execution."""

    def __init__(
        self,
        cmd_queue: mp.Queue,
        resp_queue: mp.Queue,
        shm_name: str,
        frame_shape: Tuple[int, int, int],
        drone_ip: str,
        skycontroller_ip: str | None,
    ) -> None:
        self.cmd_queue = cmd_queue
        self.resp_queue = resp_queue
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.frame_shape = frame_shape
        self.frame_array = np.ndarray(frame_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.drone_ip = drone_ip
        self.skycontroller_ip = skycontroller_ip
        self.skycontroller_enabled = bool(self.skycontroller_ip)
        self.drone: olympe.Drone | None = None
        self.pdraw: olympe.Pdraw | None = None
        self._running = True
        self.offboard_enabled = False

    def run(self) -> None:
        logging.info("Olympe process bootstrapping")
        try:
            self._connect()
            self._command_loop()
        except Exception:  # pragma: no cover - defensive log
            logging.exception("Olympe core process crashed")
        finally:
            self.shutdown()

    def _connect(self) -> None:
        if self.skycontroller_enabled:
            logging.info("Connecting to Anafi through SkyController (%s)", self.skycontroller_ip)
            self.drone = olympe.SkyController(self.skycontroller_ip)
        else:
            logging.info("Connecting directly to Anafi at %s", self.drone_ip)
            self.drone = olympe.Drone(self.drone_ip)

        if not self.drone.connect():
            raise RuntimeError(f"Failed to connect to drone at {self.drone_ip}")

        logging.info("Connected to Anafi at %s", self.drone_ip)

        # Configure Pdraw-based streaming callback.
        self.drone.set_streaming_callbacks(raw_cb=self._pdraw_frame_cb)
        self.drone.start_video_streaming()

        # Keep a direct Pdraw reference for explicit lifecycle management.
        self.pdraw = olympe.Pdraw(wait_for_connection=False)
        self.pdraw.set_callbacks(raw_cb=self._pdraw_frame_cb)
        stream_url = f"rtsp://{self.drone_ip}/live"
        if not self.pdraw.play_stream(stream_url):
            logging.warning("Failed to start standalone Pdraw stream at %s", stream_url)
        else:
            logging.info("Pdraw streaming from %s", stream_url)

    def _command_loop(self) -> None:
        assert self.drone is not None
        while self._running:
            try:
                command = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if command is None:
                continue

            if isinstance(command, str):
                command = command.strip()
                if not command:
                    continue
                if command == "__shutdown__":
                    logging.info("Received shutdown signal for drone controller")
                    break
                self._execute_text_command(command)
                continue

            if isinstance(command, dict):
                kind = command.get("type")
                if kind == "service":
                    request_id = command.get("request_id")
                    success, message = self._execute_service_action(
                        command.get("action", ""), command.get("params") or {}
                    )
                    if request_id is not None and self.resp_queue is not None:
                        self.resp_queue.put(
                            {
                                "request_id": request_id,
                                "success": success,
                                "message": message,
                            }
                        )
                elif kind == "text":
                    payload = command.get("command", "")
                    if payload == "__shutdown__":
                        logging.info("Received shutdown signal for drone controller")
                        break
                    self._execute_text_command(payload)
                else:
                    logging.warning("Unknown command payload received: %s", command)
                continue

            if command == "__shutdown__":
                break

    def _execute_text_command(self, command: str) -> None:
        assert self.drone is not None
        lower = command.lower()
        logging.info("Executing drone command: %s", command)
        try:
            if lower == "takeoff":
                self.drone(TakeOff()).wait()
            elif lower == "land":
                self.drone(Landing()).wait()
            elif lower.startswith("moveby"):
                parts = lower.split()
                if len(parts) == 5:
                    dx, dy, dz, dpsi = map(float, parts[1:])
                    self.drone(MoveBy(dx, dy, dz, dpsi)).wait()
                else:
                    logging.warning("Invalid moveBy syntax: %s", command)
            elif lower.startswith("forward"):
                dist = float(lower.split()[1])
                self.drone(MoveBy(dist, 0.0, 0.0, 0.0)).wait()
            elif lower.startswith("backward"):
                dist = float(lower.split()[1])
                self.drone(MoveBy(-dist, 0.0, 0.0, 0.0)).wait()
            elif lower.startswith("left"):
                dist = float(lower.split()[1])
                self.drone(MoveBy(0.0, dist, 0.0, 0.0)).wait()
            elif lower.startswith("right"):
                dist = float(lower.split()[1])
                self.drone(MoveBy(0.0, -dist, 0.0, 0.0)).wait()
            elif lower.startswith("up"):
                dist = float(lower.split()[1])
                self.drone(MoveBy(0.0, 0.0, -dist, 0.0)).wait()
            elif lower.startswith("down"):
                dist = float(lower.split()[1])
                self.drone(MoveBy(0.0, 0.0, dist, 0.0)).wait()
            else:
                logging.warning("Unsupported command received: %s", command)
        except Exception:
            logging.exception("Failed to execute command '%s'", command)

    def _execute_service_action(self, action: str, params: Dict | None) -> Tuple[bool, str]:
        assert self.drone is not None
        params = params or {}
        logging.info("Handling service action '%s' with params %s", action, params)
        try:
            if action == "takeoff":
                self.drone(TakeOff()).wait()
                return True, "Takeoff initiated"
            if action == "land":
                self.drone(Landing()).wait()
                return True, "Landing initiated"
            if action == "rth":
                self.drone(PCMD(flag=1, roll=0, pitch=0, yaw=0, gaz=0, timestampAndSeqNum=0)).wait()
                self.drone(return_to_home()).wait()
                return True, "Return-to-home triggered"
            if action == "halt":
                self.drone(PCMD(flag=1, roll=0, pitch=0, yaw=0, gaz=0, timestampAndSeqNum=0)).wait()
                self.drone(NavigateHome(start=0)).wait()
                self.drone(StopPilotedPOI()).wait()
                self.drone(MavlinkStop()).wait()
                self.drone(FollowMeStop()).wait()
                self.drone(RthAbort()).wait()
                return True, "Halt command applied"
            if action == "offboard":
                enable = bool(params.get("enable"))
                return self._toggle_offboard(enable)
        except Exception as exc:
            logging.exception("Service action '%s' failed", action)
            return False, f"{action} failed: {exc}"

        return False, f"Unsupported action '{action}'"

    def _toggle_offboard(self, enable: bool) -> Tuple[bool, str]:
        assert self.drone is not None
        if not self.skycontroller_enabled:
            return False, "SkyController connection required for offboard control"

        if enable and not self.offboard_enabled:
            try:
                self.drone(
                    mapper.grab(
                        buttons=(1 << 0 | 0 << 1 | 1 << 2 | 1 << 3),
                        axes=(1 << 0 | 1 << 1 | 1 << 2 | 1 << 3 | 1 << 4 | 1 << 5),
                    )
                ).wait()
                self.drone(setPilotingSource(source="Controller")).wait()
                self.offboard_enabled = True
                return True, "Offboard control enabled"
            except Exception as exc:
                logging.exception("Failed enabling offboard")
                return False, f"Failed to enable offboard: {exc}"

        if not enable and self.offboard_enabled:
            try:
                self.drone(mapper.grab(buttons=(0 << 0 | 0 << 1 | 0 << 2 | 1 << 3), axes=0)).wait()
                self.drone(setPilotingSource(source="SkyController")).wait()
                self.offboard_enabled = False
                return True, "Manual control restored"
            except Exception as exc:
                logging.exception("Failed disabling offboard")
                return False, f"Failed to disable offboard: {exc}"

        return True, "No change in offboard state"

    def _pdraw_frame_cb(self, yuv_frame) -> None:
        """Copy frames from the Pdraw callback into shared memory."""
        if yuv_frame is None:
            return

        try:
            yuv_array = yuv_frame.as_ndarray()
            bgr_frame = cv2.cvtColor(yuv_array, cv2.COLOR_YUV2BGR_NV12)
        except Exception:
            logging.debug("Could not convert incoming frame", exc_info=True)
            return

        # Resize if the source does not match the shared memory dimensions.
        if (
            bgr_frame.shape[0] != self.frame_shape[0]
            or bgr_frame.shape[1] != self.frame_shape[1]
        ):
            bgr_frame = cv2.resize(
                bgr_frame,
                (self.frame_shape[1], self.frame_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        np.copyto(self.frame_array, bgr_frame)

    def shutdown(self) -> None:
        self._running = False
        if self.pdraw is not None:
            try:
                self.pdraw.stop()
            except Exception:
                logging.debug("Failed to stop Pdraw cleanly", exc_info=True)
            try:
                self.pdraw.close()
            except Exception:
                logging.debug("Failed to close Pdraw cleanly", exc_info=True)
            self.pdraw = None

        if self.drone is not None:
            try:
                self.drone.stop_video_streaming()
            except Exception:
                logging.debug("stop_video_streaming failed", exc_info=True)
            try:
                self.drone.disconnect()
            except Exception:
                logging.debug("Drone disconnect failed", exc_info=True)
            self.drone = None

        try:
            self.shm.close()
        except FileNotFoundError:
            pass


class ROSBridgeNode(Node):
    """ROS2 node that publishes frames and pushes commands onto the queue."""

    def __init__(
        self,
        cmd_queue: mp.Queue,
        resp_queue: mp.Queue,
        shm_name: str,
        frame_shape: Tuple[int, int, int],
        publish_rate_hz: float,
    ) -> None:
        super().__init__("anafi_ros_bridge")
        self.cmd_queue = cmd_queue
        self.resp_queue = resp_queue
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.frame_shape = frame_shape
        self.frame_array = np.ndarray(frame_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.publish_rate_hz = publish_rate_hz
        self._service_lock = threading.Lock()

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._image_pub = self.create_publisher(Image, "/drone/image", qos)
        self.create_subscription(String, "/drone/cmd", self._command_cb, qos)

        period = 1.0 / max(self.publish_rate_hz, 1e-3)
        self.create_timer(period, self._publish_frame)
        self._create_services()

    def _command_cb(self, msg: String) -> None:
        try:
            self.cmd_queue.put_nowait({"type": "text", "command": msg.data})
        except queue.Full:
            self.get_logger().warn("Command queue is full, dropping command")

    def _publish_frame(self) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = self.frame_shape[0]
        msg.width = self.frame_shape[1]
        msg.encoding = "bgr8"
        msg.step = self.frame_shape[1] * self.frame_shape[2]
        msg.data = self.frame_array.tobytes()
        self._image_pub.publish(msg)

    def _create_services(self) -> None:
        self.create_service(Trigger, "/drone/takeoff", self._make_trigger_handler("takeoff"))
        self.create_service(Trigger, "/drone/land", self._make_trigger_handler("land"))
        self.create_service(Trigger, "/drone/rth", self._make_trigger_handler("rth"))
        self.create_service(Trigger, "/drone/halt", self._make_trigger_handler("halt"))
        self.create_service(SetBool, "/skycontroller/offboard", self._offboard_handler)

    def _make_trigger_handler(self, action: str):
        def _handler(request, response):
            success, message = self._call_service_action(action, {})
            response.success = success
            response.message = message
            return response

        return _handler

    def _offboard_handler(self, request, response):
        success, message = self._call_service_action("offboard", {"enable": request.data})
        response.success = success
        response.message = message
        return response

    def _call_service_action(self, action: str, params: Dict, timeout: float = 10.0) -> Tuple[bool, str]:
        request_id = uuid.uuid4().hex
        payload = {"type": "service", "action": action, "params": params, "request_id": request_id}
        with self._service_lock:
            try:
                self.cmd_queue.put_nowait(payload)
            except queue.Full:
                return False, "Command queue is full"

            deadline = time.monotonic() + timeout
            while rclpy.ok() and time.monotonic() < deadline:
                wait_time = min(0.2, max(0.0, deadline - time.monotonic()))
                try:
                    reply = self.resp_queue.get(timeout=wait_time)
                except queue.Empty:
                    continue

                if reply.get("request_id") == request_id:
                    return bool(reply.get("success")), reply.get("message", "")
                else:
                    self.get_logger().warn("Ignoring response for another request: %s", reply)

            return False, f"Timeout waiting for '{action}' response"

    def destroy_node(self) -> None:
        try:
            self.shm.close()
        except FileNotFoundError:
            pass
        super().destroy_node()


def run_olympe_process(
    cmd_queue: mp.Queue,
    resp_queue: mp.Queue,
    shm_name: str,
    frame_shape: Tuple[int, int, int],
    drone_ip: str,
    skycontroller_ip: str | None,
) -> None:
    controller = DroneController(
        cmd_queue,
        resp_queue,
        shm_name,
        frame_shape,
        drone_ip,
        skycontroller_ip,
    )
    controller.run()


def run_ros_process(
    cmd_queue: mp.Queue,
    resp_queue: mp.Queue,
    shm_name: str,
    frame_shape: Tuple[int, int, int],
    publish_rate_hz: float,
) -> None:
    def _signal_handler(signum, frame):
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    rclpy.init()
    node = ROSBridgeNode(cmd_queue, resp_queue, shm_name, frame_shape, publish_rate_hz)
    try:
        rclpy.spin(node)
    except Exception:  # pragma: no cover - defensive log
        node.get_logger().error("ROS2 process crashed", exc_info=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anafi Olympe + ROS2 bridge")
    parser.add_argument(
        "--drone-ip",
        default="192.168.42.1",
        help="IP address of the Anafi drone (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Width of the shared video frame",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help="Height of the shared video frame",
    )
    parser.add_argument(
        "--image-rate",
        type=float,
        default=DEFAULT_IMAGE_RATE,
        help="ROS image publish frequency in Hz",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_CMD_QUEUE_SIZE,
        help="Maximum number of pending control commands",
    )
    parser.add_argument(
        "--skycontroller-ip",
        default=None,
        help="Optional SkyController IP address for indirect connections",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s (%(processName)s): %(message)s",
    )

    frame_shape = (args.frame_height, args.frame_width, FRAME_CHANNELS)

    mp.set_start_method("spawn", force=True)

    frame_size = _frame_size_from_shape(frame_shape)
    shm = shared_memory.SharedMemory(create=True, size=frame_size)
    frame_view = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
    frame_view[:] = 0

    cmd_queue: mp.Queue = mp.Queue(maxsize=args.queue_size)
    resp_queue: mp.Queue = mp.Queue()

    olympe_process = mp.Process(
        target=run_olympe_process,
        args=(cmd_queue, resp_queue, shm.name, frame_shape, args.drone_ip, args.skycontroller_ip),
        name="OlympeCore",
    )
    ros_process = mp.Process(
        target=run_ros_process,
        args=(cmd_queue, resp_queue, shm.name, frame_shape, args.image_rate),
        name="ROS2Bridge",
    )

    processes = [olympe_process, ros_process]

    for proc in processes:
        proc.start()

    logging.info("Launched Olympe (pid=%s) and ROS2 (pid=%s)", olympe_process.pid, ros_process.pid)

    try:
        while all(proc.is_alive() for proc in processes):
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down")
    finally:
        try:
            cmd_queue.put("__shutdown__", timeout=0.5)
        except Exception:
            pass

        for proc in processes:
            proc.join(timeout=5.0)
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()

        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
