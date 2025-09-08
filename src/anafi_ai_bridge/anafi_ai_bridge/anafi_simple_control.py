#!/usr/bin/env python3
import sys
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_services_default
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import Header
from anafi_ros_interfaces.msg import PilotingCommand, MoveByCommand

class AnafiSimpleControl(Node):
    def __init__(self, ns="anafi"):
        super().__init__("anafi_simple_control")
        self.ns = ns

        # Publishers (drone/command = RPYT, drone/moveby = dx,dy,dz,dyaw)
        self.pub_cmd = self.create_publisher(
            PilotingCommand, f"/{ns}/drone/command", qos_profile_services_default
        )
        self.pub_moveby = self.create_publisher(
            MoveByCommand, f"/{ns}/drone/moveby", qos_profile_services_default
        )

        # Services
        self.cli_takeoff = self.create_client(Trigger, f"/{ns}/drone/takeoff")
        self.cli_land    = self.create_client(Trigger, f"/{ns}/drone/land")
        self.cli_halt    = self.create_client(Trigger, f"/{ns}/drone/halt")
        self.cli_rth     = self.create_client(Trigger, f"/{ns}/drone/rth")
        self.cli_offboard= self.create_client(SetBool, f"/{ns}/skycontroller/offboard")

        for cli in [self.cli_takeoff, self.cli_land, self.cli_halt, self.cli_rth, self.cli_offboard]:
            if not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f"Service not ready: {cli.srv_name}")

        self.get_logger().info("AnafiSimpleControl ready.")

    # ---------- High-level helpers ----------
    def set_offboard(self, on: bool):
        from std_srvs.srv import SetBool
        req = SetBool.Request()
        req.data = on
        fut = self.cli_offboard.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        self.get_logger().info(f"Offboard set -> {on}")

    def takeoff(self):
        req = Trigger.Request()
        fut = self.cli_takeoff.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        self.get_logger().warn("Takeoff requested")

    def land(self):
        req = Trigger.Request()
        fut = self.cli_land.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        self.get_logger().warn("Land requested")

    def halt(self):
        req = Trigger.Request()
        fut = self.cli_halt.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        self.get_logger().warn("HALT requested")

    def rth(self):
        req = Trigger.Request()
        fut = self.cli_rth.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        self.get_logger().warn("RTH requested")

    # ---------- RPYT (roll/pitch/yaw/gaz) ----------
    def send_rpyt(self, roll=0.0, pitch=0.0, yaw=0.0, gaz=0.0, duration=0.0, rate_hz=20.0):
        msg = PilotingCommand()
        msg.roll = float(roll)
        msg.pitch = float(pitch)
        msg.yaw = float(yaw)
        msg.gaz = float(gaz)

        if duration <= 0.0:
            self.pub_cmd.publish(msg)
            return

        period = 1.0 / rate_hz
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < duration:
            self.pub_cmd.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        # stop
        stop = PilotingCommand()
        self.pub_cmd.publish(stop)

    # ---------- moveBy (dx, dy, dz, dyaw in rad) ----------
    def move_by(self, dx=0.0, dy=0.0, dz=0.0, dyaw_rad=0.0):
        msg = MoveByCommand()
        msg.dx = float(dx)
        msg.dy = float(dy)
        msg.dz = float(dz)
        msg.dyaw = float(dyaw_rad)
        self.pub_moveby.publish(msg)
        self.get_logger().info(f"moveBy sent: dx={dx}, dy={dy}, dz={dz}, dyaw(rad)={dyaw_rad}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Minimal ROS2 controller for ANAFI via anafi_ros_nodes")
    parser.add_argument("--ns", default="anafi", help="ROS namespace (default: anafi)")

    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("offboard_on")
    sub.add_parser("offboard_off")
    sub.add_parser("takeoff")
    sub.add_parser("land")
    sub.add_parser("halt")
    sub.add_parser("rth")

    p_rpyt = sub.add_parser("rpyt", help="roll/pitch/yaw/gaz for duration")
    p_rpyt.add_argument("--roll", type=float, default=0.0)
    p_rpyt.add_argument("--pitch", type=float, default=0.0)
    p_rpyt.add_argument("--yaw", type=float, default=0.0, help="yaw rate-like input (mapped in node)")
    p_rpyt.add_argument("--gaz", type=float, default=0.0, help="vertical speed-like input")
    p_rpyt.add_argument("--duration", type=float, default=1.0)
    p_rpyt.add_argument("--rate", type=float, default=20.0)

    p_mb = sub.add_parser("moveby", help="dx dy dz dyaw(rad)")
    p_mb.add_argument("--dx", type=float, default=0.0)
    p_mb.add_argument("--dy", type=float, default=0.0)
    p_mb.add_argument("--dz", type=float, default=0.0)
    p_mb.add_argument("--dyaw", type=float, default=0.0, help="radians (+ccw)")

    args = parser.parse_args()

    rclpy.init()
    node = AnafiSimpleControl(ns=args.ns)

    try:
        if args.cmd == "offboard_on":
            node.set_offboard(True)
        elif args.cmd == "offboard_off":
            node.set_offboard(False)
        elif args.cmd == "takeoff":
            node.takeoff()
        elif args.cmd == "land":
            node.land()
        elif args.cmd == "halt":
            node.halt()
        elif args.cmd == "rth":
            node.rth()
        elif args.cmd == "rpyt":
            node.send_rpyt(args.roll, args.pitch, args.yaw, args.gaz, args.duration, args.rate)
        elif args.cmd == "moveby":
            node.move_by(args.dx, args.dy, args.dz, args.dyaw)
        else:
            node.get_logger().info("No command. Use -h to see options.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
