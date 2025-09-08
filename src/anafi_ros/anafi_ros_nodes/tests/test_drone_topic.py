import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32 as Num

import unittest
from termcolor import colored
import time
from timeit import default_timer as timer
from anafi_ros_nodes.anafi_tester import AnafiTester
from anafi_ros_interfaces.msg import PilotingCommand

import drone_utils


class TestAnafiTopic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        namespace = '/anafi'
        self.test_node = AnafiTester(namespace=namespace)

        self.start_time_test = timer()

    def tearDown(self):
        self.test_node.destroy_node()

    def test_state(self):
        print("\n***** STATE TOPIC TEST *****")
        did_connect = False
        state_retrieved = False
        did_disconnect = False

        valid_states = [
            "LANDED", "TAKINGOFF", "HOVERING", "FLYING", "LANDING",
            "EMERGENCY", "USER_TAKEOFF", "MOTOR_RAMPING", "EMERGENCY_LANDING"
        ]

        for _ in range(1000):
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            if self.test_node.drone_state == "CONTROLLER_CONNECTED" or self.test_node.drone_state == "DRONE_CONNECTED":
                did_connect = True
            if self.test_node.drone_state in valid_states:
                state_retrieved = True
            '''if self.test_node.drone_state == "DISCONNECTING":
                did_disconnect = True
                break'''
            time.sleep(0.01)

        try:
            self.assertTrue(state_retrieved)
            print(colored("Test passed: duration {:.3f}s, node connected during test: {})".format(
                timer() - self.start_time_test, did_connect), 'green'))
        except AssertionError as e:
            print(colored("Assertion failed: drone state: {}".format(self.test_node.drone_state), 'red'))
            raise e

    def test_battery(self):
        print("\n***** BATTERY TOPIC TEST *****")

        drone_utils.init_attribute(self.test_node, "battery_percentage")

        try:
            self.assertGreater(self.test_node.battery_percentage, -1)
            self.assertLess(self.test_node.battery_percentage, 101)

            print(colored("Test passed", 'green'))
        except AssertionError as e:
            print(colored("Assertion failed: battery value: {}".format(self.test_node.battery_percentage), 'red'))
            raise e

    def test_gps_fixed(self):
        print("\n***** GPS FIXED TOPIC TEST *****")

        drone_utils.init_attribute(self.test_node, "drone_gps_fix")

        try:
            self.assertTrue(isinstance(self.test_node.drone_gps_fix, bool))

            print(colored("Test passed: gps fixed {}".format("Yes" if self.test_node.drone_gps_fix else "No"), 'green'))
        except AssertionError as e:
            print(colored("Assertion failed: gps fixed {}".format("Yes" if self.test_node.drone_gps_fix else "No"),
                          'red'))
            raise e

    def test_gps_satellites(self):
        print("\n***** GPS SATELLITES TOPIC TEST *****")

        #move drone to force change of satellites
        drone_utils.init_drone_state(self.test_node)
        # Initial state must be TAKING OFF, HOVERING or FLYING
        # HOVERING gets ignored by setDroneInitialState
        if self.test_node.drone_state != "TAKINGOFF" and self.test_node.drone_state != "FLYING":
            drone_utils.set_drone_initial_state(self.test_node, "HOVERING")

        # send piloting command
        drone_command_msg = PilotingCommand()
        drone_command_msg.roll = 0.8
        drone_command_msg.pitch = 0.8
        drone_command_msg.yaw = 0.0
        drone_command_msg.gaz = 0.0

        # start test
        drone_utils.init_attribute(self.test_node, "drone_gps_satellites")
        # check number of satellites for 10 seconds
        for _ in range(100):
            self.test_node.pub_drone_command.publish(drone_command_msg)
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            satellites = self.test_node.drone_gps_satellites
            try:
                self.assertGreater(satellites, 0)
            except AssertionError as e:
                print(colored("Test failed: number of satellites: {:.3f}".format(satellites), 'red'))
                raise e
            time.sleep(0.1)

        print(colored("Test passed", 'green'))

    def test_gps_location(self):
        print("\n***** GPS LOCATION TOPIC TEST *****")

        drone_utils.init_drone_gps_position(self.test_node)

        results = {
            "latitude_is_nan": math.isnan(self.test_node.drone_gps_location[0]),
            "longitude_is_nan": math.isnan(self.test_node.drone_gps_location[1]),
            "altitude_is_nan": math.isnan(self.test_node.drone_gps_location[2]),
            #"latitude_covariance_is_nan": math.isnan(msg_gps_location.position_covariance[0]),
            #"longitude_covariance_is_nan": math.isnan(msg_gps_location.position_covariance[4]),
            #"altitude_covariance_is_nan": math.isnan(msg_gps_location.position_covariance[8])
        }
        valid_gps_data = True
        # If any value is NaN, return False
        if any(results.values()):
            valid_gps_data = False

        try:
            self.assertTrue(valid_gps_data)

            print(colored("Test passed: gps location is valid", 'green'))
        except AssertionError as e:
            print(colored("Assertion failed: gps location not valid", 'red'))
            raise e

    def test_skycontroller_attitude(self):
        print("\n***** SKYCONTROLLER RPY TOPIC TEST *****")

        drone_utils.init_skycontroller_attitude(self.test_node)

        skycontroller_rpy = self.test_node.skycontroller_rpy.vector

        try:
            self.assertTrue(
                drone_utils.is_rpy_within_range(
                    skycontroller_rpy.x, skycontroller_rpy.y, skycontroller_rpy.z
                )
            )

            print(colored("Test passed: attitude of skycontroller is valid:"
                          " roll: {:.3f} pitch: {:.3f} yaw: {:.3f}".format(
                skycontroller_rpy.x, skycontroller_rpy.y, skycontroller_rpy.z),
                'green'))
        except AssertionError as e:
            print(colored("Assertion failed: attitude of skycontroller is not valid"
                          " roll: {:.3f} pitch: {:.3f} yaw: {:.3f}".format(
                skycontroller_rpy.x, skycontroller_rpy.y, skycontroller_rpy.z), 'red'))
            raise e


if __name__ == '__main__':
    unittest.main()
