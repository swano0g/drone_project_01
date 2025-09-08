# test/test_drone_gimbal.py

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Int32 as Num

import unittest
from termcolor import colored
import time
from timeit import default_timer as timer
import numpy as np


import drone_utils

from anafi_ros_nodes.anafi_tester import AnafiTester
from anafi_ros_interfaces.msg import GimbalCommand


class TestAnafiGimbal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.delta = 0.5

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        namespace = '/anafi'
        self.test_node = AnafiTester(namespace=namespace)

        self.start_time_test = timer()
        future = self.test_node.reset_gimbal()
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

    def tearDown(self):
        self.test_node.destroy_node()

    def get_max_gimbal_speed(self):
        future = self.test_node.request_max_gimbal_speed()
        start_time = timer()
        drone_utils.spin_until_done(self.test_node, future, start_time)
        return future.result().values[0].double_value

    def set_gimbal_command(self, mode, frame, roll=None, pitch=None, yaw=None):
        """Helper function to publish a GimbalCommand."""
        msg = GimbalCommand()
        msg.mode = mode
        msg.frame = frame
        if roll is not None:
            msg.roll = roll
        if pitch is not None:
            msg.pitch = pitch
        if yaw is not None:
            msg.yaw = yaw
        self.test_node.pub_gimbal_command.publish(msg)

    def test_max_gimbal_speed(self):
        print("\n***** MAX GIMBAL SPEED TEST *****")
        max_gimbal_speed = self.get_max_gimbal_speed()
        future = self.test_node.set_max_gimbal_speed(max_gimbal_speed/2 + 1.1)
        start_time_set = timer()
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertEqual(self.get_max_gimbal_speed(), max_gimbal_speed/2 + 1.1)

            print(colored("Test passed: duration set of max gimbal speed: {:.3f}s, max gimbal speed: {:.3f}".format(
                timer() - self.start_time_test, self.get_max_gimbal_speed()),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of max gimbal speed: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_calibration(self):
        print("\n***** GIMBAL CALIBRATION TEST *****")

        future = self.test_node.calibrate_gimbal()
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertNotEqual(future.result(), None)

            print(colored("Test passed: duration calibration of gimbal: {:.3f}s".format(
                timer() - self.start_time_test),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration calibration of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e
#region ROLL

    def test_gimbal_absolute_roll_orientation(self):
        print("\n***** GIMBAL ABSOLUTE ROLL ORIENTATION TEST *****")

        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        # set gimbal command
        new_roll = 10.0
        self.set_gimbal_command(mode=0, frame=2, roll=new_roll)

        try:
            # TODO not working with real drone
            while not np.abs(self.test_node.gimbal_absolute.vector.x - new_roll) < self.delta:
                rclpy.spin_once(self.test_node, timeout_sec=0.5)
                drone_utils.check_timeout(self.test_node, self.start_time_test)
                time.sleep(0.1)

            self.assertAlmostEqual(self.test_node.gimbal_absolute.vector.x, new_roll, delta=self.delta)

            print(colored("Test passed: duration set of abs roll of gimbal: {:.3f}s, roll gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_absolute.vector.x),
                'green'))

        except TimeoutError as e:
            print(
                colored(f"Timeout Error: {str(e)}", 'red')
            )
            raise e
        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of abs roll of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_absolute_roll_velocity(self):
        print("\n***** GIMBAL ABSOLUTE ROLL VELOCITY TEST *****")

        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        starting_roll = self.test_node.gimbal_absolute.vector.x

        roll_vel = -1.0
        self.set_gimbal_command(mode=1, frame=2, roll=roll_vel)

        #TODO not working with real drone

        drone_utils.spin_for_duration(self.test_node, 100, 0.1)

        try:
            if np.sign(roll_vel) > 0:
                self.assertGreater(self.test_node.gimbal_absolute.vector.x, starting_roll)
            else:
                self.assertLess(self.test_node.gimbal_absolute.vector.x, starting_roll)
            print(colored("Test passed: duration set of abs roll of gimbal: {:.3f}s, roll gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_absolute.vector.x),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of abs roll of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e
    def test_gimbal_relative_roll_orientation(self):
        print("\n***** GIMBAL RELATIVE ROLL ORIENTATION TEST *****")

        self.start_time_test = timer()
        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        # set gimbal command
        new_roll = 20.0
        self.set_gimbal_command(mode=0, frame=1, roll=new_roll)

        while not np.abs(self.test_node.gimbal_relative.vector.x - new_roll) < self.delta:
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        try:
            self.assertAlmostEqual(self.test_node.gimbal_relative.vector.x, new_roll, delta=self.delta)

            print(colored("Test passed: duration set of rel roll of gimbal: {:.3f}s, roll gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_relative.vector.x),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of rel roll of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_relative_roll_velocity(self):
        print("\n***** GIMBAL RELATIVE ROLL VELOCITY TEST *****")

        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        starting_roll = self.test_node.gimbal_relative.vector.x

        roll_vel = -1.0
        self.set_gimbal_command(mode=1, frame=1, roll=roll_vel)

        """if np.sign(msg_gimbal_command.roll) > 0:
            while not self.test_node.gimbal_relative.vector.x > starting_roll:
                rclpy.spin_once(self.test_node, timeout_sec=0.5)
                drone_utils.check_timeout(self.test_node, start_time_test)
                time.sleep(0.1)
        else:
            while not self.test_node.gimbal_relative.vector.x < starting_roll:
                rclpy.spin_once(self.test_node, timeout_sec=0.5)
                drone_utils.check_timeout(self.test_node, start_time_test)
                time.sleep(0.1)"""

        drone_utils.spin_for_duration(self.test_node, 100, 0.1)

        try:
            if np.sign(roll_vel) > 0:
                self.assertGreater(self.test_node.gimbal_relative.vector.x, starting_roll)
            else:
                self.assertLess(self.test_node.gimbal_relative.vector.x, starting_roll)
            print(colored("Test passed: duration set of rel roll of gimbal: {:.3f}s, roll gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_relative.vector.x),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of rel roll of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    #endregion
#region PITCH
    def test_gimbal_absolute_pitch_orientation(self):
        print("\n***** GIMBAL ABSOLUTE PITCH ORIENTATION TEST *****")

        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        # set gimbal command
        new_pitch = 20.0
        self.set_gimbal_command(mode=0, frame=2, pitch=new_pitch)

        while not np.abs(self.test_node.gimbal_absolute.vector.y - new_pitch) < self.delta:
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        try:
            self.assertAlmostEqual(self.test_node.gimbal_absolute.vector.y, new_pitch, delta=self.delta)

            print(colored("Test passed: duration set of abs pitch of gimbal: {:.3f}s, pitch gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_absolute.vector.y),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of abs pitch of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_relative_pitch_orientation(self):
        print("\n***** GIMBAL RELATIVE PITCH ORIENTATION TEST *****")

        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        # set gimbal command
        new_pitch = 20.0
        self.set_gimbal_command(mode=0, frame=1, pitch=new_pitch)

        while not np.abs(self.test_node.gimbal_relative.vector.y - new_pitch) < self.delta:
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        try:
            self.assertAlmostEqual(self.test_node.gimbal_relative.vector.y, new_pitch, delta=self.delta)

            print(colored("Test passed: duration set of rel pitch of gimbal: {:.3f}s, pitch gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_relative.vector.y),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of rel pitch of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e
    def test_gimbal_relative_pitch_velocity(self):
        print("\n***** GIMBAL RELATIVE PITCH VELOCITY TEST *****")

        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        starting_pitch = self.test_node.gimbal_relative.vector.y

        # set gimbal command
        pitch_vel = -1.0
        self.set_gimbal_command(mode=1, frame=1, pitch=pitch_vel)

        drone_utils.spin_for_duration(self.test_node, 100, 0.1)

        try:
            if np.sign(pitch_vel) > 0:
                self.assertGreater(self.test_node.gimbal_relative.vector.y, starting_pitch)
            else:
                self.assertLess(self.test_node.gimbal_relative.vector.y, starting_pitch)
            print(colored("Test passed: duration set of rel pitch of gimbal: {:.3f}s, pitch gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_relative.vector.y),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of rel pitch of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_absolute_pitch_velocity(self):
        print("\n***** GIMBAL ABSOLUTE PITCH VELOCITY TEST *****")

        # initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        starting_pitch = self.test_node.gimbal_absolute.vector.y

        # set gimbal command
        pitch_vel = -1.0
        self.set_gimbal_command(mode=1, frame=2, pitch=pitch_vel)

        drone_utils.spin_for_duration(self.test_node, 100, 0.1)

        try:
            if np.sign(pitch_vel) > 0:
                self.assertGreater(self.test_node.gimbal_absolute.vector.y, starting_pitch)
            else:
                self.assertLess(self.test_node.gimbal_absolute.vector.y, starting_pitch)
            print(colored("Test passed: duration set of abs pitch of gimbal: {:.3f}s, pitch gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_absolute.vector.y),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of abs pitch of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e
    #endregion
#region YAW
    def test_gimbal_absolute_yaw_orientation(self):
        print("\n***** GIMBAL ABSOLUTE YAW ORIENTATION TEST *****")

        #initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        model = drone_utils.request_drone_model(self.test_node)
        try:
            if model != "ai":
                self.skipTest("mechanical yaw stabilization available only for ai model")
        except unittest.SkipTest as se:
            print(colored(f"Test skipped: {se}", "cyan"))
            raise se

        # set gimbal command
        new_yaw = 20.0
        self.set_gimbal_command(mode=0, frame=2, yaw=new_yaw)

        while not np.abs(self.test_node.gimbal_absolute.vector.z - new_yaw) < self.delta:
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        try:
            self.assertAlmostEqual(self.test_node.gimbal_absolute.vector.z, new_yaw, delta=self.delta)

            print(colored("Test passed: duration set of abs yaw of gimbal: {:.3f}s, yaw gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_absolute.vector.z),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of abs yaw of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_relative_yaw_orientation(self):
        print("\n***** GIMBAL RELATIVE YAW ORIENTATION TEST *****")

        # initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        model = drone_utils.request_drone_model(self.test_node)
        try:
            if model != "ai":
                self.skipTest("mechanical yaw stabilization available only for ai model")
        except unittest.SkipTest as se:
            print(colored(f"Test skipped: {se}", "cyan"))
            raise se

        # set gimbal command
        new_yaw = 20.0
        self.set_gimbal_command(mode=0, frame=1, yaw=new_yaw)

        while not np.abs(self.test_node.gimbal_relative.vector.z - new_yaw) < self.delta:
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        try:
            self.assertAlmostEqual(self.test_node.gimbal_relative.vector.z, new_yaw, delta=self.delta)

            print(colored("Test passed: duration set of rel yaw of gimbal: {:.3f}s, yaw gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_relative.vector.z),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of rel yaw of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_absolute_yaw_velocity(self):
        print("\n***** GIMBAL ABSOLUTE YAW VELOCITY TEST *****")

        # initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        model = drone_utils.request_drone_model(self.test_node)
        try:
            if model != "ai":
                self.skipTest("mechanical yaw stabilization available only for ai model")
        except unittest.SkipTest as se:
            print(colored(f"Test skipped: {se}", "cyan"))
            raise se

        starting_yaw = self.test_node.gimbal_absolute.vector.z

        # set gimbal command
        yaw_vel = -1.0
        self.set_gimbal_command(mode=1, frame=2, yaw=yaw_vel)

        drone_utils.spin_for_duration(self.test_node, 100, 0.1)

        try:
            if np.sign(yaw_vel) > 0:
                self.assertGreater(self.test_node.gimbal_absolute.vector.z, starting_yaw)
            else:
                self.assertLess(self.test_node.gimbal_absolute.vector.z, starting_yaw)
            print(colored("Test passed: duration set of abs yaw of gimbal: {:.3f}s, yaw gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_absolute.vector.z),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of abs yaw of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_relative_yaw_velocity(self):
        print("\n***** GIMBAL RELATIVE YAW VELOCITY TEST *****")

        # initialize gimbal attitude
        drone_utils.init_gimbal_attitude(self.test_node)

        model = drone_utils.request_drone_model(self.test_node)
        try:
            if model != "ai":
                self.skipTest("mechanical yaw stabilization available only for ai model")
        except unittest.SkipTest as se:
            print(colored(f"Test skipped: {se}", "cyan"))
            raise se

        starting_yaw = self.test_node.gimbal_relative.vector.z

        # set gimbal command
        yaw_vel = -1.0
        self.set_gimbal_command(mode=1, frame=1, yaw=yaw_vel)

        drone_utils.spin_for_duration(self.test_node, 100, 0.1)

        try:
            if np.sign(yaw_vel) > 0:
                self.assertGreater(self.test_node.gimbal_relative.vector.z, starting_yaw)
            else:
                self.assertLess(self.test_node.gimbal_relative.vector.z, starting_yaw)
            print(colored("Test passed: duration set of rel yaw of gimbal: {:.3f}s, yaw gimbal : {:.3f}".format(
                timer() - self.start_time_test, self.test_node.gimbal_relative.vector.z),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of rel yaw of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e
    #endregion
    def test_gimbal_reset(self):
        print("\n***** GIMBAL RESET TEST *****")

        future = self.test_node.reset_gimbal()
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertNotEqual(future.result(), None)
            #self.assertTrue(future.result().success)
            print(colored("Test passed: duration reset of gimbal: {:.3f}s".format(
                timer() - self.start_time_test),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration reset of gimbal: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_gimbal_attitude_from_metadata(self):
        print("\n***** GIMBAL ATTITUDE FROM METADATA TOPIC TEST *****")

        drone_utils.init_gimbal_attitude(self.test_node)

        metadata_gimbal_rpy = self.test_node.gimbal_from_metadata.vector

        metadata_roll = metadata_gimbal_rpy.x
        metadata_pitch = metadata_gimbal_rpy.y
        metadata_yaw = metadata_gimbal_rpy.z

        abs_roll = self.test_node.gimbal_absolute.vector.x
        abs_pitch = self.test_node.gimbal_absolute.vector.y
        abs_yaw = self.test_node.gimbal_absolute.vector.z

        try:
            self.assertAlmostEqual(metadata_roll, abs_roll, delta=self.delta)
            self.assertAlmostEqual(metadata_pitch, abs_pitch, delta=self.delta)
            self.assertAlmostEqual(metadata_yaw, abs_yaw, delta=self.delta)

            print(colored("Test passed: attitude of gimbal: roll: {:.3f} pitch: {:.3f} yaw: {:.3f}".format(
                metadata_roll, metadata_pitch, metadata_yaw
            ),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: attitude of gimbal, roll: {:.3f} pitch: {:.3f} yaw: {:.3f}".format(
                metadata_roll, metadata_pitch, metadata_yaw
            ),
                    'red'))
            raise e


if __name__ == '__main__':
    unittest.main()
