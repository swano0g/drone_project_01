import rclpy

import unittest
from termcolor import colored
from timeit import default_timer as timer

import drone_utils

from anafi_ros_nodes.anafi_tester import AnafiTester

class TestAnafiFollowMe(unittest.TestCase):

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

    def test_start_followme(self):
        print("\n***** START FOLLOWME TEST *****")

        mode = 1  # look at

        horizontal_framing = 50  # 50% from left to right (center)
        vertical_framing = 50  # 50% from bottom to top (center)

        azimuth = 0.0  # The target is directly in front of the drone (0 radians)
        elevation = -0.1  # The target is slightly below the drone (0.1 radians)
        scale_change = 0.0  # The target is neither moving closer nor further away
        confidence = 255  # Maximum confidence in detection
        is_new_selection = True  # This is a new target selection

        future = self.test_node.start_followme(mode, horizontal_framing, vertical_framing,
                                               azimuth, elevation, scale_change,
                                               confidence, is_new_selection)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertNotEqual(future.result(), None)

            print(colored("Test passed: duration start of followme: {:.3f}s ".format(
                timer() - self.start_time_test),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration start of followme: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e




if __name__ == '__main__':
    unittest.main()
