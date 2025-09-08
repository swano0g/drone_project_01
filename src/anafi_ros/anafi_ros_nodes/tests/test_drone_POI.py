import rclpy

import unittest
from termcolor import colored
from timeit import default_timer as timer

import drone_utils

from anafi_ros_nodes.anafi_tester import AnafiTester


class TestAnafiPOI(unittest.TestCase):

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

    def test_start_POI(self):
        print("\n***** START POI TEST *****")

        drone_utils.init_drone_state(self.test_node)

        # Initial state must be TAKING OFF, HOVERING or FLYING
        # HOVERING gets ignored by setDroneInitialState
        if self.test_node.drone_state != "TAKINGOFF" and self.test_node.drone_state != "FLYING":
            drone_utils.set_drone_initial_state(self.test_node, "HOVERING")

        location_POI = (2.0, 2.1, 35.0)
        locked_gimbal = False
        future = self.test_node.start_POI(location_POI, locked_gimbal)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertNotEqual(future.result(), None)

            print(colored("Test passed: duration start of POI: {:.3f}s ".format(
                timer() - self.start_time_test),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration start of POI: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e
    def test_stop_POI(self):
        print("\n***** STOP POI TEST *****")

        #start poi
        drone_utils.init_drone_state(self.test_node)

        # Initial state must be TAKING OFF, HOVERING or FLYING
        # HOVERING gets ignored by setDroneInitialState
        if self.test_node.drone_state != "TAKINGOFF" and self.test_node.drone_state != "FLYING":
            drone_utils.set_drone_initial_state(self.test_node, "HOVERING")

        location_POI = (2.0, 2.1, 35.0)
        locked_gimbal = False
        future = self.test_node.start_POI(location_POI, locked_gimbal)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        #stop poi
        future = self.test_node.stop_POI()
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertNotEqual(future.result(), None)
            # self.assertTrue(future.result().success)
            print(colored("Test passed: duration stop of POI: {:.3f}s with POI status: {}".format(
                timer() - self.start_time_test, future.result().message),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration stop of POI: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e
if __name__ == '__main__':
    unittest.main()
