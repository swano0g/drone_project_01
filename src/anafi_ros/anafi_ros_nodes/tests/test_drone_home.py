import rclpy
import numpy as np

from timeit import default_timer as timer

import unittest
from termcolor import colored

from anafi_ros_nodes.anafi_tester import AnafiTester

import drone_utils


class TestAnafiHome(unittest.TestCase):
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

    def get_home_location(self):
        # Spin until the location is obtained
        start_time = timer()
        location_retrieved = False
        while not location_retrieved:
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, start_time)
            if self.test_node.home_location is not None:
                if not np.isnan(self.test_node.home_location[0]):
                    if not np.isnan(self.test_node.home_location[1]):
                        if not np.isnan(self.test_node.home_location[2]):
                            location_retrieved = True
        return self.test_node.home_location

    def get_home_type(self):
        future = self.test_node.request_home_type()
        start_time = timer()
        drone_utils.spin_until_done(self.test_node, future, start_time)
        return future.result().values[0].integer_value

    def test_set_home(self):
        print("\n***** SET HOME TEST *****")
        new_location = (52.0, 2.1, 0.0)
        future = self.test_node.set_home(new_location)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)
        home_location = self.get_home_location()  # Returns (x, y, z)

        try:
            self.assertNotEqual(future.result(), None)
            self.assertEqual(new_location, home_location)

            print(colored("Test passed: duration set of home: {:.3f}s with home location: (x: {:.3f}, y: {:.3f}, "
                          "z: {:.3f})".format(
                timer() - self.start_time_test, home_location[0], home_location[1], home_location[2]),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of home: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_start_navigate_home(self):
        print("\n***** NAVIGATE HOME TEST *****")
        #set new home
        new_location = (50.0, 2.1, 0.0)
        future = self.test_node.set_home(new_location)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        #Initial state must be TAKING OFF, HOVERING or FLYING
        #HOVERING gets ignored by set_drone_initial_state
        drone_utils.init_drone_state(self.test_node)
        if self.test_node.drone_state != "TAKINGOFF" and self.test_node.drone_state != "FLYING":
            drone_utils.set_drone_initial_state(self.test_node, "HOVERING")

        future = self.test_node.start_navigate_home()
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)


        try:
            self.assertNotEqual(future.result(), None)

            print(colored("Test passed: duration start navigate home: {:.3f}s with current home: {:-3f}".format(
                timer() - self.start_time_test, self.get_home_location()),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration start navigate home: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_set_home_type(self):
        print("\n***** SET HOME TYPE TEST *****")

        new_home_type = 1
        future = self.test_node.set_home_type(new_home_type)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertEqual(self.get_home_type(), new_home_type)

            print(colored("Test passed: duration set of home type: {:.3f}s, home type: {:.3f}".format(
                timer() - self.start_time_test, self.get_home_type()),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of home type: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_rth(self):
        print("\n***** RTH TEST *****")

        drone_utils.init_drone_state(self.test_node)

        # Initial state must be TAKING OFF, HOVERING or FLYING
        # HOVERING gets ignored by setDroneInitialState
        if self.test_node.drone_state != "TAKINGOFF" and self.test_node.drone_state != "FLYING":
            drone_utils.set_drone_initial_state(self.test_node, "HOVERING")

        #set new home to a fixed distance from current position
        drone_utils.init_drone_gps_position(self.test_node)
        curr_position = self.test_node.drone_gps_location
        delta_lat, delta_lon = drone_utils.convert_delta_meters_to_gps(self.test_node, 5, 5)
        new_location = (delta_lat + curr_position[0], delta_lon + curr_position[1], curr_position[2])
        future = self.test_node.set_home(new_location)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        #start rth
        future = self.test_node.request_rth()
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)

        try:
            self.assertNotEqual(future.result(), None)
            self.assertTrue(future.result().success)

            print(colored("Test passed: duration return to home: {:.3f}s".format(
                timer() - self.start_time_test),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration return home: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

if __name__ == '__main__':
    unittest.main()
