# test/test_drone_camera.py
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32 as Num
from sensor_msgs.msg import Image, CameraInfo

import unittest
from termcolor import colored
from timeit import default_timer as timer
import numpy as np

import drone_utils

from anafi_ros_nodes.anafi_tester import AnafiTester
from anafi_ros_interfaces.msg import CameraCommand

TIMEOUT_LIMIT = 100  # Timeout limit in seconds
MAX_HFOV = 85.0
MIN_HFOV = 16.0

class TestAnafiCamera(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.delta_zoom = 0.5

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        namespace = '/anafi'
        self.test_node = AnafiTester(namespace=namespace)

        self.start_time_test = timer()

    def tearDown(self):
        self.test_node.destroy_node()

    def get_max_zoom_speed(self):
        future = self.test_node.request_max_zoom_speed()
        start_time = timer()
        while not future.done():
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, start_time)
        return future.result().values[0].double_value

    def get_zoom_level(self):
        start_time = timer()
        while np.isnan(self.test_node.camera_zoom):
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, start_time)
        return self.test_node.camera_zoom

    def test_get_max_zoom_speed(self):
        print("\n***** GET MAX ZOOM SPEED TEST *****")
        max_zoom_speed = drone_utils.get_drone_parameter(self.test_node, self.test_node.request_max_zoom_speed, 'double')

        try:
            self.assertNotEqual(max_zoom_speed, np.nan)
            print(colored("Test passed: duration get of max zoom speed: {:.3f}, max_zoom_speed: {:.3f}s".format(
                timer() - self.start_time_test, max_zoom_speed),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration get of max zoom speed: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_set_max_zoom_speed(self):
        print("\n***** SET MAX ZOOM SPEED TEST *****")

        starting_max_zoom_speed = drone_utils.get_drone_parameter(self.test_node, self.test_node.request_max_zoom_speed, 'double')
        new_max_zoom_speed = np.abs(starting_max_zoom_speed % 10) + 0.1
        future = self.test_node.set_max_zoom_speed(new_max_zoom_speed)
        start_time = timer()
        while not future.done():
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            # Check if the timeout has been reached
            elapsed_time = timer() - start_time
            if elapsed_time > TIMEOUT_LIMIT:
                raise Exception("Error: Out of time limit.")
            time.sleep(0.1)
        max_zoom_speed = drone_utils.get_drone_parameter(self.test_node, self.test_node.request_max_zoom_speed, 'double')

        try:
            self.assertEqual(max_zoom_speed, new_max_zoom_speed)
            print(colored("Test passed: duration set of max zoom speed: {:.3f}s, max_zoom_speed: {:.3f}".format(
                timer() - start_time, max_zoom_speed),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of max speed: {:.3f}s".format(
                    timer() - start_time),
                    'red'))
            raise e

    def test_camera_zoom_level(self):
        print("\n***** ZOOM LEVEL TEST *****")
        starting_zoom_level = self.get_zoom_level()
        print("curr zoom level is: ", starting_zoom_level)
        msg_camera_command = CameraCommand()
        msg_camera_command.mode = 0
        msg_camera_command.zoom = starting_zoom_level + 1
        #self.test_node.pub_zoom_level(msg_camera_command)
        self.test_node.pub_camera_command.publish(msg_camera_command)
        #TODO fix zoom level set never happening in simulated environment
        while not np.abs(self.get_zoom_level() - (starting_zoom_level + 1)) < self.delta_zoom:
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            #time.sleep(0.1)
        try:
            self.assertAlmostEqual(self.get_zoom_level(), starting_zoom_level + 1, delta=self.delta_zoom)

            print(colored("Test passed: duration set of zoom level: {:.3f}, zoom level: {:.3f}s".format(
                timer() - self.start_time_test, self.get_zoom_level()),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration set of zoom level: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_reset_zoom(self):
        print("\n***** ZOOM RESET TEST *****")

        #set to a different value from default (1.0)
        starting_zoom_level = self.get_zoom_level()
        while self.get_zoom_level() == 1.0:
            msg_camera_command = CameraCommand()
            msg_camera_command.mode = 0
            msg_camera_command.zoom = starting_zoom_level + 1
            #self.test_node.pub_zoom_level(msg_camera_command)
            self.test_node.pub_camera_command.publish(msg_camera_command)
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)

        #reset zoom to default value
        future = self.test_node.reset_zoom_level()
        while not future.done():
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            # Check if the timeout has been reached
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        try:
            self.assertAlmostEqual(self.get_zoom_level(), 1.0, delta=self.delta_zoom)

            print(colored("Test passed: duration reset of zoom level: {:.3f}, zoom level: {:.3f}".format(
                timer() - self.start_time_test, self.get_zoom_level()),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration reset of zoom level: {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_camera_info(self):
        print("\n***** CAMERA INFO TEST *****")

        while True:
            if isinstance(self.test_node.camera_camera_info, CameraInfo):
                break
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        camera_info = self.test_node.camera_camera_info

        try:
            self.assertIsNotNone(camera_info)

            # Check non-default values for height and width
            self.assertNotEqual(camera_info.height, 0, "Height should not be the default value of 0.")
            self.assertNotEqual(camera_info.width, 0, "Width should not be the default value of 0.")

            # Check that distortion model is set
            self.assertNotEqual(camera_info.distortion_model, "", "Distortion model should not be an empty string.")

            # Check that the distortion coefficients (d) are not empty and not all zeros
            self.assertTrue(camera_info.d, "Distortion coefficients (d) should not be empty.")
            self.assertTrue(any(coef != 0 for coef in camera_info.d),
                            "Distortion coefficients (d) should not all be zeros.")

            # Check that the camera matrix (k) is not all zeros
            self.assertTrue(any(val != 0 for val in camera_info.k), "Camera matrix (k) should not all be zeros.")

            # Check that the rectification matrix (r) is not all zeros
            self.assertTrue(any(val != 0 for val in camera_info.r), "Rectification matrix (r) should not all be zeros.")

            # Check that the projection matrix (p) is not all zeros
            self.assertTrue(any(val != 0 for val in camera_info.p), "Projection matrix (p) should not all be zeros.")

            print(colored(
                "Camera Info - Header: [frame_id: {}], Height: {}, Width: {}, Distortion Model: {}".format(
                    camera_info.header.frame_id,
                    camera_info.height,
                    camera_info.width,
                    camera_info.distortion_model
                ), 'green'))

            print(colored(
                "Distortion Coefficients (d): {}".format(
                    ', '.join(f"{coef:.3f}" for coef in camera_info.d)
                ), 'green'))

            print(colored(
                "Camera Matrix (k): {}".format(
                    ', '.join(f"{val:.3f}" for val in camera_info.k)
                ), 'green'))

            print(colored(
                "Rectification Matrix (r): {}".format(
                    ', '.join(f"{val:.3f}" for val in camera_info.r)
                ), 'green'))

            print(colored(
                "Projection Matrix (p): {}".format(
                    ', '.join(f"{val:.3f}" for val in camera_info.p)
                ), 'green'))

            print(colored(
                "Binning X: {}, Binning Y: {}".format(
                    camera_info.binning_x,
                    camera_info.binning_y
                ), 'green'))

            print(colored(
                "Region of Interest (ROI) - X offset: {}, Y offset: {}, Height: {}, Width: {}, Do Rectify: {}".format(
                    camera_info.roi.x_offset,
                    camera_info.roi.y_offset,
                    camera_info.roi.height,
                    camera_info.roi.width,
                    camera_info.roi.do_rectify
                ), 'green'))

        except AssertionError as e:
            print(colored("Test Failed: {}".format(str(e)), 'red'))
            raise e

    def test_image(self):
        print("\n***** IMAGE TOPIC TEST *****")

        while True:
            if isinstance(self.test_node.camera_image, Image):
                break
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        image_msg = self.test_node.camera_image

        try:
            self.assertIsNotNone(image_msg)

            # Assert that image data is not empty
            self.assertTrue(len(image_msg.data) > 0, "Image data should not be empty.")

            # Additional assertions to validate image dimensions and properties
            self.assertNotEqual(image_msg.height, 0, "Height should not be the default value of 0.")
            self.assertNotEqual(image_msg.width, 0, "Width should not be the default value of 0.")
            self.assertNotEqual(image_msg.step, 0, "Step (row length in bytes) should not be the default value of 0.")
            self.assertNotEqual(image_msg.encoding, "", "Encoding should not be an empty string.")

            print(colored("Image Info - Height: {}, Width: {}, Step: {}, Encoding: {}".format(
                image_msg.height,
                image_msg.width,
                image_msg.step,
                image_msg.encoding
            ), 'green'))

        except AssertionError as e:
            print(colored("Test Failed: {}".format(str(e)), 'red'))
            raise e

    def test_exposure_time(self):
        print("\n***** EXPOSURE TIME TOPIC TEST *****")

        drone_utils.init_attribute(self.test_node, "camera_exposure_time")

        exposure_time = self.test_node.camera_exposure_time

        try:
            self.assertIsNotNone(exposure_time)
            self.assertGreaterEqual(exposure_time, 0.0, "Exposure time should be greater then 0")

            print(colored("Exposure time: {}".format(
                exposure_time
            ), 'green'))

        except AssertionError as e:
            print(colored("Test Failed: {}".format(str(e)), 'red'))
            raise e

    def test_iso_gain(self):
        print("\n***** ISO GAIN TOPIC TEST *****")

        drone_utils.init_attribute(self.test_node, "camera_iso_gain")

        iso_gain = self.test_node.camera_iso_gain

        try:
            self.assertGreaterEqual(iso_gain, 25, "Iso gain should be greater or equal then 25")
            self.assertLessEqual(iso_gain, 51200, "Iso gain should be less then 51200")

            print(colored("iso gain: {}".format(
                iso_gain
            ), 'green'))

        except AssertionError as e:
            print(colored("Test Failed: {}".format(str(e)), 'red'))
            raise e

    def test_hfov(self):
        print("\n***** HFOV TOPIC TEST *****")

        drone_utils.init_attribute(self.test_node, "camera_hfov")

        hfov_rad = self.test_node.camera_hfov
        hfov_in_degree = hfov_rad * (180 / math.pi)

        try:
            self.assertGreaterEqual(hfov_in_degree, MIN_HFOV, "HFOV should be greater then 16 (tele camera)")
            self.assertLessEqual(hfov_in_degree, MAX_HFOV, "HFOV should be less then 85 (Wide Mode)")

            print(colored("hfov: {}".format(
                hfov_in_degree
            ), 'green'))

        except AssertionError as e:
            print(colored("Test Failed: {}".format(str(e)), 'red'))
            raise e

    def test_vfov(self):
        print("\n***** VFOV TOPIC TEST *****")

        drone_utils.init_attribute(self.test_node, "camera_vfov")

        vfov_rad = self.test_node.camera_vfov
        vfov_in_degree = vfov_rad * (180 / math.pi)
        aspect_ratio = 16/9

        try:
            self.assertGreaterEqual(vfov_in_degree, drone_utils.calculate_vfov(MIN_HFOV, aspect_ratio), "VFOV too small (tele mode)")
            self.assertLessEqual(vfov_in_degree, drone_utils.calculate_vfov(MAX_HFOV, aspect_ratio), "VFOV too big (Wide Mode)")

            print(colored("vfov: {}".format(
                vfov_in_degree
            ), 'green'))

        except AssertionError as e:
            print(colored("Test Failed: {}".format(str(e)), 'red'))
            raise e

    def test_take_photo(self):
        print("\n***** TAKE PHOTO TEST *****")

        #take a photo
        future = self.test_node.take_photo()
        while not future.done():
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)
        try:
            self.assertIsNotNone(future.result())
            if not drone_utils.is_simulated(self.test_node):
                self.assertNotEqual(future.result().media_id, '')

            print(colored("Test passed: duration take of a photo: {:.3f}, photo id: {}".format(
                timer() - self.start_time_test, future.result().media_id),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration : {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_video_recording(self):
        print("\n***** VIDEO RECORDING TEST *****")

        # start recording
        future = self.test_node.start_video_recording()
        while not future.done():
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        # stop recording
        future = self.test_node.stop_video_recording()
        while not future.done():
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            drone_utils.check_timeout(self.test_node, self.start_time_test)
            time.sleep(0.1)

        try:
            self.assertIsNotNone(future.result())
            if not drone_utils.is_simulated(self.test_node):
                self.assertNotEqual(future.result().media_id, '')

            print(colored("Test passed: duration video recording: {:.3f}, photo id: {}".format(
                timer() - self.start_time_test, future.result().media_id),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration : {:.3f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e

    def test_autorecord(self):
        print("\n***** VIDEO AUTORECORDING TEST *****")

        future = self.test_node.set_camera_mode(0)
        drone_utils.spin_until_done(self.test_node, future, self.start_time_test)
        camera_mode = drone_utils.get_drone_parameter(self.test_node, self.test_node.request_camera_mode, 'int')
        self.assertEqual(camera_mode, 0)

        autorecord_state = drone_utils.get_drone_parameter(self.test_node, self.test_node.request_autorecord, 'bool')
        if autorecord_state != True:
            future = self.test_node.set_autorecord(True)
            drone_utils.spin_until_done(self.test_node, future, self.start_time_test)
        autorecord_state = drone_utils.get_drone_parameter(self.test_node, self.test_node.request_autorecord, 'bool')

        # get current state
        drone_utils.init_drone_state(self.test_node)
        # set initial state to LANDED
        drone_utils.set_drone_initial_state(self.test_node, "LANDED")
        self.test_node.drone_takeoff()
        #TODO add check on recording active
        try:

            self.assertTrue(autorecord_state)

            print(colored("Test passed: duration get of autorecord: {:.3f}, autorecord state: {}".format(
                timer() - self.start_time_test, autorecord_state),
                'green'))

        except AssertionError as e:
            print(
                colored("Assertion Failed: duration get of autorecord: {:.0f}s".format(
                    timer() - self.start_time_test),
                    'red'))
            raise e


if __name__ == '__main__':
    unittest.main()
