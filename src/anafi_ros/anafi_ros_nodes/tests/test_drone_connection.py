#test/test_drone_connection.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32 as Num


import unittest
from termcolor import colored
import time
from timeit import default_timer as timer


from anafi_ros_nodes.anafi_tester import AnafiTester
import drone_utils


class TestAnafiConnection(unittest.TestCase):

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

    def test_skycontroller_connection(self):
        print("\n***** SKYCONTROLLER CONNECTION TEST *****")
        for _ in range(100):
            rclpy.spin_once(self.test_node,timeout_sec=0.5)
            if self.test_node.drone_state != "" and self.test_node.drone_state != "CONTROLLER_CONNECTING":
                break
            time.sleep(0.1)
        
        try:
            self.assertNotEqual(self.test_node.drone_state,'CONTROLLER_CONNECTING')
            self.assertNotEqual(self.test_node.drone_state,'')
        except AssertionError as e:
            print(colored("Assertion failed: drone state: {}".format(self.test_node.drone_state),'red'))
            raise e
    
    def test_drone_connection(self):
        print("\n***** DRONE CONNECTION TEST *****")
        for _ in range(100):
            rclpy.spin_once(self.test_node,timeout_sec=0.5)
            if self.test_node.drone_state != "" and self.test_node.drone_state != "DRONE_CONNECTING" and self.test_node.drone_state != "CONNECTED_SKYCONTROLLER" and self.test_node.drone_state != "SEARCHING_DRONE":
                break
            time.sleep(0.1)

        try:
            self.assertNotEqual(self.test_node.drone_state,'DRONE_CONNECTING')
            self.assertNotEqual(self.test_node.drone_state,'CONNECTED_SKYCONTROLLER')
            self.assertNotEqual(self.test_node.drone_state,'SEARCHING_DRONE')
            self.assertNotEqual(self.test_node.drone_state,'')
        except AssertionError as e:
            print(colored("Assertion failed: drone state: {}".format(self.test_node.drone_state),'red'))
            raise e

    def test_rssi(self):
        print("\n***** RSSI TOPIC TEST *****")
        if drone_utils.is_simulated(self.test_node):
            try:
                self.skipTest("rssi not available in simulation")
            except unittest.SkipTest as se:
                print(colored(f"Test skipped: {se}", "cyan"))
                raise se
        drone_utils.init_attribute(self.test_node, "link_rssi")
        rssi = None
        #check rssi for 10 seconds
        for _ in range(100):
            rclpy.spin_once(self.test_node, timeout_sec=0.5)
            rssi = self.test_node.link_rssi
            try:
                self.assertIsNotNone(rssi)
                self.assertGreater(rssi, -100)
                self.assertLess(rssi, 0)
            except AssertionError as e:
                print(colored("Test failed: rssi: {:.3f}".format(rssi), 'red'))
                raise e
            time.sleep(0.1)

        print(colored("Test passed: rssi: {:.3f}".format(rssi), 'green'))

    def test_link_goodput(self):
        print("\n***** LINK GOODPUT TOPIC TEST *****")

        drone_model = drone_utils.request_drone_model(self.test_node)
        if drone_model in {'4k', 'thermal', 'usa'} and not drone_utils.is_simulated(self.test_node):
            drone_utils.init_attribute(self.test_node, "link_goodput")
            drone_utils.init_attribute(self.test_node, "link_quality")
        else:
            try:
                self.skipTest("link goodput not available for this model or simulation")
            except unittest.SkipTest as se:
                print(colored(f"Test skipped: {se}", "cyan"))
                raise se
        try:
            self.assertIsNotNone(self.test_node.link_goodput)
            self.assertIsNotNone(self.test_node.link_quality)

            print(colored("Test passed: link goodput: {:.3f}, link quality: {:.3f}".format(self.test_node.link_goodput, self.test_node.link_quality), 'green'))
        except AssertionError as e:
            print(colored("Assertion failed: link goodput: {:.3f}, link quality: {:.3f}".format(self.test_node.link_goodput, self.test_node.link_quality), 'red'))
            raise e

    def test_link_quality(self):
        print("\n***** LINK QUALITY TOPIC TEST *****")

        if drone_utils.is_simulated(self.test_node):
            try:
                self.skipTest("link quality not available in simulation")
            except unittest.SkipTest as se:
                print(colored(f"Test skipped: {se}", "cyan"))
                raise se
        drone_utils.init_attribute(self.test_node, "link_rssi")
        drone_utils.init_attribute(self.test_node, "link_quality")

        rssi = self.test_node.link_rssi
        quality = self.test_node.link_quality
        try:
            self.assertIsNotNone(rssi)
            self.assertIsNotNone(quality)
            # Additional Condition: Fail test if rssi and quality report different quality
            if rssi < -60 and quality > 3:
                print(colored(f"Test failed: Invalid combination - link rssi: {rssi:.3f}, link quality: {quality:.3f}",
                              'red'))
                self.fail(f"Invalid combination - link rssi: {rssi:.3f}, link quality: {quality:.3f}")

            print(colored("Test passed: link rssi: {:.3f}, link quality: {:.3f}".format(self.test_node.link_rssi, self.test_node.link_quality), 'green'))
        except AssertionError as e:
            print(colored("Assertion failed: link rssi: {:.3f}, link quality: {:.3f}".format(self.test_node.link_rssi, self.test_node.link_quality), 'red'))
            raise e

if __name__== '__main__':
    unittest.main()