#!/usr/bin/env python3

import numpy as np
import rclpy
import math
import threading
import time
import datetime
import sys

from termcolor import colored
from timeit import default_timer as timer
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data, qos_profile_services_default, \
    qos_profile_parameters, qos_profile_parameter_events
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange, SetParametersResult
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import UInt8, UInt16, UInt32, UInt64, Int8, Float32, String, Header, Bool
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, TwistStamped, Vector3Stamped, Quaternion, \
    Twist, Vector3
from sensor_msgs.msg import Image, CameraInfo, NavSatFix
from builtin_interfaces.msg import Time
from std_srvs.srv import Trigger, SetBool
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters, GetParameters
from anafi_ros_interfaces.msg import PilotingCommand, MoveByCommand, MoveToCommand, CameraCommand, GimbalCommand, \
    SkycontrollerCommand, TargetTrajectory
from anafi_ros_interfaces.srv import PilotedPOI, FlightPlan, FollowMe, Location, Photo, Recording, String as StringSRV


class AnafiTester(Node):
    def __init__(self, namespace=''):
        super().__init__('anafi_tester', namespace=namespace)

        self.get_logger().info("anafi_tester is running...")

        # Publishers
        self.pub_drone_command = self.create_publisher(PilotingCommand, 'drone/command', qos_profile_system_default)
        self.pub_drone_moveto = self.create_publisher(MoveToCommand, 'drone/moveto', qos_profile_system_default)
        self.pub_drone_moveby = self.create_publisher(MoveByCommand, 'drone/moveby', qos_profile_system_default)
        self.pub_camera_command = self.create_publisher(CameraCommand, 'camera/command', qos_profile_system_default)
        self.pub_gimbal_command = self.create_publisher(GimbalCommand, 'gimbal/command', qos_profile_system_default)

        # Subscribers
        self.create_subscription(Image, 'camera/image', self.camera_image_callback, qos_profile_system_default)
        self.create_subscription(CameraInfo, 'camera/camera_info', self.camera_camera_info_callback,
                                 qos_profile_system_default)
        self.create_subscription(Time, 'time', self.time_callback, qos_profile_sensor_data)
        self.create_subscription(QuaternionStamped, 'drone/attitude', self.drone_attitude_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32, 'drone/altitude', self.drone_altitude_callback, qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, 'drone/speed', self.drone_speed_callback, qos_profile_sensor_data)
        self.create_subscription(UInt16, 'link/goodput', self.link_goodput_callback, qos_profile_system_default)
        self.create_subscription(UInt8, 'link/quality', self.link_quality_callback, qos_profile_system_default)
        self.create_subscription(Int8, 'link/rssi', self.link_rssi_callback, qos_profile_system_default)
        self.create_subscription(UInt8, 'battery/percentage', self.battery_percentage_callback,
                                 qos_profile_system_default)
        self.create_subscription(String, 'drone/state', self.drone_state_callback, qos_profile_system_default)
        self.create_subscription(Vector3Stamped, 'drone/rpy', self.drone_rpy_callback, qos_profile_sensor_data)
        self.create_subscription(Float32, 'camera/exposure_time', self.camera_exposure_time_callback,
                                 qos_profile_system_default)
        self.create_subscription(UInt16, 'camera/iso_gain', self.camera_iso_gain_callback, qos_profile_system_default)
        self.create_subscription(Float32, 'camera/awb_r_gain', self.camera_awb_r_gain_callback,
                                 qos_profile_system_default)
        self.create_subscription(Float32, 'camera/awb_b_gain', self.camera_awb_b_gain_callback,
                                 qos_profile_system_default)
        self.create_subscription(Float32, 'camera/hfov', self.camera_hfov_callback, qos_profile_system_default)
        self.create_subscription(Float32, 'camera/vfov', self.camera_vfov_callback, qos_profile_system_default)
        self.create_subscription(Bool, 'drone/gps/fix', self.drone_gps_fix_callback, qos_profile_system_default)
        self.create_subscription(Bool, 'drone/steady', self.drone_steady_callback, qos_profile_sensor_data)
        self.create_subscription(UInt8, 'battery/health', self.battery_health_callback, qos_profile_system_default)
        self.create_subscription(SkycontrollerCommand, 'skycontroller/command', self.skycontroller_command_callback,
                                 qos_profile_system_default)
        self.create_subscription(Float32, 'camera/zoom', self.camera_zoom_callback, qos_profile_system_default)
        self.create_subscription(UInt8, 'drone/gps/satellites', self.drone_gps_satellites_callback,
                                 qos_profile_system_default)
        self.create_subscription(Float32, 'drone/altitude_above_to', self.drone_altitude_above_to_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, 'drone/rpy_slow', self.drone_rpy_slow_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(NavSatFix, 'drone/gps/location', self.drone_gps_location_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'drone/pose', self.drone_pose_callback, qos_profile_sensor_data)
        self.create_subscription(Float32, 'battery/voltage', self.battery_voltage_callback, qos_profile_system_default)
        self.create_subscription(TargetTrajectory, 'target/trajectory', self.target_trajectory_callback,
                                 qos_profile_system_default)
        self.create_subscription(Vector3Stamped, 'gimbal/rpy_slow/relative', self.gimbal_relative_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, 'gimbal/rpy_slow/absolute', self.gimbal_absolute_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, 'gimbal/rpy', self.gimbal_from_metadata_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(UInt64, 'media/available', self.media_available_callback, qos_profile_system_default)
        self.create_subscription(PointStamped, 'home/location', self.home_location_callback, qos_profile_system_default)
        self.create_subscription(QuaternionStamped, 'skycontroller/attitude', self.skycontroller_attitude_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, 'skycontroller/rpy', self.skycontroller_rpy_callback,
                                 qos_profile_sensor_data)

        # Services
        self.drone_arm_client = self.create_client(SetBool, 'drone/arm')
        self.drone_takeoff_client = self.create_client(Trigger, 'drone/takeoff')
        self.drone_land_client = self.create_client(Trigger, 'drone/land')
        self.drone_emergency_client = self.create_client(Trigger, 'drone/emergency')
        self.drone_halt_client = self.create_client(Trigger, 'drone/halt')
        self.drone_rth_client = self.create_client(Trigger, 'drone/rth')
        self.drone_reboot_client = self.create_client(Trigger, 'drone/reboot')
        self.drone_calibrate_client = self.create_client(Trigger, 'drone/calibrate')
        self.skycontroller_offboard_client = self.create_client(SetBool, 'skycontroller/offboard')
        self.skycontroller_discover_drones_client = self.create_client(Trigger, 'skycontroller/discover_drones')
        self.skycontroller_forget_drone_client = self.create_client(Trigger, 'skycontroller/forget_drone')
        self.home_set_client = self.create_client(Location, 'home/set')
        self.home_navigate_client = self.create_client(SetBool, 'home/navigate')
        self.POI_start_client = self.create_client(PilotedPOI, 'POI/start')
        self.POI_stop_client = self.create_client(Trigger, 'POI/stop')
        self.flightplan_upload_client = self.create_client(FlightPlan, 'flightplan/upload')
        self.flightplan_start_client = self.create_client(FlightPlan, 'flightplan/start')
        self.flightplan_pause_client = self.create_client(Trigger, 'flightplan/pause')
        self.flightplan_stop_client = self.create_client(Trigger, 'flightplan/stop')
        self.followme_start_client = self.create_client(FollowMe, 'followme/start')
        self.followme_stop_client = self.create_client(Trigger, 'followme/stop')
        self.gimbal_reset_client = self.create_client(Trigger, 'gimbal/reset')
        self.gimbal_calibrate_client = self.create_client(Trigger, 'gimbal/calibrate')
        self.camera_reset_client = self.create_client(Trigger, 'camera/reset')
        self.camera_photo_take_client = self.create_client(Photo, 'camera/photo/take')
        self.camera_photo_stop_client = self.create_client(Trigger, 'camera/photo/stop')
        self.camera_recording_start_client = self.create_client(Recording, 'camera/recording/start')
        self.camera_recording_stop_client = self.create_client(Recording, 'camera/recording/stop')
        self.storage_download_client = self.create_client(Trigger, 'storage/download')
        self.storage_format_client = self.create_client(Trigger, 'storage/format')
        self.get_parameters_client = self.create_client(GetParameters, 'anafi/get_parameters')
        self.set_parameters_client = self.create_client(SetParameters, 'anafi/set_parameters')
        while not self.set_parameters_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('No connection...')

        # Variables
        self.camera_image = np.nan
        self.camera_camera_info = np.nan
        self.time = np.nan
        self.drone_attitude = np.nan
        self.drone_altitude = np.nan
        self.drone_speed = np.nan
        self.link_goodput = np.nan
        self.link_quality = np.nan
        self.link_rssi = np.nan
        self.battery_percentage = np.nan
        self.drone_state = ""
        self.drone_rpy = np.nan
        self.camera_exposure_time = np.nan
        self.camera_iso_gain = np.nan
        self.camera_awb_r_gain = np.nan
        self.camera_awb_b_gain = np.nan
        self.camera_hfov = np.nan
        self.camera_vfov = np.nan
        self.drone_gps_fix = np.nan
        self.drone_steady = np.nan
        self.battery_health = np.nan
        self.skycontroller_command = np.nan
        self.camera_zoom = np.nan
        self.drone_gps_satellites = np.nan
        self.drone_altitude_above_to = np.nan
        self.drone_rpy_slow = np.nan
        self.drone_gps_location = (np.nan, np.nan, np.nan)
        self.drone_pose = np.nan
        self.battery_voltage = np.nan
        self.target_trajectory = np.nan
        self.gimbal_relative = np.nan
        self.gimbal_absolute = np.nan
        self.gimbal_from_metadata = np.nan
        self.media_available = np.nan
        self.home_location = (np.nan, np.nan, np.nan)
        self.skycontroller_attitude = np.nan
        self.skycontroller_rpy = np.nan

    def send_emergency(self):
        req = Trigger.Request()
        future = self.drone_emergency_client.call_async(req)

    def drone_takeoff(self):
        req = Trigger.Request()
        future = self.drone_takeoff_client.call_async(req)

    def drone_land(self):
        req = Trigger.Request()
        future = self.drone_land_client.call_async(req)

    def request_max_zoom_speed(self):
        req = GetParameters.Request()
        req.names = ['camera/max_zoom_speed']
        return self.get_parameters_client.call_async(req)

    def set_max_zoom_speed(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='camera/max_zoom_speed',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_max_altitude(self):
        req = GetParameters.Request()
        req.names = ['drone/max_altitude']
        return self.get_parameters_client.call_async(req)

    def set_max_altitude(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='drone/max_altitude',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_max_distance(self):
        req = GetParameters.Request()
        req.names = ['drone/max_distance']
        return self.get_parameters_client.call_async(req)

    def request_max_horizontal_speed(self):
        req = GetParameters.Request()
        req.names = ['drone/max_horizontal_speed']
        return self.get_parameters_client.call_async(req)


    def set_max_horizontal_speed(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='drone/max_horizontal_speed',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_max_vertical_speed(self):
        req = GetParameters.Request()
        req.names = ['drone/max_vertical_speed']
        return self.get_parameters_client.call_async(req)

    def set_max_vertical_speed(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='drone/max_vertical_speed',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_max_pitch_roll(self):
        req = GetParameters.Request()
        req.names = ['drone/max_pitch_roll']
        return self.get_parameters_client.call_async(req)

    def set_max_pitch_roll(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='drone/max_pitch_roll',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_max_pitch_roll_rate(self):
        req = GetParameters.Request()
        req.names = ['drone/max_pitch_roll_rate']
        return self.get_parameters_client.call_async(req)

    def set_max_pitch_roll_rate(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='drone/max_pitch_roll_rate',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_max_yaw_rate(self):
        req = GetParameters.Request()
        req.names = ['drone/max_yaw_rate']
        return self.get_parameters_client.call_async(req)

    def set_max_yaw_rate(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='drone/max_yaw_rate',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)

    def set_max_distance(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='drone/max_distance',
                                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v))]
        return self.set_parameters_client.call_async(req)
    def request_max_gimbal_speed(self):
        req = GetParameters.Request()
        req.names = ['gimbal/max_speed']
        return self.get_parameters_client.call_async(req)

    def set_max_gimbal_speed(self, v):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='gimbal/max_speed', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                                                                  double_value=v))]
        return self.set_parameters_client.call_async(req)

    def reset_zoom_level(self):
        req = Trigger.Request()
        return self.camera_reset_client.call_async(req)

    def reset_gimbal(self):
        req = Trigger.Request()
        return self.gimbal_reset_client.call_async(req)

    def start_video_recording(self):
        req = Recording.Request()
        return self.camera_recording_start_client.call_async(req)

    def take_photo(self):
        req = Photo.Request()
        return self.camera_photo_take_client.call_async(req)

    def stop_video_recording(self):
        req = Recording.Request()
        return self.camera_recording_stop_client.call_async(req)

    def request_drone_ip(self):
        #TODO assign to a variable??
        req = GetParameters.Request()
        req.names = ['device/ip']
        return self.get_parameters_client.call_async(req)

    def request_drone_model(self):
        req = GetParameters.Request()
        req.names = ['drone/model']
        return self.get_parameters_client.call_async(req)

    def calibrate_gimbal(self):
        req = Trigger.Request()
        return self.gimbal_calibrate_client.call_async(req)

    def set_home(self, new_location):
        req = Location.Request()

        #set new home to given location
        req.latitude = new_location[0]
        req.longitude = new_location[1]
        req.altitude = new_location[2]
        return self.home_set_client.call_async(req)

    def request_home_type(self):
        req = GetParameters.Request()
        req.names = ['home/type']
        return self.get_parameters_client.call_async(req)

    def set_home_type(self, v):
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='home/type', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_offboard(self):
        req = GetParameters.Request()
        req.names = ['drone/offboard']
        return self.get_parameters_client.call_async(req)

    def set_offboard_parameter(self, v):
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='drone/offboard', value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v))]
        return self.set_parameters_client.call_async(req)

    def set_offboard(self, v):
        req = SetBool.Request()
        req.data = v
        return self.skycontroller_offboard_client.call_async(req)

    def request_autorecord(self):
        req = GetParameters.Request()
        req.names = ['camera/autorecord']
        return self.get_parameters_client.call_async(req)

    def set_autorecord(self, v):
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='camera/autorecord', value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v))]
        return self.set_parameters_client.call_async(req)

    def request_camera_mode(self):
        req = GetParameters.Request()
        req.names = ['camera/mode']
        return self.get_parameters_client.call_async(req)

    def set_camera_mode(self, mode):
        req = SetParameters.Request()
        req.parameters = [Parameter(name='camera/mode',
                                    value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=mode))]
        return self.set_parameters_client.call_async(req)

    def start_navigate_home(self):
        req = SetBool.Request()
        req.data = True
        return self.home_navigate_client.call_async(req)

    def stop_navigate_home(self):
        req = SetBool.Request()
        req.data = False
        return self.home_navigate_client.call_async(req)

    def arm_drone(self):
        req = SetBool.Request()
        req.data = True
        return self.drone_arm_client.call_async(req)

    def request_rth(self):
        req = Trigger.Request()
        return self.drone_rth_client.call_async(req)

    def calibrate_drone(self):
        req = Trigger.Request()
        return self.drone_calibrate_client.call_async(req)

    def halt_drone(self):
        req = Trigger.Request()
        return self.drone_halt_client.call_async(req)

    def start_POI(self, new_location, locked_gimbal):
        req = PilotedPOI.Request()

        #set new home to given location
        req.latitude = new_location[0]
        req.longitude = new_location[1]
        req.altitude = new_location[2]
        req.locked_gimbal = locked_gimbal

        return self.POI_start_client.call_async(req)

    def stop_POI(self):
        req = Trigger.Request()
        return self.POI_stop_client.call_async(req)

    def start_followme(self, mode, horizontal_framing,
                       vertical_framing, azimuth,
                       elevation, scale_change,
                       confidence, is_new_selection):
        req = FollowMe.Request()
        req.mode = mode
        req.horizontal = horizontal_framing
        req.vertical = vertical_framing
        req.target_azimuth = azimuth
        req.target_elevation = elevation
        req.change_of_scale = scale_change
        req.confidence_index = confidence
        req.is_new_selection = is_new_selection

        return self.followme_start_client.call_async(req)

    def camera_image_callback(self, msg):
        self.camera_image = msg

    def camera_camera_info_callback(self, msg):
        self.camera_camera_info = msg

    def time_callback(self, msg):
        self.time = msg.sec + msg.nanosec / 1e9

    def drone_attitude_callback(self, msg):
        self.drone_attitude = msg.quaternion

        #self.get_logger().info(
        #    f"Drone attitude: x={self.drone_attitude.x}, y={self.drone_attitude.y},z={self.drone_attitude.z}"
        #    f",w={self.drone_attitude.w}")

    def drone_altitude_callback(self, msg):
        self.drone_altitude = msg.data

    def drone_speed_callback(self, msg):
        self.drone_speed = msg

        #self.get_logger().info(
        #    f"Received speed direction - x: {self.drone_speed.vector.x}, y: {self.drone_speed.vector.y}, z: {self.drone_speed.vector.z}")

    def link_goodput_callback(self, msg):
        self.link_goodput = msg.data

    def link_quality_callback(self, msg):
        self.link_quality = msg.data

    def link_rssi_callback(self, msg):
        self.link_rssi = msg.data
        self.get_logger().info(
            f"Received rssi: {self.link_rssi}")

    def battery_percentage_callback(self, msg):
        self.battery_percentage = msg.data

    def drone_state_callback(self, msg):
        #print("state: " + str(msg.data))
        self.drone_state = msg.data
    def drone_rpy_callback(self, msg):
        self.drone_rpy = msg

    def camera_exposure_time_callback(self, msg):
        self.camera_exposure_time = msg.data

    def camera_iso_gain_callback(self, msg):
        self.camera_iso_gain = msg.data

    def camera_awb_r_gain_callback(self, msg):
        pass

    def camera_awb_b_gain_callback(self, msg):
        pass

    def camera_hfov_callback(self, msg):
        self.camera_hfov = msg.data

    def camera_vfov_callback(self, msg):
        self.camera_vfov = msg.data

    def drone_gps_fix_callback(self, msg):
        self.drone_gps_fix = msg.data

    def drone_steady_callback(self, msg):
        self.drone_steady = msg.data

    def battery_health_callback(self, msg):
        pass

    def skycontroller_command_callback(self, msg):
        pass

    def camera_zoom_callback(self, msg):
        #print("zoom level: ", msg.data)
        self.camera_zoom = msg.data

    def drone_gps_satellites_callback(self, msg):
        self.drone_gps_satellites = msg.data
        print("satellites: ", msg.data)

    def drone_altitude_above_to_callback(self, msg):
        pass

    def drone_rpy_slow_callback(self, msg):
        self.drone_rpy_slow = msg

    def drone_gps_location_callback(self, msg):
        self.get_logger().info(
            f"Received GPS location - Latitude: {msg.latitude}, Longitude: {msg.longitude}, Altitude: {msg.altitude}, "
            f"latitude covariance: {msg.position_covariance[0]}, longitude covariance: {msg.position_covariance[4]}, "
            f"altitude covariance: {msg.position_covariance[8]}")
        self.drone_gps_location = (msg.latitude, msg.longitude, msg.altitude)

    def drone_pose_callback(self, msg):
        self.drone_pose = msg.pose

    #self.get_logger().info(
    #	f"Received pose - x: {self.drone_pose.position.x}, y: {self.drone_pose.position.y}, z: {self.drone_pose.position.z}")

    def battery_voltage_callback(self, msg):
        pass

    def target_trajectory_callback(self, msg):
        pass

    def gimbal_relative_callback(self, msg):
        self.gimbal_relative = msg

    def gimbal_absolute_callback(self, msg):
        self.gimbal_absolute = msg

        # self.get_logger().info( f"Received gimbal absolute attitude - x: {self.gimbal_absolute.vector.x},
        # y: {self.gimbal_absolute.vector.y}, z: {self.gimbal_absolute.vector.z}")

    def gimbal_from_metadata_callback(self, msg):
        self.gimbal_from_metadata = msg

    def media_available_callback(self, msg):
        pass

    def home_location_callback(self, msg):
        self.home_location = (msg.point.x, msg.point.y, msg.point.z)

    def skycontroller_attitude_callback(self, msg):
        self.skycontroller_attitude = msg

    def skycontroller_rpy_callback(self, msg):
        self.skycontroller_rpy = msg


def main(args=None):
    rclpy.init(args=sys.argv)

    anafi_tester = AnafiTester()

    rclpy.spin(anafi_tester.node)

    anafi_tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
