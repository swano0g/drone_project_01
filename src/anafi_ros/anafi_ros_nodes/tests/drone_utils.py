import rclpy
from timeit import default_timer as timer
import time
import math
import numpy as np
from geometry_msgs.msg import Vector3Stamped



# Define a constant for the timeout limit
TIMEOUT_LIMIT = 100.0  # Adjust this value as needed


def request_drone_ip(test_node):
    """Request the drone's IP address asynchronously."""
    future = test_node.request_drone_ip()
    start_time = timer()
    while not future.done():
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)
    return future.result().values[0].string_value


def request_drone_model(test_node):
    future = test_node.request_drone_model()
    start_time = timer()
    while not future.done():
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)
    return future.result().values[0].string_value


def is_simulated(test_node):
    """Check if the drone is simulated by comparing its IP address."""
    ip = request_drone_ip(test_node)
    if ip == '10.202.0.1':
        return True
    return False


def spin_until_done(test_node, future, start_time):
    """Helper function to spin the node until a future is done or timeout occurs."""
    while not future.done():
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)


def spin_for_duration(test_node, steps, delay=0.1):
    """
    Spins the node for the specified number of steps with a delay between each step.

    Args:
        test_node(AnafiTester): test node.
        steps (int): Number of spin steps.
        delay (float): Time to wait between each spin step (in seconds).
    """
    for _ in range(steps):
        rclpy.spin_once(test_node, timeout_sec=0.5)  # Spin node for 0.5 seconds each step
        time.sleep(delay)  # Sleep for the specified delay


def init_drone_state(test_node):
    start_time = timer()
    """
    Spin until the initial state is obtained
    """
    initial_state_retrieved = False
    while not initial_state_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)
        if test_node.drone_state != "":
            initial_state_retrieved = True


def init_drone_attitude(test_node):
    start_time = timer()
    """
    Spin until the initial state is obtained
    """
    attitude_retrieved = False
    while not attitude_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)
        if test_node.drone_attitude is not None:
            attitude_retrieved = True


def is_rpy_within_range(roll, pitch, yaw, roll_range=(-180, 180), pitch_range=(-90, 90), yaw_range=(-180, 180)):

    roll_within_range = roll_range[0] <= roll <= roll_range[1]
    pitch_within_range = pitch_range[0] <= pitch <= pitch_range[1]
    yaw_within_range = yaw_range[0] <= yaw <= yaw_range[1]

    return roll_within_range and pitch_within_range and yaw_within_range


def is_moved_by(test_node, distance, time_interval=0.01, max_duration=120.0, moved_delta=0.5, piloting_msg=None):
    init_attribute(test_node, "time")
    start_time = test_node.time
    last_update_time = start_time
    distance_acc = [0, 0, 0]

    while True:
        # Spin to handle callbacks frequently
        rclpy.spin_once(test_node, timeout_sec=0.1)
        if piloting_msg is not None:
            test_node.pub_drone_command.publish(piloting_msg)
        # Get the current time
        current_time = test_node.time
        # Check if the maximum allowable time has passed
        #print("current time: ",current_time)
        #print("start time:",start_time)
        if current_time - start_time >= max_duration:
            print("Maximum duration reached without achieving target distance.")
            return False
        # Update and check distance only after each `time_interval` has passed
        if current_time - last_update_time >= time_interval:
            # Calculate the actual time elapsed since the last update
            elapsed_time = current_time - last_update_time

            # Update accumulated distances for each axis
            for i, axis in enumerate(["x", "y", "z"]):
                speed = getattr(test_node.drone_speed.vector, axis)
                if abs(speed) >= 0.005:  # Consider small movements only above threshold
                    distance_acc[i] += speed * elapsed_time

            # Print current speed and accumulated distance
            #print("Current speed vector:", test_node.drone_speed.vector)
            #print("Accumulated distance:", distance_acc)

            # Check if accumulated distance exceeds the target distance on all axes
            #goal distance and vector speed on y and z have inverted sign
            if (abs(distance_acc[0] - distance[0]) < moved_delta and
                    abs(distance_acc[1] + distance[1]) < moved_delta and
                    abs(distance_acc[2] + distance[2]) < moved_delta):
                return True

            # Update last_update_time for the next interval check
            last_update_time = current_time

        # Sleep briefly to allow frequent spins
        time.sleep(0.001)


def store_max_horizontal_speed(test_node, time_interval):
    init_drone_speed(test_node)
    max_h_speed_reached = 0.0
    for _ in range(time_interval * 10):
        rclpy.spin_once(test_node, timeout_sec=0.5)
        curr_h_speed = math.sqrt(pow(test_node.drone_speed.vector.x, 2) +
                                 pow(test_node.drone_speed.vector.y, 2))
        if curr_h_speed > max_h_speed_reached:
            max_h_speed_reached = curr_h_speed
        time.sleep(0.1)
    return max_h_speed_reached


def store_max_vertical_speed(test_node, time_interval):
    init_drone_speed(test_node)
    max_v_speed_reached = 0.0
    for _ in range(time_interval * 10):
        rclpy.spin_once(test_node, timeout_sec=0.5)
        curr_v_speed = abs(test_node.drone_speed.vector.z)
        if curr_v_speed > max_v_speed_reached:
            max_v_speed_reached = curr_v_speed
        time.sleep(0.1)
    return max_v_speed_reached


def store_max_tilt(test_node, time_interval):
    init_drone_rpy(test_node)
    max_tilt_reached = 0.0
    for _ in range(time_interval * 10):
        rclpy.spin_once(test_node, timeout_sec=0.5)
        rpy_vector = test_node.drone_rpy.vector
        print(rpy_vector)
        curr_tilt = max(abs(rpy_vector.x), abs(rpy_vector.y))
        if curr_tilt > max_tilt_reached:
            max_tilt_reached = curr_tilt
        time.sleep(0.1)
    return max_tilt_reached


def increment_within_range(x, increment, min_val, max_val):
    if not (min_val <= x <= max_val):
        raise ValueError(f"x must be within the range [{min_val}, {max_val}].")

    range_size = max_val - min_val
    x = min_val + (x + increment - min_val) % range_size

    return x


def init_drone_altitude(test_node):
    # Spin until the altitude is obtained
    altitude_retrieved = False
    while not altitude_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        if test_node.drone_altitude != np.nan:
            altitude_retrieved = True
    return test_node.drone_altitude


def set_drone_initial_state(test_node, drone_state):
    # Dictionary of corresponding actions
    state_actions = {
        "LANDED": test_node.drone_land,
        "HOVERING": test_node.drone_takeoff,  #TODO add condition for each starting state
        # Add more states and their corresponding methods as needed
    }

    if test_node.drone_state != drone_state:
        # Trigger the appropriate action
        if drone_state in state_actions:
            state_actions[drone_state]()
        else:
            raise ValueError(f"Unknown drone state: {drone_state}")

        start_time = time.time()
        while test_node.drone_state != drone_state:
            rclpy.spin_once(test_node, timeout_sec=0.5)
            if test_node.drone_state == drone_state:
                break

            # Check if the timeout has been reached
            check_timeout(test_node, start_time)

            time.sleep(1)


def convert_delta_meters_to_gps(test_node, distance_north, distance_east):
    # Earth's radius in meters
    earth_radius = 6378137.0

    # Calculate delta in latitude (in degrees)
    delta_latitude = distance_north / 111000.0  # 111 km per degree latitude

    init_drone_gps_position(test_node)
    location = test_node.drone_gps_location
    # Calculate delta in longitude (in degrees)
    delta_longitude = distance_east / (111000.0 * math.cos(math.radians(location[0])))
    return delta_latitude, delta_longitude


def calculate_vfov(hfov_degrees, aspect_ratio):
    # Convert HFOV from degrees to radians
    hfov_radians = math.radians(hfov_degrees)

    # Calculate VFOV in radians
    vfov_radians = 2 * math.atan(math.tan(hfov_radians / 2) / aspect_ratio)

    # Convert VFOV back to degrees
    vfov_degrees = math.degrees(vfov_radians)

    return vfov_degrees


def init_drone_gps_position(test_node):
    # Spin until the gps position is obtained
    position_retrieved = False
    while not position_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        if test_node.drone_gps_location is not None:
            if not np.isnan(test_node.drone_gps_location[0]):
                if not np.isnan(test_node.drone_gps_location[1]):
                    if not np.isnan(test_node.drone_gps_location[2]):
                        position_retrieved = True


def init_drone_rpy(test_node):
    # Spin until the drone rpy is obtained
    rpy_retrieved = False
    while not rpy_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        # Ensure is instances of Vector3Stamped
        if isinstance(test_node.drone_rpy, Vector3Stamped):
            rpy_vector = test_node.drone_rpy.vector
            if not (np.isnan(rpy_vector.x) or np.isnan(rpy_vector.y) or np.isnan(rpy_vector.z)):
                rpy_retrieved = True


def init_drone_speed(test_node):
    # Spin until the drone optical speed is obtained
    speed_retrieved = False
    while not speed_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        # Ensure is instances of Vector3Stamped
        if isinstance(test_node.drone_speed, Vector3Stamped):
            speed_vector = test_node.drone_speed.vector
            if not (np.isnan(speed_vector.x) or np.isnan(speed_vector.y) or np.isnan(speed_vector.z)):
                speed_retrieved = True


def init_drone_rpy_slow(test_node):
    # Spin until the drone rpy slow is obtained
    rpy_slow_retrieved = False
    while not rpy_slow_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        # Ensure is instances of Vector3Stamped
        if isinstance(test_node.drone_rpy_slow, Vector3Stamped):
            rpy_slow_vector = test_node.drone_rpy_slow.vector
            if not (np.isnan(rpy_slow_vector.x) or np.isnan(rpy_slow_vector.y) or np.isnan(rpy_slow_vector.z)):
                rpy_slow_retrieved = True


def init_drone_pose(test_node):
    start_time = timer()
    # Spin until the gps position is obtained
    position_retrieved = False
    while not position_retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)
        if test_node.drone_pose is not None:
            position_retrieved = True


def init_gimbal_attitude(test_node):
    start_time = timer()
    # Flag to track whether the gimbal attitude has been successfully retrieved
    attitude_retrieved = False

    while not attitude_retrieved:
        # Spin once for 0.5 seconds to allow for callbacks
        rclpy.spin_once(test_node, timeout_sec=0.5)

        # Check for timeout condition
        check_timeout(test_node, start_time)

        # Check if the gimbal attitudes are valid Vector3Stamped instances
        if is_gimbal_attitude_valid(test_node):
            attitude_retrieved = True


def init_skycontroller_attitude(test_node):
    start_time = timer()

    sky_attitude_retrieved = False

    while not sky_attitude_retrieved:
        # Spin once for 0.5 seconds to allow for callbacks
        rclpy.spin_once(test_node, timeout_sec=0.5)

        # Check for timeout condition
        check_timeout(test_node, start_time)

        if is_vector3_stamped_valid(test_node.skycontroller_rpy):
            sky_attitude_retrieved = True

def is_gimbal_attitude_valid(test_node):
    """Checks if gimbal attitudes (absolute, relative, metadata) are valid."""
    return (
        is_vector3_stamped_valid(test_node.gimbal_absolute) and
        is_vector3_stamped_valid(test_node.gimbal_relative) and
        is_vector3_stamped_valid(test_node.gimbal_from_metadata)
    )


def is_vector3_stamped_valid(vector3_stamped):
    """Checks if a Vector3Stamped instance contains valid coordinates."""
    if isinstance(vector3_stamped, Vector3Stamped):
        vector = vector3_stamped.vector
        return not (np.isnan(vector.x) or np.isnan(vector.y) or np.isnan(vector.z))
    return False


def init_attribute(test_node, attribute_name):
    start_time = timer()
    # Spin until the specified attribute is obtained
    retrieved = False
    while not retrieved:
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)
        # Access the attribute dynamically using getattr
        attribute_value = getattr(test_node, attribute_name, None)
        if attribute_value is not None and isinstance(attribute_value, (int, float, bool)) and not np.isnan(attribute_value):
            retrieved = True


def get_drone_location(test_node):
    # Determine if the drone is a simulation or in real-life

    if is_simulated(test_node):
        print("GETTING POSE: DRONE IS SIMULATED...")
        init_drone_pose(test_node)
        position = test_node.drone_pose.position
        location = (position.x, position.y, position.z)  # lat | lon | alt
        return location
    else:
        init_drone_gps_position(test_node)
        location = test_node.drone_gps_location
        return location



def check_timeout(test_node, start_time):
    """Check if the operation has timed out."""
    elapsed_time = timer() - start_time
    if elapsed_time > TIMEOUT_LIMIT:
        raise TimeoutError(
            f"Operation timed out after {elapsed_time:.3f} seconds with drone state {test_node.drone_state}")


def get_drone_parameter(test_node, request_method):
    """
    Retrieves a specific parameter from the drone.

    Parameters:
        request_method (callable): The method to call on self.test_node to request the parameter.

    Returns:
        float: The requested parameter value.
    """
    future = request_method()
    start_time = timer()
    while not future.done():
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)
    return future.result().values[0].double_value


def get_drone_parameter(test_node, request_method):
    """
    Retrieves a specific parameter from the drone.

    Parameters:
        request_method (callable): The method to call on self.test_node to request the parameter.

    Returns:
        bool: The requested parameter value.
    """
    future = request_method()
    start_time = timer()

    # Wait for the future to complete
    while not future.done():
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)

    try:
        # Retrieve the result
        response = future.result()
        if response.values:
            return response.values[0].bool_value  # Correctly access the boolean value
        else:
            test_node.get_logger().error("Parameter not found in response.")
            return False  # Default value if parameter is not found
    except Exception as e:
        test_node.get_logger().error(f"Error retrieving parameter: {e}")
        return False

def get_drone_parameter(test_node, request_method, param_type):
    """
    Retrieves a specific parameter from the drone.

    Parameters:
        test_node (rclpy.node.Node): The ROS2 node used for spinning and handling callbacks.
        request_method (callable): The method to call on the node to request the parameter.
        param_type (str): The expected type of the parameter. Must be one of 'double', 'bool', or 'string'.

    Returns:
        Union[float, bool, str]: The requested parameter value if found and of the correct type.
    """
    future = request_method()
    start_time = timer()

    # Wait for the future to complete
    while not future.done():
        rclpy.spin_once(test_node, timeout_sec=0.5)
        check_timeout(test_node, start_time)

    try:
        # Retrieve the result
        response = future.result()
        if response.values:
            param_value = response.values[0]

            # Check the type and return the corresponding value
            if param_type == 'double':
                return param_value.double_value
            elif param_type == 'bool':
                return param_value.bool_value
            elif param_type == 'string':
                return param_value.string_value
            elif param_type == 'int':
                return param_value.integer_value
            else:
                test_node.get_logger().error(f"Unsupported parameter type: {param_type}")
                return None
        else:
            test_node.get_logger().error("Parameter not found in response.")
            return None
    except Exception as e:
        test_node.get_logger().error(f"Error retrieving parameter: {e}")
        return None

