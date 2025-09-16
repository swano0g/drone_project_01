from setuptools import find_packages, setup

package_name = 'anafi_ai_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/anafi_bridge.launch.py']),
        ('share/' + package_name + '/launch', ['launch/full_follower.launch.py']),
        ('share/' + package_name + '/launch', ['launch/dryrun_test.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rubis',
    maintainer_email='dlghdydwkd79@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_bridge = anafi_ai_bridge.state_bridge_node:main',
            'crazyflie_follower = anafi_ai_bridge.crazyflie_follower_node:main',
            'rtsp_pub = anafi_ai_bridge.rtsp_image_pub_node:main',
            'anafi_simple_control = anafi_ai_bridge.anafi_simple_control:main',
            'anafi_sensing_node = anafi_ai_bridge.anafi_sensing_node:main',
            'anafi_control_node = anafi_ai_bridge.anafi_control_node:main',
            'creative_behavior_follow_cf = anafi_ai_bridge.creative_behavior_follow_cf:main',
            'anafi_follow_blue = anafi_ai_bridge.anafi_follow_blue:main',
            'anafi_follow_apriltag = anafi_ai_bridge.anafi_follow_apriltag:main',
            'apriltag_debug_overlay = anafi_ai_bridge.apriltag_debug_overlay:main',
        ],
    },
)
