from setuptools import setup
import os
from glob import glob

package_name = 'integrated_planning'

setup(
    name=package_name,
    version='1.0.0',
    packages=['integrated_planning',
              'integrated_planning.maps',
              'integrated_planning.planners',
              'integrated_planning.ros_integration'],
    package_dir={'integrated_planning': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Griphonyx Team',
    maintainer_email='griphonyx@example.com',
    description='3D Hybrid A* path planning integrated with obstacle detection for Griphonyx UAV',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'integrated_planner_node = integrated_planning.ros_integration.integrated_planner_node:main',
            'obstacle_map_bridge = integrated_planning.ros_integration.obstacle_map_bridge:main',
            'demo_driver = integrated_planning.ros_integration.demo_driver:main',
            'px4_mavlink_bridge = integrated_planning.ros_integration.px4_mavlink_bridge:main',
        ],
    },
)
