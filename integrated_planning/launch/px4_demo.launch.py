"""
Launch file for the PX4 bridge demo.

Starts the obstacle_map_bridge, integrated_planner_node, demo_driver and the
px4_mavlink_bridge so that a planned path is flown on PX4 SITL.

Usage:
    ros2 launch integrated_planning px4_demo.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return a LaunchDescription for the PX4 demo stack."""
    pkg_dir = get_package_share_directory("integrated_planning")
    default_params = os.path.join(pkg_dir, "config", "planner_params.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Planner params YAML",
    )
    connection_arg = DeclareLaunchArgument(
        "connection_url",
        default_value="udpin:127.0.0.1:14540",
        description="pymavlink connection string",
    )

    params_file = LaunchConfiguration("params_file")
    connection_url = LaunchConfiguration("connection_url")

    obstacle_map_bridge_node = Node(
        package="integrated_planning",
        executable="obstacle_map_bridge",
        name="obstacle_map_bridge",
        parameters=[params_file],
        output="screen",
        emulate_tty=True,
    )
    integrated_planner_node = Node(
        package="integrated_planning",
        executable="integrated_planner_node",
        name="integrated_planner_node",
        parameters=[params_file],
        output="screen",
        emulate_tty=True,
    )
    demo_driver_node = Node(
        package="integrated_planning",
        executable="demo_driver",
        name="demo_driver",
        output="screen",
        emulate_tty=True,
    )
    px4_mavlink_bridge_node = Node(
        package="integrated_planning",
        executable="px4_mavlink_bridge",
        name="px4_mavlink_bridge",
        parameters=[{"connection_url": connection_url}],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        params_arg,
        connection_arg,
        obstacle_map_bridge_node,
        integrated_planner_node,
        demo_driver_node,
        px4_mavlink_bridge_node,
    ])
