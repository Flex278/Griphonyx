#!/usr/bin/env python3
"""
Launch file for the Griphonyx costmap node.

Usage:
    ros2 launch costmap costmap.launch.py
    ros2 launch costmap costmap.launch.py params_file:=/path/to/custom.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('costmap'),
            'config',
            'costmap_params.yaml',
        ]),
        description='Path to costmap parameter YAML file',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time (true for Gazebo)',
    )

    costmap_node = Node(
        package='costmap',
        executable='costmap_node',
        name='costmap_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        emulate_tty=True,
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        costmap_node,
    ])
