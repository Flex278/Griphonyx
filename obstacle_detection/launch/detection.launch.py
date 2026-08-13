#!/usr/bin/env python3
"""
Launch file for obstacle detection node.

Usage:
    ros2 launch obstacle_detection detection.launch.py
    ros2 launch obstacle_detection detection.launch.py params_file:=/path/to/custom_params.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for obstacle detection."""
    
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('obstacle_detection'),
            'config',
            'params.yaml'
        ]),
        description='Path to parameter file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time (true for Gazebo)'
    )
    
    # Obstacle detector node
    obstacle_detector_node = Node(
        package='obstacle_detection',
        executable='obstacle_detector',
        name='obstacle_detector',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        emulate_tty=True
    )
    
    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        obstacle_detector_node
    ])
