#!/usr/bin/env python3
"""
Launch file for RRT* kinodynamic path planner.

Starts the RRT* planner node with configuration for large fixed-wing/VTOL UAVs.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for RRT* planner."""
    
    # Get package directory
    pkg_share = FindPackageShare('rrt_star_planner')
    
    # Path to config file
    config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'rrt_star_params.yaml'
    ])
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if running in Gazebo'
    )
    
    # RRT* planner node
    rrt_star_node = Node(
        package='rrt_star_planner',
        executable='rrt_star_planner',
        name='rrt_star_planner',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            # Add remappings if needed for your PX4 setup
        ]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        rrt_star_node
    ])
