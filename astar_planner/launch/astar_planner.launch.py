#!/usr/bin/env python3
"""
Launch file for the A* path planner node.

Starts the astar_node with parameters loaded from astar_params.yaml.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for A* planner."""

    pkg_share = FindPackageShare('astar_planner')

    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'astar_params.yaml'
    ])

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if running in Gazebo',
    )

    astar_node = Node(
        package='astar_planner',
        executable='astar_node',
        name='astar_planner',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        emulate_tty=True,
    )

    return LaunchDescription([
        use_sim_time_arg,
        astar_node,
    ])
