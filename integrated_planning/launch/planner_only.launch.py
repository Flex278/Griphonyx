"""
Launch file for the integrated planner node only (no obstacle detection).

Useful for testing path planning in isolation or with a pre-built map.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate the planner-only launch description.

    Returns:
        :class:`launch.LaunchDescription` with only the planner node.
    """
    pkg_dir = get_package_share_directory("integrated_planning")
    default_params = os.path.join(
        pkg_dir, "config", "planner_params.yaml"
    )

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the planner parameters YAML file.",
    )

    params_file = LaunchConfiguration("params_file")

    integrated_planner_node = Node(
        package="integrated_planning",
        executable="integrated_planner_node",
        name="integrated_planner_node",
        parameters=[params_file],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([params_arg, integrated_planner_node])
