"""
Launch file for the complete integrated path planning system.

Starts both the obstacle_map_bridge and integrated_planner_node with
parameters loaded from the shared YAML configuration file.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate the integrated system launch description.

    Returns:
        :class:`launch.LaunchDescription` containing both planning nodes.
    """
    pkg_dir = get_package_share_directory("integrated_planning")
    default_params = os.path.join(
        pkg_dir, "config", "planner_params.yaml"
    )

    # --- Launch arguments -----------------------------------------------------
    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the planner parameters YAML file.",
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Optional namespace for all nodes.",
    )

    params_file = LaunchConfiguration("params_file")
    namespace = LaunchConfiguration("namespace")

    # --- Nodes ----------------------------------------------------------------
    obstacle_map_bridge_node = Node(
        package="integrated_planning",
        executable="obstacle_map_bridge",
        name="obstacle_map_bridge",
        namespace=namespace,
        parameters=[params_file],
        output="screen",
        emulate_tty=True,
    )

    integrated_planner_node = Node(
        package="integrated_planning",
        executable="integrated_planner_node",
        name="integrated_planner_node",
        namespace=namespace,
        parameters=[params_file],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            params_arg,
            namespace_arg,
            obstacle_map_bridge_node,
            integrated_planner_node,
        ]
    )
