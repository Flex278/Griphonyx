#!/usr/bin/env python3
"""
Файл запуска полной системы Griphonyx

Запускает полный стек обнаружения препятствий и планирования пути A*:
  1. obstacle_detector     — LiDAR + отслеживание Калманом + классификация зон
  2. costmap_node          — OccupancyGrid на основе LiDAR + отслеживаемых препятствий
  3. astar_planner         — Глобальный планировщик пути A* (заменяет RRT*)
  4. integrated_planner    — Координирует глобальный путь A* с зонным обходом

Использование:
    ros2 launch integrated_planning full_system.launch.py
    ros2 launch integrated_planning full_system.launch.py use_sim_time:=false

Поток топиков:
    /scan / /lidar/points → obstacle_detector → /avoidance_command, /obstacle_velocity,
                                                /obstacle_detected, /obstacle_distance,
                                                /costmap/update_trigger
                         → costmap_node      → /costmap/grid
                         → astar_planner     → /planned_path
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Использовать симуляционное время (true для Gazebo)',
    )
    sim_time = LaunchConfiguration('use_sim_time')

    # ── Обнаружение препятствий ──────────────────────────────────────────
    obstacle_detector_node = Node(
        package='obstacle_detection',
        executable='obstacle_detector',
        name='obstacle_detector',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('obstacle_detection'), 'config', 'params.yaml'
            ]),
            {'use_sim_time': sim_time},
        ],
        emulate_tty=True,
    )

    # ── Costmap ──────────────────────────────────────────────────────────
    costmap_node = Node(
        package='costmap',
        executable='costmap_node',
        name='costmap_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('costmap'), 'config', 'costmap_params.yaml'
            ]),
            {'use_sim_time': sim_time},
        ],
        emulate_tty=True,
    )

    # ── Планировщик A* ──────────────────────────────────────────────────
    astar_node = Node(
        package='astar_planner',
        executable='astar_node',
        name='astar_planner',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('astar_planner'), 'config', 'astar_params.yaml'
            ]),
            {'use_sim_time': sim_time},
        ],
        emulate_tty=True,
    )

    # ── Координатор интегрированного планирования ────────────────────────
    planning_coordinator_node = Node(
        package='integrated_planning',
        executable='integrated_planner_node',
        name='planning_coordinator',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('integrated_planning'), 'config', 'planner_params.yaml'
            ]),
            {'use_sim_time': sim_time},
        ],
        emulate_tty=True,
    )

    return LaunchDescription([
        use_sim_time_arg,
        obstacle_detector_node,
        costmap_node,
        astar_node,
        planning_coordinator_node,
    ])
