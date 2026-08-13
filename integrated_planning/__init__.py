"""
Integrated Path Planning Module

This module integrates obstacle detection with 3D path planning to enable
autonomous collision-free navigation for UAVs.

Components:
    - planners: Path planning algorithms (Hybrid A*, RRT*)
    - maps: Environment representations (VoxelMap, OccupancyGrid)
    - ros_integration: ROS2 nodes and interfaces
    - config: Configuration files and parameters
    - launch: ROS2 launch files
    - tests: Unit and integration tests

Author: Griphonyx Development Team
Date: 2026-03-12
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Griphonyx Development Team"

# Import core components for easy access
from .planners.hybrid_astar_3d import HybridAStarPlanner3D, VehicleConfig, FlightMode
from .maps.voxel_map_3d import VoxelMap

__all__ = [
    "HybridAStarPlanner3D",
    "VehicleConfig", 
    "FlightMode",
    "VoxelMap",
]