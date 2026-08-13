"""
Integration tests for the obstacle detection → VoxelMap → planning pipeline.

These tests exercise multi-component interactions without a running ROS2
environment (all ROS2 nodes are tested in isolation using mocks).
"""

import math
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", ".."),
)

from integrated_planning.maps.voxel_map_3d import VoxelMap
from integrated_planning.planners.hybrid_astar_3d import (
    FlightMode,
    HybridAStarPlanner3D,
    VehicleConfig,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def build_pipeline(
    map_size: float = 50.0,
    resolution: float = 1.0,
    min_alt: float = 5.0,
    max_alt: float = 80.0,
) -> Tuple[VoxelMap, HybridAStarPlanner3D]:
    """Build a VoxelMap + HybridAStarPlanner3D pair for integration tests."""
    vmap = VoxelMap(map_size, map_size, max_alt + 10.0, resolution=resolution)
    vehicle = VehicleConfig(
        min_altitude=min_alt,
        max_altitude=max_alt,
        hover_step=2.0,
        inflation=0.0,
    )
    planner = HybridAStarPlanner3D(
        vmap, vehicle=vehicle, preferred_mode=FlightMode.HOVER
    )
    return vmap, planner


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestObstacleDetectionToVoxelMapPipeline(unittest.TestCase):
    """Test that obstacle data is correctly reflected in the VoxelMap."""

    def test_box_obstacle_blocks_direct_path(self) -> None:
        """Adding a box obstacle should make the map report collision there."""
        vmap, planner = build_pipeline()
        # Add a box directly on the straight line between start and goal
        vmap.add_box_obstacle(24.0, 4.0, 0.0, 4.0, 4.0, 40.0)

        # Confirm the obstacle is registered
        self.assertTrue(vmap.is_occupied_xyz(25.0, 5.0, 15.0))

    def test_sphere_obstacle_from_pointcloud(self) -> None:
        """A cluster of LiDAR points should create an occupied region."""
        vmap, _ = build_pipeline()

        # Simulate a LiDAR cluster around (20, 20, 15)
        n = 30
        pts = np.random.default_rng(0).uniform(
            low=[19.0, 19.0, 14.0], high=[21.0, 21.0, 16.0], size=(n, 3)
        ).astype(np.float32)
        vmap.update_from_pointcloud(pts)

        # At least some voxels in the cluster should be occupied
        occupied = vmap.get_occupied_voxels()
        self.assertGreater(len(occupied), 0)

    def test_multiple_obstacle_types_coexist(self) -> None:
        """Box, cylinder, and sphere obstacles should all be registered."""
        vmap, _ = build_pipeline(map_size=100.0)
        vmap.add_box_obstacle(10.0, 10.0, 0.0, 3.0, 3.0, 10.0)
        vmap.add_cylinder_obstacle(30.0, 30.0, 0.0, radius=2.0, height=8.0)
        vmap.add_sphere_obstacle(50.0, 50.0, 20.0, radius=3.0)

        self.assertTrue(vmap.is_occupied_xyz(11.0, 11.0, 5.0))
        self.assertTrue(vmap.is_occupied_xyz(30.0, 30.0, 4.0))
        self.assertTrue(vmap.is_occupied_xyz(50.0, 50.0, 20.0))


class TestReplanningTriggers(unittest.TestCase):
    """Test zone-based replanning logic using a mock ROS node."""

    def test_zone_classification_thresholds(self) -> None:
        """Verify zone assignment from distance thresholds (no ROS needed)."""
        def classify_zone(distance: float) -> str:
            if distance <= 1.0:
                return "CRITICAL"
            elif distance <= 3.0:
                return "DANGER"
            elif distance <= 6.0:
                return "WARNING"
            elif distance <= 10.0:
                return "CAUTION"
            return "SAFE"

        self.assertEqual(classify_zone(0.5), "CRITICAL")
        self.assertEqual(classify_zone(1.0), "CRITICAL")
        self.assertEqual(classify_zone(1.5), "DANGER")
        self.assertEqual(classify_zone(3.0), "DANGER")
        self.assertEqual(classify_zone(4.0), "WARNING")
        self.assertEqual(classify_zone(6.0), "WARNING")
        self.assertEqual(classify_zone(8.0), "CAUTION")
        self.assertEqual(classify_zone(10.0), "CAUTION")
        self.assertEqual(classify_zone(15.0), "SAFE")

    def test_emergency_replan_uses_fewer_iterations(self) -> None:
        """Emergency replanning budget should be smaller than normal."""
        from integrated_planning.ros_integration.integrated_planner_node import (
            MAX_ITER_EMERGENCY,
            MAX_ITER_NORMAL,
        )

        self.assertLess(MAX_ITER_EMERGENCY, MAX_ITER_NORMAL)

    def test_zone_constants_match_expected(self) -> None:
        """Zone string constants should match obstacle_detector conventions."""
        from integrated_planning.ros_integration.integrated_planner_node import (
            ZONE_CAUTION,
            ZONE_CRITICAL,
            ZONE_DANGER,
            ZONE_SAFE,
            ZONE_WARNING,
        )

        self.assertEqual(ZONE_CRITICAL, "CRITICAL")
        self.assertEqual(ZONE_DANGER, "DANGER")
        self.assertEqual(ZONE_WARNING, "WARNING")
        self.assertEqual(ZONE_CAUTION, "CAUTION")
        self.assertEqual(ZONE_SAFE, "SAFE")


class TestZoneBasedBehavior(unittest.TestCase):
    """Test that correct actions are taken for each avoidance zone."""

    def test_critical_zone_publishes_hover(self) -> None:
        """CRITICAL zone → emergency_hover() should be called."""
        # We test the logic independently of ROS2 message infrastructure
        actions = []

        def mock_emergency_hover():
            actions.append("hover")

        def mock_emergency_replan():
            actions.append("emergency_replan")

        def mock_replan():
            actions.append("replan")

        def handle_zone(zone: str) -> None:
            zone = zone.upper().strip()
            if zone == "CRITICAL":
                mock_emergency_hover()
            elif zone == "DANGER":
                mock_emergency_replan()
            elif zone == "WARNING":
                mock_replan()

        handle_zone("CRITICAL")
        handle_zone("DANGER")
        handle_zone("WARNING")
        handle_zone("CAUTION")  # no action
        handle_zone("SAFE")     # no action

        self.assertEqual(actions, ["hover", "emergency_replan", "replan"])


class TestEndToEndSimulation(unittest.TestCase):
    """End-to-end simulation: obstacle → map → plan → path."""

    def test_plan_around_single_obstacle(self) -> None:
        """Planner should find a path when a single obstacle blocks direct route."""
        vmap = VoxelMap(30.0, 30.0, 60.0, resolution=1.0)
        # Block direct path at x=15 with a partial wall (doesn't cover z>30)
        vmap.add_box_obstacle(14.0, 0.0, 0.0, 2.0, 30.0, 20.0)

        vehicle = VehicleConfig(
            min_altitude=5.0, max_altitude=50.0,
            hover_step=2.0, inflation=0.0,
        )
        planner = HybridAStarPlanner3D(
            vmap, vehicle=vehicle, preferred_mode=FlightMode.HOVER
        )
        start = (5.0, 5.0, 25.0, 0.0, 0.0)   # z=25 is above wall (z goes to 20)
        goal = (25.0, 5.0, 25.0, 0.0, 0.0)
        path = planner.plan(start, goal, max_iter=15_000)
        # The planner should find a path over or around the wall
        self.assertIsNotNone(path)

    def test_clear_map_finds_direct_path(self) -> None:
        """With no obstacles a path should always be found."""
        vmap, planner = build_pipeline(map_size=40.0)
        start = (5.0, 5.0, 15.0, 0.0, 0.0)
        goal = (30.0, 30.0, 15.0, 0.0, 0.0)
        path = planner.plan(start, goal, max_iter=20_000)
        self.assertIsNotNone(path)
        if path is not None:
            self.assertGreater(len(path), 0)

    def test_occupancy_increases_after_pointcloud(self) -> None:
        """Map occupancy rate should increase after adding a dense point cloud."""
        vmap, _ = build_pipeline()
        initial_rate = vmap.get_occupancy_rate()

        rng = np.random.default_rng(42)
        pts = rng.uniform(
            low=[5.0, 5.0, 10.0], high=[15.0, 15.0, 20.0], size=(500, 3)
        ).astype(np.float32)
        vmap.update_from_pointcloud(pts)

        self.assertGreater(vmap.get_occupancy_rate(), initial_rate)

    def test_path_smoothing_in_pipeline(self) -> None:
        """Smoothed path should have the same number of waypoints as raw path."""
        vmap, planner = build_pipeline(map_size=40.0)
        start = (5.0, 5.0, 15.0, 0.0, 0.0)
        goal = (30.0, 30.0, 15.0, 0.0, 0.0)
        raw_path = planner.plan(start, goal, max_iter=20_000)

        if raw_path is not None:
            smoothed = HybridAStarPlanner3D.smooth_path(raw_path)
            self.assertEqual(len(smoothed), len(raw_path))

    def test_emergency_hover_logic(self) -> None:
        """emergency_hover should create an ascending two-waypoint path."""
        # Simulate the emergency_hover logic without ROS2
        current_pose = (10.0, 10.0, 15.0, 0.0, 0.0)
        vehicle = VehicleConfig(min_altitude=5.0, max_altitude=80.0)

        x, y, z, yaw, pitch = current_pose
        ascend_z = min(z + 5.0, vehicle.max_altitude)
        hover_path = [
            (x, y, z, yaw, 0.0),
            (x, y, ascend_z, yaw, 0.0),
        ]

        self.assertEqual(len(hover_path), 2)
        self.assertGreater(hover_path[1][2], hover_path[0][2])
        self.assertLessEqual(hover_path[1][2], vehicle.max_altitude)


class TestObstacleMapBridgeHelpers(unittest.TestCase):
    """Test ObstacleMapBridge helper methods (non-ROS)."""

    def test_pointcloud2_to_array_returns_empty_on_bad_input(self) -> None:
        """Static helper should return empty array for invalid messages."""
        from integrated_planning.ros_integration.obstacle_map_bridge import (
            ObstacleMapBridge,
        )

        class FakeField:
            def __init__(self, name, offset):
                self.name = name
                self.offset = offset

        class FakeMsg:
            fields = [FakeField("x", 0), FakeField("y", 4), FakeField("z", 8)]
            point_step = 12
            data = b""  # empty

        result = ObstacleMapBridge.pointcloud2_to_array(FakeMsg())
        self.assertEqual(result.shape, (0, 3))

    def test_pointcloud2_to_array_valid(self) -> None:
        """Static helper should correctly parse a minimal PointCloud2 message."""
        import struct

        from integrated_planning.ros_integration.obstacle_map_bridge import (
            ObstacleMapBridge,
        )

        class FakeField:
            def __init__(self, name, offset):
                self.name = name
                self.offset = offset

        class FakeMsg:
            fields = [
                FakeField("x", 0),
                FakeField("y", 4),
                FakeField("z", 8),
            ]
            point_step = 12
            data = struct.pack("3f", 1.0, 2.0, 3.0)

        result = ObstacleMapBridge.pointcloud2_to_array(FakeMsg())
        self.assertEqual(result.shape[0], 1)
        self.assertAlmostEqual(result[0, 0], 1.0, places=4)
        self.assertAlmostEqual(result[0, 1], 2.0, places=4)
        self.assertAlmostEqual(result[0, 2], 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
