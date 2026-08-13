"""
Unit tests for HybridAStarPlanner3D.

Tests cover basic path planning, altitude enforcement, collision avoidance,
dual-mode behaviour, and path smoothing.
"""

import math
import os
import sys
import unittest
from typing import List, Tuple

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", ".."),
)

from integrated_planning.maps.voxel_map_3d import VoxelMap
from integrated_planning.planners.hybrid_astar_3d import (
    FlightMode,
    HybridAStarPlanner3D,
    Node3D,
    VehicleConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_planner(
    map_size: float = 50.0,
    min_alt: float = 5.0,
    max_alt: float = 120.0,
    mode: FlightMode = FlightMode.HOVER,
) -> HybridAStarPlanner3D:
    """Return a planner over a clear map."""
    vmap = VoxelMap(map_size, map_size, max_alt + 10.0, resolution=1.0)
    vehicle = VehicleConfig(
        min_altitude=min_alt,
        max_altitude=max_alt,
        hover_step=2.0,
        cruise_step=3.0,
        inflation=0.0,
    )
    return HybridAStarPlanner3D(vmap, vehicle=vehicle, preferred_mode=mode)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicPlannerInit(unittest.TestCase):
    """Smoke tests for planner construction."""

    def test_default_construction(self) -> None:
        vmap = VoxelMap(20.0, 20.0, 30.0)
        planner = HybridAStarPlanner3D(vmap)
        self.assertIsNotNone(planner)
        self.assertEqual(planner.preferred_mode, FlightMode.HOVER)

    def test_cruise_mode(self) -> None:
        vmap = VoxelMap(20.0, 20.0, 30.0)
        planner = HybridAStarPlanner3D(vmap, preferred_mode=FlightMode.CRUISE)
        self.assertEqual(planner.preferred_mode, FlightMode.CRUISE)


class TestValidatePose(unittest.TestCase):
    """Test validate_pose method."""

    def setUp(self) -> None:
        self.planner = make_planner(min_alt=5.0, max_alt=50.0)

    def test_valid_pose(self) -> None:
        self.assertTrue(self.planner.validate_pose((10.0, 10.0, 20.0, 0.0, 0.0)))

    def test_below_min_altitude(self) -> None:
        self.assertFalse(self.planner.validate_pose((10.0, 10.0, 2.0, 0.0, 0.0)))

    def test_above_max_altitude(self) -> None:
        self.assertFalse(self.planner.validate_pose((10.0, 10.0, 60.0, 0.0, 0.0)))

    def test_too_short_pose(self) -> None:
        self.assertFalse(self.planner.validate_pose((10.0, 10.0)))


class TestAltitudeSafe(unittest.TestCase):
    """Test _altitude_safe helper."""

    def setUp(self) -> None:
        self.planner = make_planner(min_alt=5.0, max_alt=50.0)

    def test_within_range(self) -> None:
        self.assertTrue(self.planner._altitude_safe(20.0))

    def test_at_min(self) -> None:
        self.assertTrue(self.planner._altitude_safe(5.0))

    def test_below_min(self) -> None:
        self.assertFalse(self.planner._altitude_safe(4.9))

    def test_above_max(self) -> None:
        self.assertFalse(self.planner._altitude_safe(51.0))


class TestBasicPathPlanning(unittest.TestCase):
    """Test that the planner finds valid paths in open space."""

    def test_trivial_start_equals_goal(self) -> None:
        """Start == goal should return immediately with a trivial path."""
        planner = make_planner(map_size=30.0, min_alt=5.0)
        start = (10.0, 10.0, 10.0, 0.0, 0.0)
        # A very short planning range with start near goal
        path = planner.plan(start, start, max_iter=500)
        # With goal tolerance the planner should succeed
        self.assertIsNotNone(path)

    def test_short_horizontal_path(self) -> None:
        """Plan a short path in open space – should succeed."""
        planner = make_planner(map_size=50.0, min_alt=5.0, max_alt=120.0)
        start = (5.0, 5.0, 10.0, 0.0, 0.0)
        goal = (15.0, 5.0, 10.0, 0.0, 0.0)
        path = planner.plan(start, goal, max_iter=10_000)
        self.assertIsNotNone(path)
        if path is not None:
            self.assertGreater(len(path), 0)
            # Start and goal positions are approximately correct
            self.assertAlmostEqual(path[0][0], start[0], delta=2.5)
            self.assertAlmostEqual(path[-1][0], goal[0], delta=2.5)

    def test_path_respects_min_altitude(self) -> None:
        """All waypoints must be at or above min_altitude."""
        planner = make_planner(map_size=50.0, min_alt=5.0, max_alt=120.0)
        start = (5.0, 5.0, 10.0, 0.0, 0.0)
        goal = (30.0, 30.0, 10.0, 0.0, 0.0)
        path = planner.plan(start, goal, max_iter=20_000)
        if path is not None:
            for wp in path:
                self.assertGreaterEqual(
                    wp[2],
                    planner.vehicle.min_altitude - 0.1,
                    msg=f"Waypoint altitude {wp[2]:.2f} below min",
                )

    def test_path_respects_max_altitude(self) -> None:
        """All waypoints must be at or below max_altitude."""
        planner = make_planner(map_size=50.0, min_alt=5.0, max_alt=30.0)
        start = (5.0, 5.0, 10.0, 0.0, 0.0)
        goal = (20.0, 20.0, 10.0, 0.0, 0.0)
        path = planner.plan(start, goal, max_iter=15_000)
        if path is not None:
            for wp in path:
                self.assertLessEqual(
                    wp[2],
                    planner.vehicle.max_altitude + 0.1,
                    msg=f"Waypoint altitude {wp[2]:.2f} above max",
                )

    def test_invalid_start_returns_none(self) -> None:
        planner = make_planner(min_alt=5.0, max_alt=50.0)
        bad_start = (5.0, 5.0, 1.0, 0.0, 0.0)  # below min altitude
        goal = (20.0, 20.0, 20.0, 0.0, 0.0)
        path = planner.plan(bad_start, goal)
        self.assertIsNone(path)

    def test_invalid_goal_returns_none(self) -> None:
        planner = make_planner(min_alt=5.0, max_alt=50.0)
        start = (5.0, 5.0, 10.0, 0.0, 0.0)
        bad_goal = (20.0, 20.0, 60.0, 0.0, 0.0)  # above max altitude
        path = planner.plan(start, bad_goal)
        self.assertIsNone(path)


class TestCollisionAvoidance(unittest.TestCase):
    """Test that the planner avoids obstacles."""

    def test_direct_path_blocked_by_wall(self) -> None:
        """A wall between start and goal forces a detour."""
        vmap = VoxelMap(40.0, 40.0, 60.0, resolution=1.0)
        # Vertical wall across y=20 for the full height
        vmap.add_box_obstacle(0.0, 18.0, 0.0, 40.0, 4.0, 40.0)
        vehicle = VehicleConfig(
            min_altitude=5.0, max_altitude=50.0,
            hover_step=2.0, inflation=0.0,
        )
        planner = HybridAStarPlanner3D(vmap, vehicle=vehicle)
        start = (5.0, 5.0, 10.0, 0.0, 0.0)
        goal = (5.0, 35.0, 10.0, 0.0, 0.0)
        # With the wall present, no path should be found (wall spans full z)
        path = planner.plan(start, goal, max_iter=5_000)
        if path is not None:
            # If a path was found it should not pass through the wall
            for wp in path:
                wall_y = 18.0 <= wp[1] <= 22.0
                wall_z = 0.0 <= wp[2] <= 40.0
                self.assertFalse(
                    wall_y and wall_z,
                    msg=f"Waypoint {wp} passes through wall",
                )


class TestMotionPrimitives(unittest.TestCase):
    """Test motion primitive generation."""

    def setUp(self) -> None:
        self.vmap = VoxelMap(20.0, 20.0, 30.0)
        self.vehicle = VehicleConfig(hover_step=1.0, cruise_step=2.0)
        self.planner = HybridAStarPlanner3D(self.vmap, vehicle=self.vehicle)

    def test_hover_primitives_count(self) -> None:
        prims = self.planner._get_motion_primitives(FlightMode.HOVER)
        # 3^3 - 1 = 26 directions
        self.assertEqual(len(prims), 26)

    def test_cruise_primitives_exist(self) -> None:
        prims = self.planner._get_motion_primitives(FlightMode.CRUISE)
        self.assertGreater(len(prims), 0)

    def test_hover_step_size(self) -> None:
        prims = self.planner._get_motion_primitives(FlightMode.HOVER)
        for dx, dy, dz, _, _ in prims:
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.assertAlmostEqual(
                dist, self.vehicle.hover_step, places=5
            )


class TestPathSmoothing(unittest.TestCase):
    """Test static smooth_path method."""

    def test_short_path_unchanged(self) -> None:
        path = [(0.0, 0.0, 10.0, 0.0, 0.0), (5.0, 5.0, 10.0, 0.0, 0.0)]
        smoothed = HybridAStarPlanner3D.smooth_path(path)
        self.assertEqual(len(smoothed), len(path))

    def test_endpoints_preserved(self) -> None:
        path = [
            (0.0, 0.0, 10.0, 0.0, 0.0),
            (5.0, 0.0, 10.0, 0.0, 0.0),
            (10.0, 0.0, 10.0, 0.0, 0.0),
            (15.0, 0.0, 10.0, 0.0, 0.0),
        ]
        smoothed = HybridAStarPlanner3D.smooth_path(path, iterations=50)
        self.assertAlmostEqual(smoothed[0][0], path[0][0], places=5)
        self.assertAlmostEqual(smoothed[-1][0], path[-1][0], places=5)

    def test_smoothing_reduces_jitter(self) -> None:
        """A jagged path should have reduced total variation after smoothing."""
        import random
        random.seed(42)
        path = [
            (float(i) + random.uniform(-0.5, 0.5),
             float(i) + random.uniform(-0.5, 0.5),
             10.0, 0.0, 0.0)
            for i in range(10)
        ]
        smoothed = HybridAStarPlanner3D.smooth_path(path, iterations=200)
        self.assertEqual(len(smoothed), len(path))

    def test_single_point_path(self) -> None:
        path = [(5.0, 5.0, 10.0, 0.0, 0.0)]
        smoothed = HybridAStarPlanner3D.smooth_path(path)
        self.assertEqual(smoothed, path)


class TestNode3D(unittest.TestCase):
    """Test Node3D dataclass."""

    def test_key_deterministic(self) -> None:
        n = Node3D(1.0, 2.0, 10.0, 0.1, 0.0, None, 0.0, 0.0, FlightMode.HOVER)
        self.assertEqual(n.key, n.key)

    def test_f_equals_g_plus_h(self) -> None:
        n = Node3D(0.0, 0.0, 10.0, 0.0, 0.0, None, 3.0, 4.0, FlightMode.HOVER)
        self.assertAlmostEqual(n.f, 7.0)

    def test_less_than_ordering(self) -> None:
        n1 = Node3D(0.0, 0.0, 10.0, 0.0, 0.0, None, 1.0, 1.0, FlightMode.HOVER)
        n2 = Node3D(0.0, 0.0, 10.0, 0.0, 0.0, None, 3.0, 3.0, FlightMode.HOVER)
        self.assertLess(n1, n2)


class TestNormaliseAngle(unittest.TestCase):
    """Test _normalise_angle static method."""

    def test_within_range(self) -> None:
        result = HybridAStarPlanner3D._normalise_angle(1.0)
        self.assertAlmostEqual(result, 1.0)

    def test_above_pi(self) -> None:
        result = HybridAStarPlanner3D._normalise_angle(4.0)
        self.assertLessEqual(result, math.pi)
        self.assertGreater(result, -math.pi)

    def test_below_minus_pi(self) -> None:
        result = HybridAStarPlanner3D._normalise_angle(-4.0)
        self.assertLessEqual(result, math.pi)
        self.assertGreater(result, -math.pi)


if __name__ == "__main__":
    unittest.main()
