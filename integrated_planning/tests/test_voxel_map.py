"""
Unit tests for VoxelMap.

Tests cover obstacle addition, collision detection, coordinate conversion,
point-cloud integration, save/load round-trip, and utility methods.
"""

import math
import os
import tempfile
import unittest

import numpy as np

# Adjust sys.path so tests can run from the repo root without installation
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", ".."),
)

from integrated_planning.maps.voxel_map_3d import VoxelMap
from integrated_planning.planners.hybrid_astar_3d import VehicleConfig


class TestVoxelMapInit(unittest.TestCase):
    """Test VoxelMap construction and basic properties."""

    def test_basic_construction(self) -> None:
        vmap = VoxelMap(10.0, 20.0, 5.0, resolution=1.0)
        self.assertEqual(vmap.nx, 10)
        self.assertEqual(vmap.ny, 20)
        self.assertEqual(vmap.nz, 5)

    def test_non_integer_size(self) -> None:
        """Dimensions that are not integer multiples of resolution."""
        vmap = VoxelMap(10.5, 10.5, 5.5, resolution=1.0)
        self.assertEqual(vmap.nx, 11)
        self.assertEqual(vmap.ny, 11)
        self.assertEqual(vmap.nz, 6)

    def test_invalid_dimensions_raise(self) -> None:
        with self.assertRaises(ValueError):
            VoxelMap(-1.0, 10.0, 5.0)
        with self.assertRaises(ValueError):
            VoxelMap(10.0, 0.0, 5.0)
        with self.assertRaises(ValueError):
            VoxelMap(10.0, 10.0, 5.0, resolution=0.0)

    def test_initial_map_is_free(self) -> None:
        vmap = VoxelMap(5.0, 5.0, 5.0)
        self.assertEqual(vmap.get_occupancy_rate(), 0.0)


class TestCoordinateConversion(unittest.TestCase):
    """Test world ↔ voxel coordinate conversion."""

    def setUp(self) -> None:
        self.vmap = VoxelMap(10.0, 10.0, 10.0, resolution=1.0)

    def test_world_to_voxel_origin(self) -> None:
        ix, iy, iz = self.vmap.world_to_voxel(0.0, 0.0, 0.0)
        self.assertEqual((ix, iy, iz), (0, 0, 0))

    def test_world_to_voxel_middle(self) -> None:
        ix, iy, iz = self.vmap.world_to_voxel(5.5, 3.2, 7.9)
        self.assertEqual(ix, 5)
        self.assertEqual(iy, 3)
        self.assertEqual(iz, 7)

    def test_voxel_to_world_centre(self) -> None:
        x, y, z = self.vmap.voxel_to_world(0, 0, 0)
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)
        self.assertAlmostEqual(z, 0.5)

    def test_round_trip(self) -> None:
        for wx in (0.7, 3.1, 8.9):
            ix, iy, iz = self.vmap.world_to_voxel(wx, wx, wx)
            cx, cy, cz = self.vmap.voxel_to_world(ix, iy, iz)
            # World centre of voxel should be within one voxel
            self.assertAlmostEqual(cx, ix + 0.5)


class TestBoxObstacle(unittest.TestCase):
    """Test box obstacle addition and querying."""

    def setUp(self) -> None:
        self.vmap = VoxelMap(20.0, 20.0, 20.0, resolution=1.0)

    def test_box_marks_correct_voxels(self) -> None:
        self.vmap.add_box_obstacle(5.0, 5.0, 0.0, 3.0, 3.0, 3.0)
        # Point inside box
        self.assertTrue(self.vmap.is_occupied_xyz(6.0, 6.0, 1.0))

    def test_outside_box_is_free(self) -> None:
        self.vmap.add_box_obstacle(5.0, 5.0, 0.0, 3.0, 3.0, 3.0)
        self.assertFalse(self.vmap.is_occupied_xyz(1.0, 1.0, 1.0))

    def test_out_of_bounds_is_occupied(self) -> None:
        """Out-of-bounds positions should be treated as occupied."""
        self.assertTrue(self.vmap.is_occupied_xyz(-1.0, 0.0, 0.0))
        self.assertTrue(self.vmap.is_occupied_xyz(0.0, 100.0, 0.0))

    def test_clear_map(self) -> None:
        self.vmap.add_box_obstacle(5.0, 5.0, 0.0, 3.0, 3.0, 3.0)
        self.vmap.clear_map()
        self.assertFalse(self.vmap.is_occupied_xyz(6.0, 6.0, 1.0))
        self.assertEqual(self.vmap.get_occupancy_rate(), 0.0)


class TestCylinderObstacle(unittest.TestCase):
    """Test cylinder obstacle addition."""

    def setUp(self) -> None:
        self.vmap = VoxelMap(30.0, 30.0, 30.0, resolution=1.0)

    def test_centre_is_occupied(self) -> None:
        self.vmap.add_cylinder_obstacle(15.0, 15.0, 0.0, radius=3.0, height=10.0)
        self.assertTrue(self.vmap.is_occupied_xyz(15.0, 15.0, 5.0))

    def test_outside_radius_is_free(self) -> None:
        self.vmap.add_cylinder_obstacle(15.0, 15.0, 0.0, radius=3.0, height=10.0)
        self.assertFalse(self.vmap.is_occupied_xyz(20.0, 20.0, 5.0))

    def test_above_height_is_free(self) -> None:
        self.vmap.add_cylinder_obstacle(15.0, 15.0, 0.0, radius=3.0, height=5.0)
        self.assertFalse(self.vmap.is_occupied_xyz(15.0, 15.0, 8.0))


class TestSphereObstacle(unittest.TestCase):
    """Test sphere obstacle addition."""

    def setUp(self) -> None:
        self.vmap = VoxelMap(30.0, 30.0, 30.0, resolution=1.0)

    def test_centre_is_occupied(self) -> None:
        self.vmap.add_sphere_obstacle(15.0, 15.0, 15.0, radius=4.0)
        self.assertTrue(self.vmap.is_occupied_xyz(15.0, 15.0, 15.0))

    def test_outside_radius_is_free(self) -> None:
        self.vmap.add_sphere_obstacle(15.0, 15.0, 15.0, radius=2.0)
        self.assertFalse(self.vmap.is_occupied_xyz(15.0, 15.0, 22.0))

    def test_surface_voxels(self) -> None:
        """Voxels on the sphere surface should be marked occupied."""
        self.vmap.add_sphere_obstacle(15.0, 15.0, 15.0, radius=3.0)
        # A point clearly inside the sphere
        self.assertTrue(self.vmap.is_occupied_xyz(15.5, 15.5, 15.5))


class TestPointCloudIntegration(unittest.TestCase):
    """Test VoxelMap.update_from_pointcloud."""

    def setUp(self) -> None:
        self.vmap = VoxelMap(20.0, 20.0, 20.0, resolution=1.0)

    def test_single_point(self) -> None:
        pts = np.array([[5.5, 5.5, 5.5]], dtype=np.float32)
        self.vmap.update_from_pointcloud(pts)
        self.assertTrue(self.vmap.is_occupied_xyz(5.5, 5.5, 5.5))

    def test_multiple_points(self) -> None:
        pts = np.array(
            [[1.0, 1.0, 1.0], [10.0, 10.0, 10.0], [15.0, 15.0, 15.0]],
            dtype=np.float32,
        )
        self.vmap.update_from_pointcloud(pts)
        self.assertTrue(self.vmap.is_occupied_xyz(1.5, 1.5, 1.5))
        self.assertTrue(self.vmap.is_occupied_xyz(10.5, 10.5, 10.5))

    def test_out_of_bounds_points_ignored(self) -> None:
        pts = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
        self.vmap.update_from_pointcloud(pts)
        # Should not crash and map should still be empty
        self.assertEqual(self.vmap.get_occupancy_rate(), 0.0)

    def test_empty_pointcloud(self) -> None:
        pts = np.empty((0, 3), dtype=np.float32)
        self.vmap.update_from_pointcloud(pts)
        self.assertEqual(self.vmap.get_occupancy_rate(), 0.0)

    def test_nan_points_filtered(self) -> None:
        pts = np.array(
            [[float("nan"), 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=np.float32
        )
        self.vmap.update_from_pointcloud(pts)
        self.assertTrue(self.vmap.is_occupied_xyz(5.5, 5.5, 5.5))


class TestAABBCollision(unittest.TestCase):
    """Test axis-aligned bounding box collision detection."""

    def setUp(self) -> None:
        self.vmap = VoxelMap(30.0, 30.0, 30.0, resolution=1.0)
        self.vehicle = VehicleConfig(
            length=2.0, width=1.5, height=0.5, inflation=0.0
        )

    def test_no_obstacle_no_collision(self) -> None:
        result = self.vmap.aabb_collision(
            15.0, 15.0, 15.0, 0.0, 0.0, self.vehicle
        )
        self.assertFalse(result)

    def test_obstacle_causes_collision(self) -> None:
        self.vmap.add_box_obstacle(14.0, 14.0, 14.0, 3.0, 3.0, 3.0)
        result = self.vmap.aabb_collision(
            15.0, 15.0, 15.0, 0.0, 0.0, self.vehicle
        )
        self.assertTrue(result)

    def test_distant_obstacle_no_collision(self) -> None:
        self.vmap.add_box_obstacle(25.0, 25.0, 25.0, 1.0, 1.0, 1.0)
        result = self.vmap.aabb_collision(
            5.0, 5.0, 5.0, 0.0, 0.0, self.vehicle
        )
        self.assertFalse(result)


class TestGetOccupiedVoxels(unittest.TestCase):
    """Test get_occupied_voxels utility method."""

    def test_empty_map_returns_empty(self) -> None:
        vmap = VoxelMap(5.0, 5.0, 5.0)
        self.assertEqual(vmap.get_occupied_voxels(), [])

    def test_returns_correct_count(self) -> None:
        vmap = VoxelMap(10.0, 10.0, 10.0, resolution=1.0)
        vmap.add_box_obstacle(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)  # 2x2x2 = 8
        occupied = vmap.get_occupied_voxels()
        self.assertEqual(len(occupied), 8)


class TestSaveLoad(unittest.TestCase):
    """Test VoxelMap persistence (save/load round-trip)."""

    def test_round_trip(self) -> None:
        vmap = VoxelMap(10.0, 10.0, 10.0, resolution=1.0)
        vmap.add_box_obstacle(2.0, 2.0, 2.0, 3.0, 3.0, 3.0)
        rate_before = vmap.get_occupancy_rate()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_map")
            vmap.save_map(path)
            loaded = VoxelMap.load_map(path + ".npz")

        self.assertAlmostEqual(loaded.get_occupancy_rate(), rate_before)
        self.assertEqual(loaded.nx, vmap.nx)
        self.assertEqual(loaded.ny, vmap.ny)
        self.assertEqual(loaded.nz, vmap.nz)


if __name__ == "__main__":
    unittest.main()
