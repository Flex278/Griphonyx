#!/usr/bin/env python3
"""
Unit tests for the A* pathfinding core (astar_algorithm.py).

Covers the result container, input validation, trivial and open-grid paths,
4- vs 8-connected movement, obstacle avoidance, unknown-cell handling, cost
penalties, the iteration cap, and path-validity invariants.
"""

import os
import sys
import unittest

import numpy as np

# The `astar_planner` Python package lives one directory up from this file
# (repo layout: astar_planner/astar_planner/astar_algorithm.py), so add the
# parent directory to sys.path to make `import astar_planner` work without
# a ROS 2 workspace installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astar_planner.astar_algorithm import AStarResult, astar_search


def make_grid(shape, value=0):
    """Return an int8 occupancy grid of ``shape`` filled with ``value``."""
    return np.full(shape, value, dtype=np.int8)


# ---------------------------------------------------------------------------
# AStarResult
# ---------------------------------------------------------------------------
class TestAStarResult(unittest.TestCase):
    """Test the AStarResult container."""

    def test_default_construction(self) -> None:
        r = AStarResult()
        self.assertEqual(r.path, [])
        self.assertEqual(r.cost, 0.0)
        self.assertEqual(r.nodes_expanded, 0)
        self.assertEqual(r.elapsed_ms, 0.0)
        self.assertFalse(r.success)

    def test_custom_construction(self) -> None:
        r = AStarResult(path=[(0, 0), (1, 1)], cost=2.5,
                        nodes_expanded=7, elapsed_ms=1.5, success=True)
        self.assertEqual(r.path, [(0, 0), (1, 1)])
        self.assertEqual(r.cost, 2.5)
        self.assertEqual(r.nodes_expanded, 7)
        self.assertEqual(r.elapsed_ms, 1.5)
        self.assertTrue(r.success)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation(unittest.TestCase):
    """Test rejection of invalid start/goal positions."""

    def setUp(self) -> None:
        self.grid = make_grid((5, 5))

    def test_start_out_of_bounds(self) -> None:
        for start in ((-1, 0), (5, 0), (0, -1), (0, 5)):
            r = astar_search(self.grid, start, (0, 0))
            self.assertFalse(r.success)

    def test_goal_out_of_bounds(self) -> None:
        for goal in ((-1, 0), (5, 0), (0, -1), (0, 5)):
            r = astar_search(self.grid, (0, 0), goal)
            self.assertFalse(r.success)

    def test_lethal_start(self) -> None:
        grid = make_grid((5, 5))
        grid[0, 0] = 100
        r = astar_search(grid, (0, 0), (4, 4))
        self.assertFalse(r.success)

    def test_lethal_goal(self) -> None:
        grid = make_grid((5, 5))
        grid[4, 4] = 100
        r = astar_search(grid, (0, 0), (4, 4))
        self.assertFalse(r.success)


# ---------------------------------------------------------------------------
# Basic pathfinding
# ---------------------------------------------------------------------------
class TestBasicPathfinding(unittest.TestCase):
    """Test that valid paths are found in open space."""

    def test_trivial_start_equals_goal(self) -> None:
        grid = make_grid((5, 5))
        r = astar_search(grid, (2, 2), (2, 2))
        self.assertTrue(r.success)
        self.assertEqual(r.path, [(2, 2)])
        self.assertEqual(r.cost, 0.0)
        self.assertEqual(r.nodes_expanded, 1)

    def test_straight_line_diagonal(self) -> None:
        grid = make_grid((5, 5))
        r = astar_search(grid, (0, 0), (4, 4))
        self.assertTrue(r.success)
        self.assertEqual(r.path[0], (0, 0))
        self.assertEqual(r.path[-1], (4, 4))
        self.assertEqual(len(r.path), 5)
        self.assertAlmostEqual(r.cost, 4 * 1.414, places=2)

    def test_straight_line_cardinal(self) -> None:
        grid = make_grid((5, 5))
        r = astar_search(grid, (0, 0), (0, 4), diagonal=False)
        self.assertTrue(r.success)
        self.assertEqual(r.path[0], (0, 0))
        self.assertEqual(r.path[-1], (0, 4))
        self.assertEqual(len(r.path), 5)
        self.assertAlmostEqual(r.cost, 4.0, places=5)

    def test_elapsed_and_expanded_populated(self) -> None:
        grid = make_grid((5, 5))
        r = astar_search(grid, (0, 0), (4, 4))
        self.assertTrue(r.success)
        self.assertGreaterEqual(r.elapsed_ms, 0.0)
        self.assertGreaterEqual(r.nodes_expanded, 1)


# ---------------------------------------------------------------------------
# Movement modes (4- vs 8-connected)
# ---------------------------------------------------------------------------
class TestMovementModes(unittest.TestCase):
    """Test diagonal vs cardinal-only movement."""

    def test_diagonal_allowed_shortcuts(self) -> None:
        grid = make_grid((3, 3))
        r = astar_search(grid, (0, 0), (2, 2), diagonal=True)
        self.assertTrue(r.success)
        self.assertEqual(len(r.path), 3)
        self.assertAlmostEqual(r.cost, 2 * 1.414, places=2)

    def test_no_diagonal_forces_longer_path(self) -> None:
        grid = make_grid((3, 3))
        r = astar_search(grid, (0, 0), (2, 2), diagonal=False)
        self.assertTrue(r.success)
        self.assertEqual(len(r.path), 5)
        self.assertAlmostEqual(r.cost, 4.0, places=5)

    def test_no_diagonal_uses_only_cardinal_steps(self) -> None:
        grid = make_grid((5, 5))
        r = astar_search(grid, (0, 0), (4, 4), diagonal=False)
        self.assertTrue(r.success)
        for a, b in zip(r.path, r.path[1:]):
            dr = abs(a[0] - b[0])
            dc = abs(a[1] - b[1])
            self.assertTrue((dr == 1 and dc == 0) or (dr == 0 and dc == 1))


# ---------------------------------------------------------------------------
# Obstacle avoidance
# ---------------------------------------------------------------------------
class TestObstacleAvoidance(unittest.TestCase):
    """Test that lethal cells are avoided."""

    def test_path_avoids_lethal_cells(self) -> None:
        grid = make_grid((5, 5))
        grid[0:4, 2] = 100  # vertical wall, open only at row 4
        r = astar_search(grid, (2, 0), (2, 4))
        self.assertTrue(r.success)
        self.assertEqual(r.path[0], (2, 0))
        self.assertEqual(r.path[-1], (2, 4))
        for cell in r.path:
            self.assertLess(grid[cell[0], cell[1]], 90)

    def test_no_path_when_enclosed(self) -> None:
        grid = make_grid((5, 5))
        grid[0:3, 0:3] = 100  # wall off the start
        grid[1, 1] = 0       # start is the only free cell in the corner
        r = astar_search(grid, (1, 1), (4, 4))
        self.assertFalse(r.success)


# ---------------------------------------------------------------------------
# Unknown cells and cost penalties
# ---------------------------------------------------------------------------
class TestUnknownAndCost(unittest.TestCase):
    """Test unknown-cell handling and cost penalties."""

    def test_unknown_cells_are_traversable(self) -> None:
        grid = make_grid((3, 3))
        grid[0, 1] = -1  # unknown
        r = astar_search(grid, (0, 0), (0, 2))
        self.assertTrue(r.success)
        self.assertEqual(r.path[-1], (0, 2))

    def test_high_cost_cell_raises_path_cost(self) -> None:
        free = make_grid((1, 3))
        r_free = astar_search(free, (0, 0), (0, 2))
        self.assertAlmostEqual(r_free.cost, 2.0, places=5)

        costly = make_grid((1, 3))
        costly[0, 1] = 50
        r_costly = astar_search(costly, (0, 0), (0, 2))
        self.assertGreater(r_costly.cost, r_free.cost)

    def test_prefers_free_detour_over_high_cost_cell(self) -> None:
        grid = make_grid((3, 3))
        grid[1, 1] = 80  # expensive but passable straight-line cell
        r = astar_search(grid, (0, 1), (2, 1))
        self.assertTrue(r.success)
        self.assertNotIn((1, 1), r.path)


# ---------------------------------------------------------------------------
# Iteration cap
# ---------------------------------------------------------------------------
class TestMaxIterations(unittest.TestCase):
    """Test the hard cap on node expansions."""

    def test_max_iterations_caps_search(self) -> None:
        grid = make_grid((20, 20))
        r = astar_search(grid, (0, 0), (19, 19), max_iterations=1)
        self.assertFalse(r.success)


# ---------------------------------------------------------------------------
# Path validity invariants
# ---------------------------------------------------------------------------
class TestPathInvariants(unittest.TestCase):
    """Test that returned paths are structurally valid."""

    def test_path_is_contiguous_8_connected(self) -> None:
        grid = make_grid((10, 10))
        r = astar_search(grid, (0, 0), (9, 9), diagonal=True)
        self.assertTrue(r.success)
        for a, b in zip(r.path, r.path[1:]):
            self.assertLessEqual(abs(a[0] - b[0]), 1)
            self.assertLessEqual(abs(a[1] - b[1]), 1)
            self.assertNotEqual(a, b)

    def test_path_cells_within_bounds_and_free(self) -> None:
        grid = make_grid((10, 10))
        grid[3, 5] = 100
        grid[4, 5] = 100
        r = astar_search(grid, (0, 0), (9, 9))
        self.assertTrue(r.success)
        h, w = grid.shape
        for cell in r.path:
            self.assertTrue(0 <= cell[0] < h)
            self.assertTrue(0 <= cell[1] < w)
            self.assertLess(grid[cell[0], cell[1]], 90)


# ---------------------------------------------------------------------------
# Lethal threshold
# ---------------------------------------------------------------------------
class TestLethalThreshold(unittest.TestCase):
    """Test the lethal_threshold boundary and custom values."""

    def test_default_threshold_treats_90_as_lethal(self) -> None:
        grid = make_grid((1, 3))
        grid[0, 1] = 90
        r = astar_search(grid, (0, 0), (0, 2))
        self.assertFalse(r.success)

    def test_default_threshold_treats_89_as_traversable(self) -> None:
        grid = make_grid((1, 3))
        grid[0, 1] = 89
        r = astar_search(grid, (0, 0), (0, 2))
        self.assertTrue(r.success)

    def test_custom_threshold_blocks_50(self) -> None:
        grid = make_grid((1, 3))
        grid[0, 1] = 50
        r = astar_search(grid, (0, 0), (0, 2), lethal_threshold=50)
        self.assertFalse(r.success)


# ---------------------------------------------------------------------------
# Cost penalty factor
# ---------------------------------------------------------------------------
class TestCostPenaltyFactor(unittest.TestCase):
    """Test that cost_penalty_factor scales the cost of high-cost cells."""

    def test_higher_factor_raises_cost(self) -> None:
        costs = []
        for factor in (0.0, 2.0, 10.0):
            grid = make_grid((1, 3))
            grid[0, 1] = 50
            r = astar_search(grid, (0, 0), (0, 2),
                             cost_penalty_factor=factor)
            self.assertTrue(r.success)
            costs.append(r.cost)
        self.assertLess(costs[0], costs[1])
        self.assertLess(costs[1], costs[2])


# ---------------------------------------------------------------------------
# Heuristic weight (weighted A*)
# ---------------------------------------------------------------------------
class TestHeuristicWeight(unittest.TestCase):
    """Test weighted A* (heuristic_weight > 1) behaviour."""

    def _grid(self):
        grid = make_grid((8, 8))
        grid[2:6, 3] = 100  # a wall forcing a detour
        return grid

    def test_weighted_search_finds_valid_path(self) -> None:
        grid = self._grid()
        r = astar_search(grid, (0, 0), (7, 7), heuristic_weight=3.0)
        self.assertTrue(r.success)
        self.assertEqual(r.path[0], (0, 0))
        self.assertEqual(r.path[-1], (7, 7))
        for cell in r.path:
            self.assertLess(grid[cell[0], cell[1]], 90)

    def test_weighted_search_not_cheaper_than_optimal(self) -> None:
        grid = self._grid()
        optimal = astar_search(grid, (0, 0), (7, 7), heuristic_weight=1.0)
        weighted = astar_search(grid, (0, 0), (7, 7), heuristic_weight=3.0)
        self.assertTrue(optimal.success)
        self.assertTrue(weighted.success)
        self.assertGreaterEqual(weighted.cost, optimal.cost - 1e-6)


if __name__ == "__main__":
    unittest.main()
