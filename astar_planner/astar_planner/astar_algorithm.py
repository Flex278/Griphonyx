#!/usr/bin/env python3
"""
A* pathfinding algorithm for 2D occupancy grids.
"""

import heapq
import time

import numpy as np


class AStarResult:
    """Container for A* search results."""

    __slots__ = ('path', 'cost', 'nodes_expanded', 'elapsed_ms', 'success')

    def __init__(self, path=None, cost=0.0, nodes_expanded=0,
                 elapsed_ms=0.0, success=False):
        self.path = path or []
        self.cost = cost
        self.nodes_expanded = nodes_expanded
        self.elapsed_ms = elapsed_ms
        self.success = success


def astar_search(grid: np.ndarray,
                 start: tuple,
                 goal: tuple,
                 heuristic_weight: float = 1.0,
                 diagonal: bool = True,
                 max_iterations: int = 50000,
                 lethal_threshold: int = 90,
                 cost_penalty_factor: float = 2.0) -> AStarResult:
    """
    Weighted A* on a 2D occupancy grid.

    Args:
        grid:               (H, W) int8 array.  -1=unknown, 0=free, 1-100=cost
        start:              (row, col) grid indices
        goal:               (row, col) grid indices
        heuristic_weight:   epsilon for weighted A* (1.0=optimal, >1=greedy)
        diagonal:           allow 8-connected movement
        max_iterations:     hard cap on node expansions
        lethal_threshold:   cells >= this are impassable
        cost_penalty_factor: multiplier for traversing high-cost cells

    Returns:
        AStarResult with path as list of (row, col) grid cells.
    """
    t0 = time.monotonic()
    rows, cols = grid.shape

    # Validate start / goal
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return AStarResult(success=False)
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return AStarResult(success=False)
    if grid[start[0], start[1]] >= lethal_threshold:
        return AStarResult(success=False)
    if grid[goal[0], goal[1]] >= lethal_threshold:
        return AStarResult(success=False)

    # Movement directions
    SQRT2 = 1.414
    if diagonal:
        directions = [
            (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
            (-1, -1, SQRT2), (-1,  1, SQRT2), ( 1, -1, SQRT2), ( 1,  1, SQRT2),
        ]
    else:
        directions = [
            (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
        ]

    # Heuristic: octile distance (admissible for 8-connected)
    def heuristic(r, c):
        dr = abs(r - goal[0])
        dc = abs(c - goal[1])
        if diagonal:
            return (max(dr, dc) + (SQRT2 - 1) * min(dr, dc)) * heuristic_weight
        return (dr + dc) * heuristic_weight

    # Open list: (f_cost, counter, row, col)
    counter = 0
    open_list = []
    heapq.heappush(open_list, (heuristic(start[0], start[1]), counter, start[0], start[1]))
    counter += 1

    g_cost = np.full((rows, cols), np.inf, dtype=np.float64)
    g_cost[start[0], start[1]] = 0.0

    came_from = {}
    closed = np.zeros((rows, cols), dtype=bool)

    nodes_expanded = 0

    while open_list and nodes_expanded < max_iterations:
        f, _, r, c = heapq.heappop(open_list)

        if closed[r, c]:
            continue
        closed[r, c] = True
        nodes_expanded += 1

        # Goal reached
        if r == goal[0] and c == goal[1]:
            path = []
            cr, cc = r, c
            while (cr, cc) in came_from:
                path.append((cr, cc))
                cr, cc = came_from[(cr, cc)]
            path.append(start)
            path.reverse()

            elapsed = (time.monotonic() - t0) * 1000.0
            return AStarResult(
                path=path,
                cost=g_cost[r, c],
                nodes_expanded=nodes_expanded,
                elapsed_ms=elapsed,
                success=True)

        for dr, dc, move_cost in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if closed[nr, nc]:
                continue

            cell_val = grid[nr, nc]

            if cell_val >= lethal_threshold:
                continue

            # Unknown cells: treat as traversable but penalised
            if cell_val < 0:
                traverse_cost = move_cost * 1.5
            elif cell_val == 0:
                traverse_cost = move_cost
            else:
                traverse_cost = move_cost * (1.0 + cost_penalty_factor * cell_val / 100.0)

            tentative_g = g_cost[r, c] + traverse_cost
            if tentative_g < g_cost[nr, nc]:
                g_cost[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                f_new = tentative_g + heuristic(nr, nc)
                heapq.heappush(open_list, (f_new, counter, nr, nc))
                counter += 1

    elapsed = (time.monotonic() - t0) * 1000.0
    return AStarResult(nodes_expanded=nodes_expanded, elapsed_ms=elapsed, success=False)
