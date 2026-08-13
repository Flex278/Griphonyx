#!/usr/bin/env python3
"""
Costmap interface for the A* planner.

Bridges the A* planner with the existing costmap node (costmap/costmap_node.py),
accepting nav_msgs/OccupancyGrid messages and converting them to the numpy
arrays consumed by astar_algorithm.astar_search.

The costmap node publishes on /costmap/grid (OccupancyGrid).  This interface
subscribes to that topic and also accepts the /obstacle_velocity topic so the
planner can account for dynamic obstacle inflation already baked in.
"""

import numpy as np

try:
    from nav_msgs.msg import OccupancyGrid
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


class CostmapInterface:
    """
    Converts nav_msgs/OccupancyGrid messages to numpy arrays for A*.

    Usage (inside a ROS 2 node)::

        from astar_planner.costmap_interface import CostmapInterface

        self.costmap_iface = CostmapInterface(lethal_threshold=70)
        self.create_subscription(OccupancyGrid, '/costmap/grid',
                                 self.costmap_iface.update, 10)

        # Later, to plan:
        grid_np, meta = self.costmap_iface.get_grid()
        if grid_np is not None:
            start_rc = self.costmap_iface.world_to_grid(sx, sy, meta)
            goal_rc  = self.costmap_iface.world_to_grid(gx, gy, meta)
    """

    def __init__(self, lethal_threshold: int = 70):
        self.lethal_threshold = lethal_threshold
        self._grid_msg = None          # latest OccupancyGrid
        self._grid_np: np.ndarray | None = None   # cached numpy view

    # ------------------------------------------------------------------
    # Subscriber callback
    # ------------------------------------------------------------------

    def update(self, msg: 'OccupancyGrid') -> None:
        """Store the latest OccupancyGrid message and invalidate the cache."""
        self._grid_msg = msg
        self._grid_np = None  # invalidate cache

    # ------------------------------------------------------------------
    # Grid access
    # ------------------------------------------------------------------

    def get_grid(self):
        """
        Return (grid_np, grid_msg) where grid_np is an (H, W) int8 array.

        Returns (None, None) if no costmap has been received yet.
        """
        if self._grid_msg is None:
            return None, None

        if self._grid_np is None:
            self._grid_np = np.array(
                self._grid_msg.data, dtype=np.int8
            ).reshape(self._grid_msg.info.height, self._grid_msg.info.width)

        return self._grid_np, self._grid_msg

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def world_to_grid(wx: float, wy: float, grid_msg: 'OccupancyGrid'):
        """
        Convert world (x, y) metres to grid (row, col) indices.

        Returns (row, col) or None if the point is outside the grid.
        """
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y

        col = int((wx - ox) / res)
        row = int((wy - oy) / res)

        if 0 <= row < grid_msg.info.height and 0 <= col < grid_msg.info.width:
            return row, col
        return None

    @staticmethod
    def grid_to_world(row: int, col: int, grid_msg: 'OccupancyGrid'):
        """Convert grid (row, col) to world (x, y) metres (cell centre)."""
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y

        wx = ox + (col + 0.5) * res
        wy = oy + (row + 0.5) * res
        return wx, wy

    def find_nearest_free(self, grid_np: np.ndarray, start_rc: tuple,
                          search_radius: int = 10):
        """
        If the start cell is lethal, spiral outward to find the nearest free cell.

        Returns (row, col) or None if no free cell is found within search_radius.
        """
        r, c = start_rc
        if grid_np[r, c] < self.lethal_threshold:
            return start_rc

        rows, cols = grid_np.shape
        for radius in range(1, search_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid_np[nr, nc] < self.lethal_threshold:
                            return nr, nc
        return None
