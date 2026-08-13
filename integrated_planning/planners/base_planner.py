"""
Abstract base class for all path planners.

This module provides the interface that all planner implementations must satisfy,
along with shared validation utilities.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BasePlanner(ABC):
    """Abstract base class for all path planners.

    Every concrete planner must implement :meth:`plan` and
    :meth:`validate_pose` so that the rest of the system can swap planning
    back-ends without changing consumer code.

    Attributes:
        name: Human-readable planner identifier used for logging.
    """

    def __init__(self, name: str = "BasePlanner") -> None:
        """Initialize the planner.

        Args:
            name: Human-readable identifier for the planner (used in logs).
        """
        self.name = name

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def plan(
        self,
        start: Tuple[float, ...],
        goal: Tuple[float, ...],
    ) -> Optional[List[Tuple[float, ...]]]:
        """Compute a collision-free path from *start* to *goal*.

        Args:
            start: Starting pose tuple.  The exact length and semantics are
                planner-specific (e.g. ``(x, y, z, yaw, pitch)``).
            goal: Goal pose tuple in the same format as *start*.

        Returns:
            An ordered list of pose tuples from *start* to *goal*, or
            ``None`` if no path could be found within the allowed
            computational budget.
        """

    @abstractmethod
    def validate_pose(self, pose: Tuple[float, ...]) -> bool:
        """Validate that *pose* is a legal configuration.

        Checks include bounds, altitude limits, and any planner-specific
        geometric constraints.

        Args:
            pose: Pose tuple to validate.

        Returns:
            ``True`` if the pose is valid, ``False`` otherwise.
        """

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def path_length(path: List[Tuple[float, ...]]) -> float:
        """Compute the total Euclidean arc-length of a path.

        Only the first three components of each pose tuple (x, y, z) are
        used for the distance calculation.

        Args:
            path: Ordered sequence of pose tuples.

        Returns:
            Total 3-D arc-length in the same units as the pose coordinates.

        Raises:
            ValueError: If *path* is empty or poses have fewer than three
                components.
        """
        if not path:
            raise ValueError("path must contain at least one pose")
        if len(path[0]) < 3:
            raise ValueError("each pose must have at least 3 components (x, y, z)")

        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            dz = path[i][2] - path[i - 1][2]
            total += math.sqrt(dx * dx + dy * dy + dz * dz)
        return total

    @staticmethod
    def validate_path(
        path: List[Tuple[float, ...]],
        max_segment_length: float = 10.0,
    ) -> bool:
        """Validate that no two consecutive waypoints are too far apart.

        This is a sanity check for planner output: unreasonably long
        segments may indicate a bug in the planner or a corrupted path.

        Args:
            path: Ordered sequence of pose tuples.
            max_segment_length: Maximum allowed Euclidean distance between
                consecutive waypoints.  Defaults to 10 metres.

        Returns:
            ``True`` if the path is well-formed, ``False`` otherwise.
        """
        if len(path) < 2:
            return True
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            dz = path[i][2] - path[i - 1][2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) > max_segment_length:
                return False
        return True

    @staticmethod
    def euclidean_distance_3d(
        a: Tuple[float, ...],
        b: Tuple[float, ...],
    ) -> float:
        """Return the 3-D Euclidean distance between poses *a* and *b*.

        Only the first three components are used.

        Args:
            a: First pose tuple (at least 3 components).
            b: Second pose tuple (at least 3 components).

        Returns:
            3-D Euclidean distance.
        """
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
