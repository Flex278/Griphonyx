"""
3D Hybrid A* path planner for UAV autonomous navigation.

Implements a 5-DOF (x, y, z, yaw, pitch) Hybrid A* search that supports
both holonomic HOVER flight and non-holonomic CRUISE flight.  Cost weights
are tuned for passenger comfort and FAA Part 107 altitude compliance.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..planners.base_planner import BasePlanner

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Pose5D = Tuple[float, float, float, float, float]  # (x, y, z, yaw, pitch)


# ---------------------------------------------------------------------------
# Cost weights
# ---------------------------------------------------------------------------
W_STEER: float = 1.2       # Penalise lateral steering (smoothness)
W_PITCH: float = 1.0       # Penalise pitch (altitude change effort)
W_PITCH_CHANGE: float = 2.0  # Penalise rapid pitch changes (comfort)
W_MODE_SWITCH: float = 3.0   # Penalise mode transitions (stability)
W_ALTITUDE_FLOOR: float = 50.0  # Heavy penalty for breaching min altitude


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FlightMode(IntEnum):
    """Flight mode enumeration for the hybrid planner.

    Attributes:
        HOVER: Holonomic 26-direction movement; used for precision manoeuvres.
        CRUISE: Non-holonomic forward flight; used for efficient transit.
    """

    HOVER = 1
    CRUISE = 2


# ---------------------------------------------------------------------------
# Vehicle configuration
# ---------------------------------------------------------------------------

@dataclass
class VehicleConfig:
    """Physical dimensions and operational constraints of the UAV.

    Attributes:
        length: Body length in metres.
        width: Body width (span) in metres.
        height: Body height in metres.
        rotor_diameter: Maximum rotor diameter in metres.
        min_altitude: Minimum flight altitude AGL in metres (default 5 m).
        max_altitude: Maximum flight altitude AGL in metres (default 120 m,
            FAA Part 107).
        min_turn_radius: Minimum horizontal turn radius in metres.
        hover_step: Voxel step size for HOVER motion primitives in metres.
        cruise_step: Voxel step size for CRUISE motion primitives in metres.
        inflation: Safety inflation margin added to each body dimension.
    """

    length: float = 1.8
    width: float = 1.2
    height: float = 0.6
    rotor_diameter: float = 0.5
    min_altitude: float = 5.0
    max_altitude: float = 120.0
    min_turn_radius: float = 3.0
    hover_step: float = 1.0
    cruise_step: float = 2.0
    inflation: float = 0.35


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node3D:
    """A single state in the Hybrid A* search graph.

    Attributes:
        x: World-frame x position in metres.
        y: World-frame y position in metres.
        z: World-frame z (altitude) position in metres.
        yaw: Heading angle in radians.
        pitch: Pitch angle in radians.
        parent: Index key of the parent node in the closed-set dict, or
            ``None`` for the root.
        g: Cost-to-come from the start.
        h: Heuristic estimate of cost-to-go to the goal.
        mode: Current :class:`FlightMode`.
    """

    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    parent: Optional[str]
    g: float
    h: float
    mode: FlightMode

    # f is derived so we keep it out of __init__
    f: float = field(init=False)

    def __post_init__(self) -> None:
        self.f = self.g + self.h

    # ------------------------------------------------------------------
    # Heap ordering (min-heap on f)
    # ------------------------------------------------------------------

    def __lt__(self, other: "Node3D") -> bool:  # noqa: D105
        return self.f < other.f

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, Node3D):
            return NotImplemented
        return self.key == other.key

    # ------------------------------------------------------------------
    # Grid discretisation
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        """Discrete hash key used for the closed/open sets.

        Positions are rounded to 0.5 m and angles to 15°.
        """
        xi = round(self.x * 2) / 2
        yi = round(self.y * 2) / 2
        zi = round(self.z * 2) / 2
        yaw_d = round(math.degrees(self.yaw) / 15) * 15
        pitch_d = round(math.degrees(self.pitch) / 15) * 15
        mode_i = int(self.mode)
        return f"{xi},{yi},{zi},{yaw_d},{pitch_d},{mode_i}"


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

class HybridAStarPlanner3D(BasePlanner):
    """3D Hybrid A* path planner for UAV collision-free navigation.

    Supports 5-DOF state (x, y, z, yaw, pitch) with two flight modes:
    *HOVER* (26-direction holonomic) and *CRUISE* (non-holonomic forward).

    Args:
        voxel_map: A :class:`~integrated_planning.maps.voxel_map_3d.VoxelMap`
            instance that provides obstacle information.
        vehicle: Vehicle configuration.  Defaults to ``VehicleConfig()``.
        preferred_mode: The initial / preferred flight mode.  Defaults to
            ``FlightMode.HOVER``.

    Example::

        vmap = VoxelMap(100, 100, 50, resolution=1.0)
        vmap.add_box_obstacle(50, 50, 0, 10, 10, 20)

        planner = HybridAStarPlanner3D(vmap)
        path = planner.plan((0, 0, 10, 0, 0), (90, 90, 15, 0, 0))
    """

    def __init__(
        self,
        voxel_map,  # VoxelMap – avoid circular import at type-hint level
        vehicle: Optional[VehicleConfig] = None,
        preferred_mode: FlightMode = FlightMode.HOVER,
    ) -> None:
        super().__init__(name="HybridAStarPlanner3D")
        self.voxel_map = voxel_map
        self.vehicle: VehicleConfig = vehicle or VehicleConfig()
        self.preferred_mode = preferred_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: Pose5D,
        goal: Pose5D,
        max_iter: int = 80_000,
    ) -> Optional[List[Pose5D]]:
        """Find a collision-free path from *start* to *goal*.

        Runs 3D Hybrid A* up to *max_iter* node expansions.

        Args:
            start: Starting pose ``(x, y, z, yaw, pitch)``.
            goal: Goal pose ``(x, y, z, yaw, pitch)``.
            max_iter: Maximum number of node expansions before giving up.
                Use smaller values (e.g. 20 000) for fast emergency replanning.

        Returns:
            Ordered list of pose tuples from *start* to *goal*, or ``None``
            if no path was found within *max_iter* iterations.
        """
        if not self.validate_pose(start):
            return None
        if not self.validate_pose(goal):
            return None

        start_node = self._make_node(
            start[0], start[1], start[2],
            start[3], start[4],
            parent=None,
            mode=self.preferred_mode,
        )
        start_node.g = 0.0
        start_node.h = self._heuristic(start_node, goal)
        start_node.f = start_node.g + start_node.h

        open_heap: List[Tuple[float, Node3D]] = []
        heapq.heappush(open_heap, (start_node.f, start_node))

        closed: Dict[str, Node3D] = {}
        all_nodes: Dict[str, Node3D] = {start_node.key: start_node}

        goal_node: Optional[Node3D] = None
        goal_tolerance = max(self.vehicle.hover_step * 1.5, 1.5)

        for _ in range(max_iter):
            if not open_heap:
                break

            _, current = heapq.heappop(open_heap)
            key = current.key

            if key in closed:
                continue
            closed[key] = current

            # ---- Goal check ----
            dist_to_goal = math.sqrt(
                (current.x - goal[0]) ** 2
                + (current.y - goal[1]) ** 2
                + (current.z - goal[2]) ** 2
            )
            if dist_to_goal <= goal_tolerance:
                goal_node = current
                break

            # ---- Expand neighbours ----
            for mode in (FlightMode.HOVER, FlightMode.CRUISE):
                for dx, dy, dz, dyaw, dpitch in self._get_motion_primitives(mode):
                    nx = current.x + dx
                    ny = current.y + dy
                    nz = current.z + dz
                    nyaw = self._normalise_angle(current.yaw + dyaw)
                    npitch = max(
                        -math.pi / 4,
                        min(math.pi / 4, current.pitch + dpitch),
                    )

                    candidate = self._make_node(
                        nx, ny, nz, nyaw, npitch,
                        parent=key,
                        mode=mode,
                    )

                    if candidate.key in closed:
                        continue
                    if not self._altitude_safe(nz):
                        continue
                    if not self._collision_free(candidate):
                        continue

                    # Cost-to-come
                    step_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    mode_penalty = (
                        W_MODE_SWITCH if mode != current.mode else 0.0
                    )
                    steer_penalty = W_STEER * abs(dyaw)
                    pitch_penalty = W_PITCH * abs(dpitch)
                    pitch_change_penalty = (
                        W_PITCH_CHANGE * abs(npitch - current.pitch)
                    )
                    alt_floor_penalty = (
                        W_ALTITUDE_FLOOR
                        if nz < self.vehicle.min_altitude + 1.0
                        else 0.0
                    )
                    g_new = (
                        current.g
                        + step_dist
                        + mode_penalty
                        + steer_penalty
                        + pitch_penalty
                        + pitch_change_penalty
                        + alt_floor_penalty
                    )

                    if candidate.key in all_nodes:
                        existing = all_nodes[candidate.key]
                        if g_new >= existing.g:
                            continue

                    candidate.g = g_new
                    candidate.h = self._heuristic(candidate, goal)
                    candidate.f = candidate.g + candidate.h
                    all_nodes[candidate.key] = candidate
                    heapq.heappush(open_heap, (candidate.f, candidate))

        if goal_node is None:
            return None

        return self._reconstruct_path(goal_node, all_nodes)

    def validate_pose(self, pose: Tuple[float, ...]) -> bool:
        """Check that a pose is within operational bounds.

        Args:
            pose: At least ``(x, y, z)`` tuple.

        Returns:
            ``True`` if the pose passes all checks.
        """
        if len(pose) < 3:
            return False
        z = pose[2]
        return self._altitude_safe(z)

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------

    def _get_motion_primitives(
        self, mode: FlightMode
    ) -> List[Tuple[float, float, float, float, float]]:
        """Generate motion primitives for the given flight mode.

        Args:
            mode: :class:`FlightMode` determining the set of actions.

        Returns:
            List of ``(dx, dy, dz, dyaw, dpitch)`` tuples.
        """
        if mode == FlightMode.HOVER:
            return self._hover_primitives()
        return self._cruise_primitives()

    def _hover_primitives(
        self,
    ) -> List[Tuple[float, float, float, float, float]]:
        """26-direction holonomic motion primitives for HOVER mode.

        Returns:
            List of ``(dx, dy, dz, 0, 0)`` unit steps scaled by
            ``vehicle.hover_step``.
        """
        s = self.vehicle.hover_step
        directions = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                    directions.append(
                        (s * dx / norm, s * dy / norm, s * dz / norm, 0.0, 0.0)
                    )
        return directions

    def _cruise_primitives(
        self,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Non-holonomic forward motion primitives for CRUISE mode.

        Generates straight-ahead, banked left/right, climb, and dive
        primitives based on vehicle cruise step and turn radius.

        Returns:
            List of ``(dx, dy, dz, dyaw, dpitch)`` tuples.
        """
        step = self.vehicle.cruise_step
        r = self.vehicle.min_turn_radius
        # Approximate steering angle from turn radius and step length
        d_steer = math.asin(min(step / (2.0 * r), 1.0)) if r > 0 else 0.0
        d_pitch = math.radians(10.0)

        primitives = []
        for dyaw in (0.0, d_steer, -d_steer):
            for dpitch in (0.0, d_pitch, -d_pitch):
                dx = step * math.cos(dpitch) * math.cos(dyaw)
                dy = step * math.cos(dpitch) * math.sin(dyaw)
                dz = step * math.sin(dpitch)
                primitives.append((dx, dy, dz, dyaw, dpitch))
        return primitives

    # ------------------------------------------------------------------
    # Collision & altitude
    # ------------------------------------------------------------------

    def _collision_free(self, node: Node3D) -> bool:
        """Check whether the vehicle AABB at *node* is obstacle-free.

        Args:
            node: Candidate node to test.

        Returns:
            ``True`` if the node is collision-free.
        """
        return not self.voxel_map.aabb_collision(
            node.x, node.y, node.z,
            node.yaw, node.pitch,
            self.vehicle,
        )

    def _altitude_safe(self, z: float) -> bool:
        """Return ``True`` if *z* is within the allowed altitude band.

        Args:
            z: Altitude in metres (AGL).

        Returns:
            ``True`` when ``vehicle.min_altitude <= z <= vehicle.max_altitude``.
        """
        return self.vehicle.min_altitude <= z <= self.vehicle.max_altitude

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(node: Node3D, goal: Pose5D) -> float:
        """3-D Euclidean distance from *node* to *goal*.

        Args:
            node: Current search node.
            goal: Goal pose tuple.

        Returns:
            Lower-bound cost estimate.
        """
        dx = node.x - goal[0]
        dy = node.y - goal[1]
        dz = node.z - goal[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # ------------------------------------------------------------------
    # Path reconstruction & smoothing
    # ------------------------------------------------------------------

    def _reconstruct_path(
        self,
        goal_node: Node3D,
        all_nodes: Dict[str, Node3D],
    ) -> List[Pose5D]:
        """Walk the parent chain to build the path from start to goal.

        Args:
            goal_node: The node at (or near) the goal position.
            all_nodes: Dictionary mapping node keys to :class:`Node3D`.

        Returns:
            Ordered list of ``(x, y, z, yaw, pitch)`` tuples.
        """
        path: List[Pose5D] = []
        current: Optional[Node3D] = goal_node
        while current is not None:
            path.append((current.x, current.y, current.z,
                          current.yaw, current.pitch))
            if current.parent is None:
                break
            current = all_nodes.get(current.parent)
        path.reverse()
        return path

    @staticmethod
    def smooth_path(
        path: List[Pose5D],
        iterations: int = 150,
    ) -> List[Pose5D]:
        """Smooth a path using iterative gradient descent.

        Applies an iterative averaging filter while keeping the start and
        goal fixed.  The algorithm is inspired by the gradient descent path
        smoother used in navigation stacks.

        Args:
            path: Raw path as a list of ``(x, y, z, yaw, pitch)`` tuples.
            iterations: Number of smoothing passes.  More passes → smoother
                path but more computation.  Defaults to 150.

        Returns:
            Smoothed path with the same number of waypoints.  Start and goal
            are unchanged.
        """
        if len(path) < 3:
            return path

        # Work on a mutable copy (only x, y, z are smoothed)
        pts = [[p[0], p[1], p[2]] for p in path]
        alpha = 0.5  # Data weight
        beta = 0.3   # Smoothness weight

        for _ in range(iterations):
            for i in range(1, len(pts) - 1):
                for dim in range(3):
                    original = path[i][dim]
                    prev = pts[i - 1][dim]
                    curr = pts[i][dim]
                    nxt = pts[i + 1][dim]
                    pts[i][dim] = (
                        curr
                        + alpha * (original - curr)
                        + beta * (prev + nxt - 2.0 * curr)
                    )

        # Re-attach yaw and pitch from original path
        smoothed: List[Pose5D] = [
            (pts[i][0], pts[i][1], pts[i][2], path[i][3], path[i][4])
            for i in range(len(path))
        ]
        return smoothed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_node(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        pitch: float,
        parent: Optional[str],
        mode: FlightMode,
    ) -> Node3D:
        """Construct a :class:`Node3D` with zero cost values.

        Args:
            x: X position.
            y: Y position.
            z: Z position (altitude).
            yaw: Heading in radians.
            pitch: Pitch angle in radians.
            parent: Key of the parent node, or ``None``.
            mode: :class:`FlightMode`.

        Returns:
            A new :class:`Node3D` instance with ``g=0``, ``h=0``.
        """
        node = Node3D(
            x=x, y=y, z=z,
            yaw=yaw, pitch=pitch,
            parent=parent,
            g=0.0, h=0.0,
            mode=mode,
        )
        return node

    @staticmethod
    def _normalise_angle(angle: float) -> float:
        """Normalise an angle to the range ``(-π, π]``.

        Args:
            angle: Angle in radians (any value).

        Returns:
            Equivalent angle in ``(-π, π]``.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle
