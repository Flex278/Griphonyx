"""
3D voxel occupancy grid for environment representation.

Provides efficient storage and query of 3-D obstacle geometry using a
dense numpy array.  Supports box, cylinder, and sphere obstacle primitives
as well as point-cloud updates from LiDAR.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


class VoxelMap:
    """3D occupancy grid backed by a numpy ``uint8`` array.

    The map spans from the world origin ``(0, 0, 0)`` to
    ``(size_x, size_y, size_z)`` metres with uniform voxel side length
    *resolution*.

    Voxel values:
        * ``0`` – free
        * ``1`` – occupied

    Args:
        size_x: Map extent along the x axis in metres.
        size_y: Map extent along the y axis in metres.
        size_z: Map extent along the z axis in metres.
        resolution: Side length of each cubic voxel in metres.

    Example::

        vmap = VoxelMap(100, 100, 50, resolution=1.0)
        vmap.add_box_obstacle(50, 50, 0, 10, 10, 20)
        occupied = vmap.is_occupied_xyz(55, 55, 10)  # True
    """

    def __init__(
        self,
        size_x: float,
        size_y: float,
        size_z: float,
        resolution: float = 1.0,
    ) -> None:
        """Initialise the voxel map.

        Args:
            size_x: X extent in metres.
            size_y: Y extent in metres.
            size_z: Z extent in metres.
            resolution: Voxel side length in metres (default 1 m).

        Raises:
            ValueError: If any dimension or the resolution is not positive.
        """
        if size_x <= 0 or size_y <= 0 or size_z <= 0:
            raise ValueError("All map dimensions must be positive.")
        if resolution <= 0:
            raise ValueError("Resolution must be positive.")

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.resolution = resolution

        # Number of voxels along each axis
        self.nx: int = max(1, math.ceil(size_x / resolution))
        self.ny: int = max(1, math.ceil(size_y / resolution))
        self.nz: int = max(1, math.ceil(size_z / resolution))

        # Dense occupancy array: 0 = free, 1 = occupied
        self._grid: np.ndarray = np.zeros(
            (self.nx, self.ny, self.nz), dtype=np.uint8
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_voxel(
        self, x: float, y: float, z: float
    ) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices.

        Args:
            x: World x coordinate in metres.
            y: World y coordinate in metres.
            z: World z coordinate in metres.

        Returns:
            ``(ix, iy, iz)`` voxel indices (may be out of bounds).
        """
        ix = int(math.floor(x / self.resolution))
        iy = int(math.floor(y / self.resolution))
        iz = int(math.floor(z / self.resolution))
        return ix, iy, iz

    def voxel_to_world(
        self, ix: int, iy: int, iz: int
    ) -> Tuple[float, float, float]:
        """Convert voxel indices to world coordinates (voxel centre).

        Args:
            ix: Voxel index along x.
            iy: Voxel index along y.
            iz: Voxel index along z.

        Returns:
            ``(x, y, z)`` world coordinates at the centre of the voxel.
        """
        x = (ix + 0.5) * self.resolution
        y = (iy + 0.5) * self.resolution
        z = (iz + 0.5) * self.resolution
        return x, y, z

    def _in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        """Return ``True`` if voxel indices are within the grid."""
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

    # ------------------------------------------------------------------
    # Obstacle management
    # ------------------------------------------------------------------

    def add_box_obstacle(
        self,
        x: float,
        y: float,
        z: float,
        size_x: float,
        size_y: float,
        size_z: float,
    ) -> None:
        """Mark a rectangular box as occupied.

        The box is axis-aligned with its minimum corner at
        ``(x, y, z)`` and extents ``(size_x, size_y, size_z)``.

        Args:
            x: Minimum x of the box in metres.
            y: Minimum y of the box in metres.
            z: Minimum z of the box in metres.
            size_x: Box length along x in metres.
            size_y: Box length along y in metres.
            size_z: Box length along z in metres.
        """
        x_min, x_max = x, x + size_x
        y_min, y_max = y, y + size_y
        z_min, z_max = z, z + size_z

        ix0 = max(0, int(math.floor(x_min / self.resolution)))
        iy0 = max(0, int(math.floor(y_min / self.resolution)))
        iz0 = max(0, int(math.floor(z_min / self.resolution)))
        ix1 = min(self.nx, int(math.ceil(x_max / self.resolution)))
        iy1 = min(self.ny, int(math.ceil(y_max / self.resolution)))
        iz1 = min(self.nz, int(math.ceil(z_max / self.resolution)))

        if ix0 < ix1 and iy0 < iy1 and iz0 < iz1:
            self._grid[ix0:ix1, iy0:iy1, iz0:iz1] = 1

    def add_cylinder_obstacle(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        height: float,
    ) -> None:
        """Mark a vertical cylinder as occupied.

        The cylinder is centred at ``(x, y)`` with its base at altitude *z*.

        Args:
            x: Centre x of the cylinder in metres.
            y: Centre y of the cylinder in metres.
            z: Base altitude of the cylinder in metres.
            radius: Cylinder radius in metres.
            height: Cylinder height in metres.
        """
        ix0 = max(0, int(math.floor((x - radius) / self.resolution)))
        ix1 = min(self.nx, int(math.ceil((x + radius) / self.resolution)))
        iy0 = max(0, int(math.floor((y - radius) / self.resolution)))
        iy1 = min(self.ny, int(math.ceil((y + radius) / self.resolution)))
        iz0 = max(0, int(math.floor(z / self.resolution)))
        iz1 = min(self.nz, int(math.ceil((z + height) / self.resolution)))

        r2 = radius * radius
        for ix in range(ix0, ix1):
            wx = (ix + 0.5) * self.resolution
            for iy in range(iy0, iy1):
                wy = (iy + 0.5) * self.resolution
                if (wx - x) ** 2 + (wy - y) ** 2 <= r2:
                    if iz0 < iz1:
                        self._grid[ix, iy, iz0:iz1] = 1

    def add_sphere_obstacle(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
    ) -> None:
        """Mark a sphere as occupied.

        Args:
            x: Centre x of the sphere in metres.
            y: Centre y of the sphere in metres.
            z: Centre z of the sphere in metres.
            radius: Sphere radius in metres.
        """
        ix0 = max(0, int(math.floor((x - radius) / self.resolution)))
        ix1 = min(self.nx, int(math.ceil((x + radius) / self.resolution)))
        iy0 = max(0, int(math.floor((y - radius) / self.resolution)))
        iy1 = min(self.ny, int(math.ceil((y + radius) / self.resolution)))
        iz0 = max(0, int(math.floor((z - radius) / self.resolution)))
        iz1 = min(self.nz, int(math.ceil((z + radius) / self.resolution)))

        r2 = radius * radius
        for ix in range(ix0, ix1):
            wx = (ix + 0.5) * self.resolution
            for iy in range(iy0, iy1):
                wy = (iy + 0.5) * self.resolution
                for iz in range(iz0, iz1):
                    wz = (iz + 0.5) * self.resolution
                    if (wx - x) ** 2 + (wy - y) ** 2 + (wz - z) ** 2 <= r2:
                        self._grid[ix, iy, iz] = 1

    def clear_map(self) -> None:
        """Reset all voxels to free (0)."""
        self._grid[:] = 0

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def is_occupied_voxel(self, ix: int, iy: int, iz: int) -> bool:
        """Return ``True`` if the voxel at index ``(ix, iy, iz)`` is occupied.

        Out-of-bounds indices are treated as occupied (conservative).

        Args:
            ix: Voxel x index.
            iy: Voxel y index.
            iz: Voxel z index.

        Returns:
            ``True`` if occupied or out of bounds.
        """
        if not self._in_bounds(ix, iy, iz):
            return True
        return bool(self._grid[ix, iy, iz])

    def is_occupied_xyz(self, x: float, y: float, z: float) -> bool:
        """Return ``True`` if the world-frame point ``(x, y, z)`` is occupied.

        Args:
            x: World x coordinate in metres.
            y: World y coordinate in metres.
            z: World z coordinate in metres.

        Returns:
            ``True`` if occupied or outside the map.
        """
        ix, iy, iz = self.world_to_voxel(x, y, z)
        return self.is_occupied_voxel(ix, iy, iz)

    def aabb_collision(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        pitch: float,
        vehicle,
    ) -> bool:
        """Check for collision using an axis-aligned bounding box sweep.

        Rotates the vehicle footprint by *yaw* and *pitch* to produce a
        conservative AABB, then queries the voxel grid.

        Args:
            x: Vehicle centre x in metres.
            y: Vehicle centre y in metres.
            z: Vehicle centre z in metres.
            yaw: Vehicle heading in radians.
            pitch: Vehicle pitch in radians.
            vehicle: :class:`~integrated_planning.planners.hybrid_astar_3d.VehicleConfig`
                describing vehicle geometry.

        Returns:
            ``True`` if any voxel within the AABB is occupied.
        """
        # Half-extents with inflation
        hl = (vehicle.length / 2.0) + vehicle.inflation
        hw = (vehicle.width / 2.0) + vehicle.inflation
        hh = (vehicle.height / 2.0) + vehicle.inflation

        # Conservatively expand AABB to account for rotation
        cos_y = abs(math.cos(yaw))
        sin_y = abs(math.sin(yaw))
        cos_p = abs(math.cos(pitch))
        sin_p = abs(math.sin(pitch))

        aabb_hx = hl * cos_y * cos_p + hw * sin_y + hh * sin_p
        aabb_hy = hl * sin_y * cos_p + hw * cos_y + hh * sin_p
        aabb_hz = hl * sin_p + hh * cos_p

        ix0 = max(0, int(math.floor((x - aabb_hx) / self.resolution)))
        ix1 = min(self.nx, int(math.ceil((x + aabb_hx) / self.resolution)))
        iy0 = max(0, int(math.floor((y - aabb_hy) / self.resolution)))
        iy1 = min(self.ny, int(math.ceil((y + aabb_hy) / self.resolution)))
        iz0 = max(0, int(math.floor((z - aabb_hz) / self.resolution)))
        iz1 = min(self.nz, int(math.ceil((z + aabb_hz) / self.resolution)))

        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            return False

        return bool(np.any(self._grid[ix0:ix1, iy0:iy1, iz0:iz1]))

    # ------------------------------------------------------------------
    # Point cloud integration
    # ------------------------------------------------------------------

    def update_from_pointcloud(self, points: np.ndarray) -> None:
        """Mark voxels as occupied from a point-cloud array.

        Points are downsampled to voxel resolution before marking.

        Args:
            points: ``(N, 3)`` float array of ``(x, y, z)`` points in the
                world frame.  Points outside the map are silently ignored.
        """
        if points.ndim != 2 or points.shape[1] < 3:
            return

        # Filter out NaN/Inf before floor to avoid cast warnings
        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        pts = points[valid_mask, :3]
        if pts.size == 0:
            return

        # Compute voxel indices for all points at once
        idx = np.floor(pts / self.resolution).astype(int)

        # Filter in-bounds
        mask = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.nx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < self.ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < self.nz)
        )
        idx = idx[mask]

        # Deduplicate (unique rows) then mark
        unique_idx = np.unique(idx, axis=0)
        self._grid[unique_idx[:, 0], unique_idx[:, 1], unique_idx[:, 2]] = 1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_occupied_voxels(self) -> List[Tuple[int, int, int]]:
        """Return a list of all occupied voxel indices.

        Returns:
            List of ``(ix, iy, iz)`` tuples for every occupied voxel.
        """
        indices = np.argwhere(self._grid == 1)
        return [(int(r[0]), int(r[1]), int(r[2])) for r in indices]

    def get_occupancy_rate(self) -> float:
        """Return the fraction of voxels that are occupied.

        Returns:
            Value in ``[0.0, 1.0]``.
        """
        total = self._grid.size
        if total == 0:
            return 0.0
        return float(np.sum(self._grid)) / total

    def save_map(self, filename: str) -> None:
        """Persist the voxel map to a compressed numpy file.

        Args:
            filename: Output file path (the ``.npz`` extension is added
                automatically by :func:`numpy.savez_compressed`).
        """
        np.savez_compressed(
            filename,
            grid=self._grid,
            size_x=self.size_x,
            size_y=self.size_y,
            size_z=self.size_z,
            resolution=self.resolution,
        )

    @classmethod
    def load_map(cls, filename: str) -> "VoxelMap":
        """Load a voxel map from a compressed numpy file.

        Args:
            filename: Path to the ``.npz`` file previously created by
                :meth:`save_map`.

        Returns:
            A fully restored :class:`VoxelMap` instance.
        """
        data = np.load(filename)
        vmap = cls(
            size_x=float(data["size_x"]),
            size_y=float(data["size_y"]),
            size_z=float(data["size_z"]),
            resolution=float(data["resolution"]),
        )
        vmap._grid = data["grid"].astype(np.uint8)
        return vmap

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VoxelMap("
            f"size=({self.size_x}, {self.size_y}, {self.size_z}), "
            f"resolution={self.resolution}, "
            f"voxels=({self.nx}x{self.ny}x{self.nz}))"
        )
