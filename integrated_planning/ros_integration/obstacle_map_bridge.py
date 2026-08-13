"""
ObstacleMapBridge – ROS2 node that bridges obstacle detection to VoxelMap.

Subscribes to LiDAR point-cloud and obstacle-position topics and keeps
a shared :class:`~integrated_planning.maps.voxel_map_3d.VoxelMap` up to date.
Publishes a ``MarkerArray`` visualization for RViz2.
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Vector3Stamped
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import Marker, MarkerArray

    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover – ROS2 not installed in CI
    _ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

from ..maps.voxel_map_3d import VoxelMap


class ObstacleMapBridge(Node):
    """ROS2 node that converts obstacle detection data into a VoxelMap.

    Subscribes to LiDAR point-clouds and obstacle-position messages and
    marks the corresponding voxels as occupied.  Periodically publishes a
    :class:`visualization_msgs.msg.MarkerArray` so that the map can be
    visualised in RViz2.

    ROS2 Parameters:
        map_size_x (float): X extent of the voxel map in metres (default 100).
        map_size_y (float): Y extent in metres (default 100).
        map_size_z (float): Z extent in metres (default 50).
        map_resolution (float): Voxel side length in metres (default 1.0).
        obstacle_inflation (float): Extra inflation added around detected
            obstacles in metres (default 0.5).
        update_rate (float): Visualization publish rate in Hz (default 10.0).

    Subscribed Topics:
        /lidar/points (sensor_msgs/PointCloud2): Live LiDAR scan.
        /obstacle_position (geometry_msgs/Vector3Stamped): Position of a
            detected obstacle (from obstacle_detector node).

    Published Topics:
        /voxel_map_viz (visualization_msgs/MarkerArray): Occupied-voxel
            markers for RViz2.
    """

    def __init__(self) -> None:
        """Initialise node, voxel map, subscribers, publishers, and timer."""
        super().__init__("obstacle_map_bridge")

        # --- Parameters -------------------------------------------------------
        self.declare_parameter("map_size_x", 100.0)
        self.declare_parameter("map_size_y", 100.0)
        self.declare_parameter("map_size_z", 50.0)
        self.declare_parameter("map_resolution", 1.0)
        self.declare_parameter("obstacle_inflation", 0.5)
        self.declare_parameter("update_rate", 10.0)

        sx = self.get_parameter("map_size_x").get_parameter_value().double_value
        sy = self.get_parameter("map_size_y").get_parameter_value().double_value
        sz = self.get_parameter("map_size_z").get_parameter_value().double_value
        res = self.get_parameter("map_resolution").get_parameter_value().double_value
        self.obstacle_inflation = (
            self.get_parameter("obstacle_inflation")
            .get_parameter_value()
            .double_value
        )
        update_rate = (
            self.get_parameter("update_rate").get_parameter_value().double_value
        )

        # --- Shared VoxelMap --------------------------------------------------
        self.voxel_map = VoxelMap(sx, sy, sz, resolution=res)
        self.get_logger().info(
            f"VoxelMap initialised: "
            f"({sx}x{sy}x{sz}) m @ {res} m/voxel"
        )

        # --- Subscribers ------------------------------------------------------
        self._pc_sub = self.create_subscription(
            PointCloud2,
            "/lidar/points",
            self.pointcloud_callback,
            10,
        )
        self._obs_sub = self.create_subscription(
            Vector3Stamped,
            "/obstacle_position",
            self.obstacle_position_callback,
            10,
        )

        # --- Publishers -------------------------------------------------------
        self._viz_pub = self.create_publisher(MarkerArray, "/voxel_map_viz", 10)

        # --- Visualization timer ----------------------------------------------
        period = 1.0 / max(update_rate, 0.1)
        self._viz_timer = self.create_timer(period, self.publish_map_visualization)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def pointcloud_callback(self, msg: "PointCloud2") -> None:
        """Process an incoming LiDAR scan and update the voxel map.

        Args:
            msg: Incoming :class:`sensor_msgs.msg.PointCloud2` message.
        """
        try:
            points = self.pointcloud2_to_array(msg)
            if points.shape[0] > 0:
                self.voxel_map.update_from_pointcloud(points)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(
                f"pointcloud_callback error: {exc}", throttle_duration_sec=5.0
            )

    def obstacle_position_callback(self, msg: "Vector3Stamped") -> None:
        """Add a sphere obstacle at the detected position.

        The sphere radius is set to ``obstacle_inflation``.

        Args:
            msg: Obstacle position message.
        """
        try:
            radius = max(self.obstacle_inflation, 0.5)
            self.voxel_map.add_sphere_obstacle(
                msg.vector.x, msg.vector.y, msg.vector.z, radius
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(
                f"obstacle_position_callback error: {exc}",
                throttle_duration_sec=5.0,
            )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def publish_map_visualization(self) -> None:
        """Publish occupied voxels as a :class:`MarkerArray` for RViz2.

        Only publishes if there are any occupied voxels.
        """
        try:
            occupied = self.voxel_map.get_occupied_voxels()
            if not occupied:
                return

            marker_array = MarkerArray()
            res = self.voxel_map.resolution

            # Use a single CUBE_LIST marker for efficiency
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "voxel_map"
            m.id = 0
            m.type = Marker.CUBE_LIST
            m.action = Marker.ADD
            m.scale.x = res * 0.95
            m.scale.y = res * 0.95
            m.scale.z = res * 0.95
            m.color.r = 0.8
            m.color.g = 0.2
            m.color.b = 0.2
            m.color.a = 0.7

            from geometry_msgs.msg import Point  # local import

            for ix, iy, iz in occupied:
                wx, wy, wz = self.voxel_map.voxel_to_world(ix, iy, iz)
                pt = Point()
                pt.x = wx
                pt.y = wy
                pt.z = wz
                m.points.append(pt)

            marker_array.markers.append(m)
            self._viz_pub.publish(marker_array)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(
                f"publish_map_visualization error: {exc}",
                throttle_duration_sec=5.0,
            )

    # ------------------------------------------------------------------
    # PointCloud2 conversion
    # ------------------------------------------------------------------

    @staticmethod
    def pointcloud2_to_array(msg: "PointCloud2") -> np.ndarray:
        """Convert a :class:`sensor_msgs.msg.PointCloud2` to a numpy array.

        Parses the field layout of the message and extracts XYZ coordinates
        for every point.

        Args:
            msg: ROS2 PointCloud2 message.

        Returns:
            ``(N, 3)`` float32 numpy array of ``(x, y, z)`` coordinates.
            Returns an empty ``(0, 3)`` array on parse errors.
        """
        try:
            # Build field-offset map
            field_map = {f.name: f.offset for f in msg.fields}
            if not all(k in field_map for k in ("x", "y", "z")):
                return np.empty((0, 3), dtype=np.float32)

            x_off = field_map["x"]
            y_off = field_map["y"]
            z_off = field_map["z"]
            point_step = msg.point_step
            data = bytes(msg.data)
            n_points = len(data) // point_step

            points = np.empty((n_points, 3), dtype=np.float32)
            fmt = "f"  # 4-byte float
            for i in range(n_points):
                base = i * point_step
                points[i, 0] = struct.unpack_from(fmt, data, base + x_off)[0]
                points[i, 1] = struct.unpack_from(fmt, data, base + y_off)[0]
                points[i, 2] = struct.unpack_from(fmt, data, base + z_off)[0]

            # Remove NaN / Inf
            valid = np.isfinite(points).all(axis=1)
            return points[valid]
        except Exception:  # pylint: disable=broad-except
            return np.empty((0, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    """ROS2 entry point for the obstacle_map_bridge node."""
    if not _ROS_AVAILABLE:
        raise RuntimeError(
            "rclpy is not installed. "
            "Please source your ROS2 workspace before running this node."
        )
    rclpy.init(args=args)
    node = ObstacleMapBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
