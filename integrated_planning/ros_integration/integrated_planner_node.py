"""
IntegratedPlannerNode – ROS2 node that integrates obstacle detection with
3D Hybrid A* path planning.

Connects the existing obstacle_detector outputs to the HybridAStarPlanner3D
and publishes nav_msgs/Path waypoints for PX4 consumption.
"""

from __future__ import annotations

import math
import struct
from typing import List, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped
    from nav_msgs.msg import Path
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Bool, Float32, String

    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover – ROS2 not installed in CI
    _ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

from ..maps.voxel_map_3d import VoxelMap
from ..planners.hybrid_astar_3d import (
    FlightMode,
    HybridAStarPlanner3D,
    VehicleConfig,
)

# ---------------------------------------------------------------------------
# Zone constants (must match obstacle_detector.py)
# ---------------------------------------------------------------------------
ZONE_CRITICAL = "CRITICAL"   # 0–1 m
ZONE_DANGER = "DANGER"       # 1–3 m
ZONE_WARNING = "WARNING"     # 3–6 m
ZONE_CAUTION = "CAUTION"     # 6–10 m
ZONE_SAFE = "SAFE"           # > 10 m

# Maximum iterations for normal and emergency planning passes
MAX_ITER_NORMAL = 80_000
MAX_ITER_EMERGENCY = 20_000


class IntegratedPlannerNode(Node):
    """ROS2 node that integrates obstacle detection with path planning.

    Subscribes to obstacle-detection topics published by *obstacle_detector*,
    maintains a live :class:`~integrated_planning.maps.voxel_map_3d.VoxelMap`,
    and replans the UAV path whenever the obstacle state changes.

    ROS2 Parameters:
        map_size_x (float): Voxel map X extent in metres (default 100).
        map_size_y (float): Voxel map Y extent in metres (default 100).
        map_size_z (float): Voxel map Z extent in metres (default 50).
        map_resolution (float): Voxel side length in metres (default 1.0).
        vehicle_length (float): Vehicle body length in metres (default 1.8).
        vehicle_width (float): Vehicle body width in metres (default 1.2).
        vehicle_height (float): Vehicle body height in metres (default 0.6).
        vehicle_inflation (float): Safety inflation in metres (default 0.35).
        vehicle_min_altitude (float): Minimum altitude AGL in metres (default 5).
        vehicle_max_altitude (float): Maximum altitude AGL (default 120).
        max_iterations (int): Max A* iterations for normal replanning (80 000).
        preferred_mode (str): "hover" or "cruise" (default "hover").
        replan_on_obstacle (bool): Enable obstacle-triggered replanning (True).
        emergency_replan_distance (float): Distance threshold for emergency
            replanning in metres (default 3.0).

    Subscribed Topics:
        /obstacle_detected (std_msgs/Bool)
        /obstacle_distance (std_msgs/Float32)
        /avoidance_zone (std_msgs/String)
        /goal_pose (geometry_msgs/PoseStamped)
        /current_pose (geometry_msgs/PoseStamped)

    Published Topics:
        /planned_path (nav_msgs/Path): Waypoints for PX4.
        /planner_status (std_msgs/String): Human-readable status messages.
    """

    def __init__(self) -> None:
        """Initialise the node, planner, subscribers, publishers, and timers."""
        super().__init__("integrated_planner_node")

        # --- Parameters -------------------------------------------------------
        self._declare_parameters()

        sx = self._param_float("map_size_x")
        sy = self._param_float("map_size_y")
        sz = self._param_float("map_size_z")
        res = self._param_float("map_resolution")

        vehicle = VehicleConfig(
            length=self._param_float("vehicle_length"),
            width=self._param_float("vehicle_width"),
            height=self._param_float("vehicle_height"),
            inflation=self._param_float("vehicle_inflation"),
            min_altitude=self._param_float("vehicle_min_altitude"),
            max_altitude=self._param_float("vehicle_max_altitude"),
            min_turn_radius=self._param_float("min_turn_radius"),
            hover_step=self._param_float("hover_step"),
            cruise_step=self._param_float("cruise_step"),
        )

        preferred_str = (
            self.get_parameter("preferred_mode")
            .get_parameter_value()
            .string_value
        )
        preferred_mode = (
            FlightMode.CRUISE
            if preferred_str.lower() == "cruise"
            else FlightMode.HOVER
        )

        self.max_iterations: int = (
            self.get_parameter("max_iterations")
            .get_parameter_value()
            .integer_value
        )
        self.replan_on_obstacle: bool = (
            self.get_parameter("replan_on_obstacle")
            .get_parameter_value()
            .bool_value
        )
        self.emergency_replan_distance: float = self._param_float(
            "emergency_replan_distance"
        )

        # --- State variables --------------------------------------------------
        self.voxel_map = VoxelMap(sx, sy, sz, resolution=res)
        self.vehicle = vehicle
        self.planner = HybridAStarPlanner3D(
            self.voxel_map, vehicle=vehicle, preferred_mode=preferred_mode
        )

        self.current_pose: Optional[Tuple[float, float, float, float, float]] = None
        self.goal_pose: Optional[Tuple[float, float, float, float, float]] = None
        self.current_path: Optional[List[Tuple]] = None

        self.obstacle_detected: bool = False
        self.obstacle_distance: float = float("inf")
        self.avoidance_zone: str = ZONE_SAFE
        self.replanning_in_progress: bool = False

        # --- Subscribers ------------------------------------------------------
        self.create_subscription(Bool, "/obstacle_detected",
                                 self.obstacle_callback, 10)
        self.create_subscription(Float32, "/obstacle_distance",
                                 self.distance_callback, 10)
        self.create_subscription(String, "/avoidance_zone",
                                 self.zone_callback, 10)
        self.create_subscription(PoseStamped, "/goal_pose",
                                 self.goal_callback, 10)
        self.create_subscription(PoseStamped, "/current_pose",
                                 self.pose_callback, 10)
        # Obstacle-map subscribers (same as obstacle_map_bridge)
        self._pc_sub = self.create_subscription(
            PointCloud2, "/lidar/points", self._pointcloud_callback, 10
        )
        self._obs_pos_sub = self.create_subscription(
            Vector3Stamped, "/obstacle_position", self._obstacle_position_callback, 10
        )

        # --- Publishers -------------------------------------------------------
        self._path_pub = self.create_publisher(Path, "/planned_path", 10)
        self._status_pub = self.create_publisher(String, "/planner_status", 10)

        # --- Periodic replan timer (1 Hz) -------------------------------------
        self._replan_timer = self.create_timer(1.0, self.replan_timer_callback)

        self.get_logger().info("IntegratedPlannerNode ready.")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def obstacle_callback(self, msg: "Bool") -> None:
        """Handle an obstacle-detected notification.

        Triggers replanning when obstacle detection state changes to *True*
        and ``replan_on_obstacle`` is enabled.

        Args:
            msg: Boolean obstacle-detected flag.
        """
        was_detected = self.obstacle_detected
        self.obstacle_detected = bool(msg.data)

        if (
            self.obstacle_detected
            and not was_detected
            and self.replan_on_obstacle
        ):
            self.get_logger().info("Obstacle detected – triggering replan.")
            self.replan()

    def distance_callback(self, msg: "Float32") -> None:
        """Update the cached obstacle distance.

        Triggers emergency replanning when the distance drops below
        ``emergency_replan_distance``.

        Args:
            msg: Current obstacle distance in metres.
        """
        self.obstacle_distance = float(msg.data)

        if (
            self.obstacle_distance <= self.emergency_replan_distance
            and not self.replanning_in_progress
        ):
            self.get_logger().warning(
                f"Emergency distance {self.obstacle_distance:.2f} m – "
                "triggering emergency replan."
            )
            self.emergency_replan()

    def zone_callback(self, msg: "String") -> None:
        """Handle avoidance zone transitions.

        Applies zone-based replanning strategy:
        * CRITICAL → emergency hover + ascend
        * DANGER → emergency replan (fast)
        * WARNING → standard replan
        * CAUTION / SAFE → monitor only

        Args:
            msg: Zone name string.
        """
        zone = msg.data.upper().strip()
        self.avoidance_zone = zone

        if zone == ZONE_CRITICAL:
            self.get_logger().error("CRITICAL zone – emergency hover.")
            self.emergency_hover()
        elif zone == ZONE_DANGER:
            self.get_logger().warning("DANGER zone – emergency replan.")
            self.emergency_replan()
        elif zone == ZONE_WARNING:
            self.get_logger().info("WARNING zone – standard replan.")
            self.replan()
        # CAUTION and SAFE require no immediate action

    def goal_callback(self, msg: "PoseStamped") -> None:
        """Set a new goal pose and trigger replanning.

        Args:
            msg: New goal as a :class:`geometry_msgs.msg.PoseStamped`.
        """
        yaw = self.quaternion_to_yaw(msg.pose.orientation)
        self.goal_pose = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            yaw,
            0.0,
        )
        self.get_logger().info(
            f"New goal received: {self.goal_pose[:3]}"
        )
        self.replan()

    def pose_callback(self, msg: "PoseStamped") -> None:
        """Update the cached current pose.

        Args:
            msg: Current UAV pose as a :class:`geometry_msgs.msg.PoseStamped`.
        """
        yaw = self.quaternion_to_yaw(msg.pose.orientation)
        self.current_pose = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            yaw,
            0.0,
        )

    # ------------------------------------------------------------------
    # Obstacle-map callbacks (same as obstacle_map_bridge)
    # ------------------------------------------------------------------
    
    def _pointcloud_callback(self, msg: "PointCloud2") -> None:
        """Process an incoming LiDAR scan and update the voxel map.

        Args:
            msg: Incoming :class:`sensor_msgs.msg.PointCloud2` message.
        """
        try:
            points = self._pointcloud2_to_array(msg)
            if points.shape[0] > 0:
                self.voxel_map.update_from_pointcloud(points)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(
                f"_pointcloud_callback error: {exc}", throttle_duration_sec=5.0
            )
    
    def _obstacle_position_callback(self, msg: "Vector3Stamped") -> None:
        """Add a sphere obstacle at the detected position.

        The sphere radius is set to ``obstacle_inflation``.

        Args:
            msg: Obstacle position message.
        """
        try:
            radius = max(self._param_float("obstacle_inflation"), 0.5)
            self.voxel_map.add_sphere_obstacle(
                msg.vector.x, msg.vector.y, msg.vector.z, radius
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(
                f"_obstacle_position_callback error: {exc}",
                throttle_duration_sec=5.0,
            )
    
    @staticmethod
    def _pointcloud2_to_array(msg: "PointCloud2") -> np.ndarray:
        """Convert a PointCloud2 to a numpy array of XYZ points.

        Args:
            msg: ROS2 PointCloud2 message.

        Returns:
            (N, 3) float32 numpy array of (x, y, z) coordinates.
        """
        try:
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
    
    def replan_timer_callback(self) -> None:
        """Periodic (1 Hz) check: replan if an obstacle is active.

        Only triggers when replanning is not already in progress and a goal
        has been set.
        """
        if (
            self.obstacle_detected
            and self.goal_pose is not None
            and not self.replanning_in_progress
        ):
            self.replan()

    # ------------------------------------------------------------------
    # Planning methods
    # ------------------------------------------------------------------

    def replan(self) -> None:
        """Run normal replanning with ``max_iterations`` A* expansions.

        No-ops if ``current_pose`` or ``goal_pose`` is not set or if
        replanning is already in progress.
        """
        if self.current_pose is None or self.goal_pose is None:
            return
        if self.replanning_in_progress:
            return

        self.replanning_in_progress = True
        self.publish_status("Replanning...")
        try:
            path = self.planner.plan(
                self.current_pose, self.goal_pose,
                max_iter=self.max_iterations,
            )
            if path is not None:
                smoothed = HybridAStarPlanner3D.smooth_path(path)
                self.current_path = smoothed
                self.publish_path(smoothed)
                self.publish_status(f"Path found: {len(smoothed)} waypoints")
                self.get_logger().info(
                    f"Path planned: {len(smoothed)} waypoints."
                )
            else:
                self.publish_status("No path found – retaining previous path")
                self.get_logger().warning("Replan failed: no path found.")
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f"replan() exception: {exc}")
            self.publish_status(f"Replan error: {exc}")
        finally:
            self.replanning_in_progress = False

    def emergency_replan(self) -> None:
        """Fast replanning with a reduced iteration budget (20 000).

        Uses fewer iterations for lower latency at the expense of path
        optimality.
        """
        if self.current_pose is None or self.goal_pose is None:
            return
        if self.replanning_in_progress:
            return

        self.replanning_in_progress = True
        self.publish_status("Emergency replanning...")
        try:
            path = self.planner.plan(
                self.current_pose, self.goal_pose,
                max_iter=MAX_ITER_EMERGENCY,
            )
            if path is not None:
                self.current_path = path
                self.publish_path(path)
                self.publish_status(
                    f"Emergency path found: {len(path)} waypoints"
                )
            else:
                self.get_logger().warning(
                    "Emergency replan failed – triggering hover."
                )
                self.emergency_hover()
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f"emergency_replan() exception: {exc}")
            self.emergency_hover()
        finally:
            self.replanning_in_progress = False

    def emergency_hover(self) -> None:
        """Publish an emergency stop / hover-ascend path.

        Generates a short upward path from the current position to create
        vertical clearance, then publishes it as the planned path.
        """
        if self.current_pose is None:
            self.publish_status("EMERGENCY HOVER – no pose available")
            return

        x, y, z, yaw, pitch = self.current_pose
        ascend_z = min(
            z + 5.0, self.vehicle.max_altitude
        )
        hover_path = [
            (x, y, z, yaw, 0.0),
            (x, y, ascend_z, yaw, 0.0),
        ]
        self.current_path = hover_path
        self.publish_path(hover_path)
        self.publish_status(
            f"EMERGENCY HOVER: ascending to {ascend_z:.1f} m"
        )
        self.get_logger().error(
            f"Emergency hover: ascending from {z:.1f} m to {ascend_z:.1f} m"
        )

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def publish_path(
        self, waypoints: List[Tuple[float, float, float, float, float]]
    ) -> None:
        """Convert waypoints to :class:`nav_msgs.msg.Path` and publish.

        Args:
            waypoints: List of ``(x, y, z, yaw, pitch)`` tuples.
        """
        path_msg = Path()
        path_msg.header.frame_id = "map"

        for wp in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = wp[2]
            quat = self.yaw_to_quaternion(wp[3])
            pose.pose.orientation = quat
            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)

    def publish_status(self, status: str) -> None:
        """Publish a human-readable status string.

        Args:
            status: Status message to publish.
        """
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def quaternion_to_yaw(quat: "Quaternion") -> float:
        """Extract yaw (heading) from a :class:`geometry_msgs.msg.Quaternion`.

        Uses the standard ZYX Euler conversion.

        Args:
            quat: Input quaternion.

        Returns:
            Yaw angle in radians.
        """
        # siny_cosp = 2 * (w*z + x*y)
        # cosy_cosp = 1 - 2 * (y*y + z*z)
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> "Quaternion":
        """Convert a yaw angle to a :class:`geometry_msgs.msg.Quaternion`.

        Assumes zero roll and pitch.

        Args:
            yaw: Heading angle in radians.

        Returns:
            Quaternion representing a pure yaw rotation.
        """
        quat = Quaternion()
        quat.x = 0.0
        quat.y = 0.0
        quat.z = math.sin(yaw / 2.0)
        quat.w = math.cos(yaw / 2.0)
        return quat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        """Declare all ROS2 parameters with default values."""
        self.declare_parameter("map_size_x", 100.0)
        self.declare_parameter("map_size_y", 100.0)
        self.declare_parameter("map_size_z", 50.0)
        self.declare_parameter("map_resolution", 1.0)
        self.declare_parameter("vehicle_length", 1.8)
        self.declare_parameter("vehicle_width", 1.2)
        self.declare_parameter("vehicle_height", 0.6)
        self.declare_parameter("vehicle_inflation", 0.35)
        self.declare_parameter("vehicle_min_altitude", 5.0)
        self.declare_parameter("vehicle_max_altitude", 120.0)
        self.declare_parameter("min_turn_radius", 3.0)
        self.declare_parameter("hover_step", 1.0)
        self.declare_parameter("cruise_step", 2.0)
        self.declare_parameter("max_iterations", MAX_ITER_NORMAL)
        self.declare_parameter("preferred_mode", "hover")
        self.declare_parameter("replan_on_obstacle", True)
        self.declare_parameter("emergency_replan_distance", 3.0)
        self.declare_parameter("obstacle_inflation", 0.5)

    def _param_float(self, name: str) -> float:
        """Retrieve a float ROS2 parameter value by name.

        Args:
            name: Parameter name.

        Returns:
            Parameter value as a Python float.
        """
        return float(
            self.get_parameter(name).get_parameter_value().double_value
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    """ROS2 entry point for the integrated_planner_node."""
    if not _ROS_AVAILABLE:
        raise RuntimeError(
            "rclpy is not installed. "
            "Please source your ROS2 workspace before running this node."
        )
    rclpy.init(args=args)
    node = IntegratedPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
