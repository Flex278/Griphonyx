#!/usr/bin/env python3
"""
A* Global Path Planner Node for Griphonyx UAV Obstacle Avoidance.

Replaces the RRT* planner with the weighted A* pathfinding algorithm.

Architecture:
    /lidar/points            → obstacle_detector → /avoidance_command
                                                 → /obstacle_velocity
                                                 → /costmap/update_trigger → costmap_node
    /lidar/points            → costmap_node      → /costmap/grid
    /costmap/grid            → astar_node        → /planned_path
    /obstacle_detected, etc  → astar_node        (trigger replanning)

Subscribed Topics:
    /costmap/grid            (nav_msgs/OccupancyGrid) — from costmap_node
    /obstacle_detected       (std_msgs/Bool)          — trigger replanning
    /obstacle_distance       (std_msgs/Float32)        — closest obstacle distance
    /avoidance_command       (std_msgs/String)         — zone command from detector
    /obstacle_velocity       (geometry_msgs/Vector3Stamped) — Kalman velocity
    /vehicle/pose            (geometry_msgs/PoseStamped)    — UAV pose
    /planning/goal           (geometry_msgs/PoseStamped)    — navigation goal

Published Topics:
    /planned_path            (nav_msgs/Path)   — A*-generated global path
    /astar_planner/status    (std_msgs/String) — human-readable planner status
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Bool, Float32, String
from builtin_interfaces.msg import Time

from .astar_algorithm import astar_search, AStarResult
from .costmap_interface import CostmapInterface


class AStarPlannerNode(Node):
    """ROS 2 node that runs A* path planning over the local costmap.

    Integrates with the existing obstacle_detection and costmap packages,
    preserving all topic interfaces used by the avoidance system.
    """

    def __init__(self):
        super().__init__('astar_planner')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('grid_resolution', 0.5)
        self.declare_parameter('heuristic_weight', 1.0)
        self.declare_parameter('max_iterations', 10000)
        self.declare_parameter('planning_frequency', 2.0)
        self.declare_parameter('diagonal_movement', True)
        self.declare_parameter('cost_penalty_factor', 15.0)
        self.declare_parameter('lethal_cost_threshold', 70)
        self.declare_parameter('unknown_cell_cost', 70)
        self.declare_parameter('goal_tolerance_m', 3.0)
        self.declare_parameter('path_deviation_threshold_m', 5.0)
        self.declare_parameter('safety_margin', 3.0)
        self.declare_parameter('min_altitude_m', 5.0)
        self.declare_parameter('max_altitude_m', 120.0)

        self.heuristic_weight = self.get_parameter('heuristic_weight').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.planning_frequency = self.get_parameter('planning_frequency').value
        self.diagonal = self.get_parameter('diagonal_movement').value
        self.cost_penalty_factor = self.get_parameter('cost_penalty_factor').value
        self.lethal_threshold = self.get_parameter('lethal_cost_threshold').value
        self.goal_tolerance_m = self.get_parameter('goal_tolerance_m').value
        self.deviation_threshold = self.get_parameter('path_deviation_threshold_m').value
        self.min_alt = self.get_parameter('min_altitude_m').value
        self.max_alt = self.get_parameter('max_altitude_m').value

        # ── Costmap interface ──────────────────────────────────────────
        self.costmap_iface = CostmapInterface(lethal_threshold=self.lethal_threshold)

        # ── State ─────────────────────────────────────────────────────
        self.vehicle_pose: PoseStamped | None = None
        self.goal_pose: PoseStamped | None = None
        self.current_path: Path | None = None
        self.planner_state = 'IDLE'
        self.replan_requested = False
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.avoidance_zone = 'NORMAL_FLIGHT'

        # ── QoS ───────────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── Subscribers ───────────────────────────────────────────────
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/costmap/grid',
            self._costmap_cb, map_qos)

        self.obstacle_detected_sub = self.create_subscription(
            Bool, '/obstacle_detected',
            self._obstacle_detected_cb, 10)

        self.obstacle_distance_sub = self.create_subscription(
            Float32, '/obstacle_distance',
            self._obstacle_distance_cb, 10)

        self.avoidance_command_sub = self.create_subscription(
            String, '/avoidance_command',
            self._avoidance_command_cb, 10)

        self.obstacle_velocity_sub = self.create_subscription(
            Vector3Stamped, '/obstacle_velocity',
            self._obstacle_velocity_cb, 10)

        self.vehicle_pose_sub = self.create_subscription(
            PoseStamped, '/vehicle/pose',
            self._vehicle_pose_cb, 10)

        self.goal_sub = self.create_subscription(
            PoseStamped, '/planning/goal',
            self._goal_cb, 10)

        # ── Publishers ────────────────────────────────────────────────
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.status_pub = self.create_publisher(String, '/astar_planner/status', 10)

        # ── Timers ────────────────────────────────────────────────────
        self.planning_timer = self.create_timer(
            1.0 / self.planning_frequency, self._planning_tick)
        self.status_timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            f'A* Planner started (w={self.heuristic_weight}, '
            f'max_iter={self.max_iterations}, '
            f'lethal≥{self.lethal_threshold}, '
            f'freq={self.planning_frequency}Hz)')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        self.costmap_iface.update(msg)
        if self.replan_requested:
            self._run_planning()

    def _obstacle_detected_cb(self, msg: Bool) -> None:
        if msg.data and not self.obstacle_detected:
            self.replan_requested = True
        self.obstacle_detected = msg.data

    def _obstacle_distance_cb(self, msg: Float32) -> None:
        self.obstacle_distance = msg.data

    def _avoidance_command_cb(self, msg: String) -> None:
        self.avoidance_zone = msg.data
        if msg.data in ('EMERGENCY_HOVER', 'HARD_AVOID', 'REROUTE'):
            self.replan_requested = True

    def _obstacle_velocity_cb(self, msg: Vector3Stamped) -> None:
        pass

    def _vehicle_pose_cb(self, msg: PoseStamped) -> None:
        self.vehicle_pose = msg

    def _goal_cb(self, msg: PoseStamped) -> None:
        self.goal_pose = msg
        self.replan_requested = True
        self.planner_state = 'PLANNING'
        self.get_logger().info(
            f'New goal: ({msg.pose.position.x:.1f}, '
            f'{msg.pose.position.y:.1f}, '
            f'{msg.pose.position.z:.1f})')

    # ── Planning loop ─────────────────────────────────────────────────

    def _planning_tick(self) -> None:
        """Periodic planning tick."""
        needs_replan = self.replan_requested
        if not needs_replan and self.current_path is not None:
            needs_replan = self._check_path_deviation()

        if not needs_replan and self.planner_state == 'ACTIVE':
            if self._goal_reached():
                self.planner_state = 'IDLE'
                self.goal_pose = None
                self.current_path = None
                self.get_logger().info('Goal reached!')
                self._publish_empty_path()
            return

        if not needs_replan:
            return

        self._run_planning()

    def _run_planning(self) -> None:
        """Execute A* planning and publish result."""
        if self.vehicle_pose is None or self.goal_pose is None:
            return

        grid_np, grid_msg = self.costmap_iface.get_grid()
        if grid_np is None:
            self.planner_state = 'NO_PATH'
            self.get_logger().warn(
                'No costmap received — cannot plan',
                throttle_duration_sec=5.0)
            return

        self.replan_requested = False

        sx = self.vehicle_pose.pose.position.x
        sy = self.vehicle_pose.pose.position.y
        gx = self.goal_pose.pose.position.x
        gy = self.goal_pose.pose.position.y

        start_rc = self.costmap_iface.world_to_grid(sx, sy, grid_msg)
        goal_rc = self.costmap_iface.world_to_grid(gx, gy, grid_msg)

        if start_rc is None or goal_rc is None:
            self.planner_state = 'NO_PATH'
            self.get_logger().warn(
                'Start or goal is outside the costmap grid',
                throttle_duration_sec=2.0)
            return

        # Clamp start to free space (costmap inflation can cover the UAV)
        start_rc = self.costmap_iface.find_nearest_free(grid_np, start_rc)
        if start_rc is None:
            self.planner_state = 'NO_PATH'
            self.get_logger().warn('Cannot find free cell near start position')
            return

        result: AStarResult = astar_search(
            grid=grid_np,
            start=start_rc,
            goal=goal_rc,
            heuristic_weight=self.heuristic_weight,
            diagonal=self.diagonal,
            max_iterations=self.max_iterations,
            lethal_threshold=self.lethal_threshold,
            cost_penalty_factor=self.cost_penalty_factor)

        if result.success:
            path_msg = self._build_path_msg(result.path, grid_msg)
            self.path_pub.publish(path_msg)
            self.current_path = path_msg
            self.planner_state = 'ACTIVE'
            self.get_logger().info(
                f'A* path found: {len(result.path)} cells, '
                f'{result.nodes_expanded} expanded, '
                f'{result.elapsed_ms:.1f} ms, cost={result.cost:.1f}')
        else:
            self.planner_state = 'NO_PATH'
            self._publish_empty_path()
            self.get_logger().warn(
                f'A* failed after {result.nodes_expanded} expansions '
                f'({result.elapsed_ms:.1f} ms)')

    # ── Path helpers ──────────────────────────────────────────────────

    def _build_path_msg(self, grid_path: list, grid_msg: OccupancyGrid) -> Path:
        """Convert (row, col) grid path to nav_msgs/Path."""
        path_msg = Path()
        stamp = self.get_clock().now().to_msg()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = grid_msg.header.frame_id or 'map'

        goal_z = max(self.min_alt,
                     min(self.max_alt, self.goal_pose.pose.position.z
                         if self.goal_pose else self.min_alt))

        for i, (r, c) in enumerate(grid_path):
            wx, wy = self.costmap_iface.grid_to_world(r, c, grid_msg)

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = goal_z

            # Yaw toward next waypoint
            if i < len(grid_path) - 1:
                nr, nc = grid_path[i + 1]
                nwx, nwy = self.costmap_iface.grid_to_world(nr, nc, grid_msg)
                yaw = math.atan2(nwy - wy, nwx - wx)
            elif i > 0:
                pr, pc = grid_path[i - 1]
                pwx, pwy = self.costmap_iface.grid_to_world(pr, pc, grid_msg)
                yaw = math.atan2(wy - pwy, wx - pwx)
            else:
                yaw = 0.0

            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path_msg.poses.append(pose)

        return path_msg

    def _publish_empty_path(self) -> None:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        self.path_pub.publish(path_msg)

    # ── Monitoring helpers ────────────────────────────────────────────

    def _check_path_deviation(self) -> bool:
        if self.current_path is None or not self.current_path.poses:
            return False
        if self.vehicle_pose is None:
            return False

        px = self.vehicle_pose.pose.position.x
        py = self.vehicle_pose.pose.position.y
        min_dist = min(
            math.hypot(p.pose.position.x - px, p.pose.position.y - py)
            for p in self.current_path.poses)
        return min_dist > self.deviation_threshold

    def _goal_reached(self) -> bool:
        if self.vehicle_pose is None or self.goal_pose is None:
            return False
        dx = self.goal_pose.pose.position.x - self.vehicle_pose.pose.position.x
        dy = self.goal_pose.pose.position.y - self.vehicle_pose.pose.position.y
        dz = self.goal_pose.pose.position.z - self.vehicle_pose.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz) < self.goal_tolerance_m

    def _publish_status(self) -> None:
        status = {
            'state': self.planner_state,
            'obstacle_detected': self.obstacle_detected,
            'obstacle_distance_m': round(self.obstacle_distance, 2),
            'avoidance_zone': self.avoidance_zone,
            'has_costmap': self.costmap_iface._grid_msg is not None,
            'has_vehicle_pose': self.vehicle_pose is not None,
            'has_goal': self.goal_pose is not None,
            'path_length': (len(self.current_path.poses)
                            if self.current_path else 0),
        }
        if self.goal_pose is not None:
            status['goal_x'] = round(self.goal_pose.pose.position.x, 2)
            status['goal_y'] = round(self.goal_pose.pose.position.y, 2)
            status['goal_z'] = round(self.goal_pose.pose.position.z, 2)

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
