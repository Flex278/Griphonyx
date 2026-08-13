#!/usr/bin/env python3
"""
3-D / 2-D Costmap Node for Griphonyx UAV Obstacle Avoidance

Builds and maintains a local costmap around the UAV from LiDAR point-cloud data,
Kalman-tracked obstacle positions/velocities, and zone-based inflation radii.

Architecture
────────────
  Inputs:
    /lidar/points           (PointCloud2)    – raw sensor data
    /obstacle_velocity      (Vector3Stamped) – velocity of closest tracked obstacle
    /avoidance_command      (String)         – current zone (drives inflation radius)
    /costmap/update_trigger (Bool)           – pulse from obstacle_detector to rebuild
    /vehicle/pose           (PoseStamped)    – UAV pose (grid follows the UAV)

  Outputs:
    /costmap/grid           (OccupancyGrid)  – 2-D nav_msgs costmap (XY slice at UAV alt)
    /costmap/markers        (MarkerArray)    – RViz visualisation

Costmap model
─────────────
  Cell cost  0  : free
  Cell cost  1–69 : inflated / near obstacle (scales with proximity)
  Cell cost  70–99: danger zone (configurable lethal threshold in RRT*)
  Cell cost  100 : lethal (inside obstacle footprint)

  Inflation is Gaussian:  cost = 100 * exp(-0.5 * (d / sigma)^2)
  where sigma is chosen per zone:
    CRITICAL / DANGER   → sigma = obstacle_clearance * 1.0
    WARNING             → sigma = obstacle_clearance * 1.5
    CAUTION / SAFE      → sigma = obstacle_clearance * 2.0

  Dynamic obstacles (non-zero velocity) receive an additional forward-projection
  of their expected position over `velocity_horizon` seconds, with cost blended
  proportionally to elapsed projection time.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
from visualization_msgs.msg import MarkerArray, Marker
from builtin_interfaces.msg import Duration

import numpy as np
import struct
from typing import Optional


# ---------------------------------------------------------------------------
# Zone → sigma multiplier mapping
# ---------------------------------------------------------------------------
ZONE_SIGMA_SCALE = {
    'EMERGENCY_HOVER': 1.0,
    'HARD_AVOID':      1.0,
    'REROUTE':         1.5,
    'ADJUST_HEADING':  2.0,
    'NORMAL_FLIGHT':   2.0,
}


class CostmapNode(Node):
    """
    ROS 2 node that builds a local 2-D OccupancyGrid costmap from LiDAR
    and tracked obstacle data.

    The grid is centred on the UAV's current XY position and published
    whenever the obstacle detector triggers an update (or on a periodic timer).
    """

    def __init__(self):
        super().__init__('costmap_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('grid_size_m',       60.0)   # metres per side
        self.declare_parameter('resolution',         0.5)   # metres per cell
        self.declare_parameter('obstacle_clearance', 3.0)   # base inflation sigma (m)
        self.declare_parameter('max_cost',         100)     # lethal cost value
        self.declare_parameter('publish_rate',       2.0)   # Hz (background refresh)
        self.declare_parameter('velocity_horizon',   3.0)   # seconds for dynamic projection
        self.declare_parameter('lidar_topic',  '/lidar/points')

        p = self.get_parameter
        self.grid_size_m        = p('grid_size_m').value
        self.resolution         = p('resolution').value
        self.obstacle_clearance = p('obstacle_clearance').value
        self.max_cost           = p('max_cost').value
        self.publish_rate       = p('publish_rate').value
        self.velocity_horizon   = p('velocity_horizon').value
        self.lidar_topic        = p('lidar_topic').value

        # Grid dimensions
        self.n_cells = int(self.grid_size_m / self.resolution)

        # ── State ─────────────────────────────────────────────────────
        self.uav_position      : np.ndarray = np.zeros(3)
        self.obstacle_points   : Optional[np.ndarray] = None
        self.obstacle_velocity : np.ndarray = np.zeros(3)
        self.avoidance_zone    : str        = 'NORMAL_FLIGHT'

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(PointCloud2,    self.lidar_topic,          self._lidar_cb,    10)
        self.create_subscription(PoseStamped,    '/vehicle/pose',           self._pose_cb,     10)
        self.create_subscription(Vector3Stamped, '/obstacle_velocity',      self._velocity_cb, 10)
        self.create_subscription(String,         '/avoidance_command',      self._zone_cb,     10)
        self.create_subscription(Bool,           '/costmap/update_trigger', self._trigger_cb,  10)

        # ── Publishers ────────────────────────────────────────────────
        self.grid_pub    = self.create_publisher(OccupancyGrid, '/costmap/grid',    10)
        self.marker_pub  = self.create_publisher(MarkerArray,   '/costmap/markers', 10)

        # ── Background publish timer ──────────────────────────────────
        self.create_timer(1.0 / self.publish_rate, self._publish_costmap)

        self.get_logger().info(
            f'Costmap Node initialized — {self.n_cells}×{self.n_cells} cells '
            f'@ {self.resolution} m/cell ({self.grid_size_m} m grid)'
        )

    # ──────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ──────────────────────────────────────────────────────────────────
    def _lidar_cb(self, msg: PointCloud2):
        self.obstacle_points = self._parse_pointcloud(msg)

    def _pose_cb(self, msg: PoseStamped):
        self.uav_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def _velocity_cb(self, msg: Vector3Stamped):
        self.obstacle_velocity = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def _zone_cb(self, msg: String):
        self.avoidance_zone = msg.data

    def _trigger_cb(self, msg: Bool):
        if msg.data:
            self._publish_costmap()

    # ──────────────────────────────────────────────────────────────────
    # Costmap builder
    # ──────────────────────────────────────────────────────────────────
    def _build_grid(self) -> np.ndarray:
        """
        Build an N×N float32 cost grid centred on the UAV's XY position.

        Returns:
            N×N array with values in [0, 100].
        """
        grid = np.zeros((self.n_cells, self.n_cells), dtype=np.float32)

        if self.obstacle_points is None or len(self.obstacle_points) == 0:
            return grid

        # Sigma based on current avoidance zone
        sigma_scale = ZONE_SIGMA_SCALE.get(self.avoidance_zone, 2.0)
        sigma       = self.obstacle_clearance * sigma_scale

        # Origin of the grid in world coords (bottom-left corner)
        origin_x = self.uav_position[0] - self.grid_size_m / 2.0
        origin_y = self.uav_position[1] - self.grid_size_m / 2.0

        # Build cell-centre coordinate arrays for vectorised distance calc
        cols = np.arange(self.n_cells)
        rows = np.arange(self.n_cells)
        cell_x = origin_x + (cols + 0.5) * self.resolution   # shape (n_cells,)
        cell_y = origin_y + (rows + 0.5) * self.resolution   # shape (n_cells,)
        # Meshgrid: shape (n_cells, n_cells)
        grid_xx, grid_yy = np.meshgrid(cell_x, cell_y)

        # Project obstacle velocity forward for dynamic obstacles
        obs_pts = self.obstacle_points[:, :2].copy()  # XY only
        speed   = np.linalg.norm(self.obstacle_velocity[:2])
        if speed > 0.1:
            # Add ghost points along projected trajectory
            for t_step in np.linspace(0, self.velocity_horizon, num=5):
                proj = obs_pts + self.obstacle_velocity[:2] * t_step
                # Blend cost by remaining fraction (closer to now = higher cost)
                frac = 1.0 - t_step / self.velocity_horizon
                grid += frac * self._gaussian_layer(grid_xx, grid_yy, proj, sigma)
        else:
            grid += self._gaussian_layer(grid_xx, grid_yy, obs_pts, sigma)

        # Clamp to [0, max_cost]
        np.clip(grid, 0, self.max_cost, out=grid)
        return grid

    @staticmethod
    def _gaussian_layer(
        grid_xx: np.ndarray,
        grid_yy: np.ndarray,
        obstacle_pts: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Accumulate Gaussian inflation from all obstacle points into a cost layer.

        cost_cell = Σ_obstacles  100 * exp(-0.5 * (dist/sigma)^2)
        """
        layer = np.zeros_like(grid_xx, dtype=np.float32)
        for pt in obstacle_pts:
            dx   = grid_xx - pt[0]
            dy   = grid_yy - pt[1]
            dist = np.sqrt(dx * dx + dy * dy)
            layer += 100.0 * np.exp(-0.5 * (dist / max(sigma, 1e-6)) ** 2)
        return layer

    # ──────────────────────────────────────────────────────────────────
    # Publisher
    # ──────────────────────────────────────────────────────────────────
    def _publish_costmap(self):
        grid_data = self._build_grid()

        # ── OccupancyGrid ─────────────────────────────────────────────
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = float(self.resolution)
        msg.info.width      = self.n_cells
        msg.info.height     = self.n_cells
        msg.info.origin.position.x = float(
            self.uav_position[0] - self.grid_size_m / 2.0
        )
        msg.info.origin.position.y = float(
            self.uav_position[1] - self.grid_size_m / 2.0
        )
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # OccupancyGrid uses int8 in row-major order
        msg.data = grid_data.astype(np.int8).flatten().tolist()
        self.grid_pub.publish(msg)

        # ── RViz markers (one cube per lethal cell) ───────────────────
        self._publish_markers(grid_data, msg.info)

        self.get_logger().debug(
            f'Costmap published — zone={self.avoidance_zone}  '
            f'max_cost={grid_data.max():.0f}'
        )

    def _publish_markers(self, grid_data: np.ndarray, info):
        """Publish a MarkerArray of coloured cubes for RViz visualisation."""
        marker_array = MarkerArray()
        stamp        = self.get_clock().now().to_msg()
        lethal_mask  = grid_data >= 70   # only visualise danger+ cells

        rows, cols = np.where(lethal_mask)
        # Batch into a single CUBE_LIST marker for efficiency
        if len(rows) == 0:
            return

        marker = Marker()
        marker.header.stamp    = stamp
        marker.header.frame_id = 'map'
        marker.ns              = 'costmap'
        marker.id              = 0
        marker.type            = Marker.CUBE_LIST
        marker.action          = Marker.ADD
        marker.scale.x         = float(self.resolution)
        marker.scale.y         = float(self.resolution)
        marker.scale.z         = 0.3
        marker.lifetime        = Duration(sec=2, nanosec=0)

        origin_x = self.uav_position[0] - self.grid_size_m / 2.0
        origin_y = self.uav_position[1] - self.grid_size_m / 2.0

        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        for r, c in zip(rows.tolist(), cols.tolist()):
            cost = float(grid_data[r, c])
            p    = Point()
            p.x  = origin_x + (c + 0.5) * self.resolution
            p.y  = origin_y + (r + 0.5) * self.resolution
            p.z  = float(self.uav_position[2])
            marker.points.append(p)

            # Colour: green (low cost) → red (lethal)
            t = min(cost / 100.0, 1.0)
            colour       = ColorRGBA()
            colour.r     = t
            colour.g     = 1.0 - t
            colour.b     = 0.0
            colour.a     = 0.6
            marker.colors.append(colour)

        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    # ──────────────────────────────────────────────────────────────────
    # PointCloud2 parser
    # ──────────────────────────────────────────────────────────────────
    def _parse_pointcloud(self, msg: PointCloud2) -> Optional[np.ndarray]:
        point_step = msg.point_step
        num_points = len(msg.data) // point_step
        points = []
        for i in range(num_points):
            offset = i * point_step
            x = struct.unpack_from('f', msg.data, offset + 0)[0]
            y = struct.unpack_from('f', msg.data, offset + 4)[0]
            z = struct.unpack_from('f', msg.data, offset + 8)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append([x, y, z])
        return np.array(points) if points else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    try:
        node = CostmapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in costmap node: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
