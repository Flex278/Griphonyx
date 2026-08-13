"""
DemoDriver – ROS2 node that feeds the integrated planner a synthetic scenario.

Publishes a start pose, a goal, and a LiDAR-like obstacle wall so the
integrated planning system (``obstacle_map_bridge`` + ``integrated_planner_node``)
produces a 3D path that :class:`px4_mavlink_bridge.PX4MavlinkBridge` then flies.

The scenario places the vehicle at (5, 5, 12), the goal at (25, 25, 15) and a
vertical wall at x=14 (y=10..19, z=6..19) that blocks the straight line.
"""

from __future__ import annotations

import array
import struct

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Vector3Stamped
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Bool, Float32, String

    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover – ROS2 not installed in CI
    _ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]


def build_pointcloud2(points, frame_id="map"):
    """Pack a list of ``(x, y, z)`` tuples into a PointCloud2 message."""
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_bigendian = False
    msg.is_dense = True
    msg.data = array.array(
        "B", b"".join(struct.pack("<3f", x, y, z) for (x, y, z) in points)
    )
    return msg


class DemoDriver(Node):
    """Synthetic demo driver publishing pose, goal, and obstacle data."""

    def __init__(self) -> None:
        """Initialise publishers, wall point cloud, and the periodic timer."""
        super().__init__("demo_driver")
        self.pose_pub = self.create_publisher(PoseStamped, "/current_pose", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.pc_pub = self.create_publisher(PointCloud2, "/lidar/points", 10)
        self.obs_pub = self.create_publisher(
            Vector3Stamped, "/obstacle_position", 10
        )
        self.det_pub = self.create_publisher(Bool, "/obstacle_detected", 10)
        self.dist_pub = self.create_publisher(Float32, "/obstacle_distance", 10)
        self.zone_pub = self.create_publisher(String, "/avoidance_zone", 10)

        # A compact vertical wall across the straight start -> goal line.
        pts = []
        for y in range(10, 20):
            for z in range(6, 20):
                pts.append((14.0, float(y), float(z)))
        self.cloud = build_pointcloud2(pts)

        self.timer = self.create_timer(0.5, self.timer_cb)
        self.ticks = 0
        self.get_logger().info(
            "DemoDriver ready (start=5,5,12 goal=25,25,15)."
        )

    def timer_cb(self) -> None:
        """Publish the demo pose, goal, and obstacle data every 0.5 s."""
        self.ticks += 1
        now = self.get_clock().now().to_msg()

        # Current pose (published continuously).
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = now
        p.pose.position.x = 5.0
        p.pose.position.y = 5.0
        p.pose.position.z = 12.0
        p.pose.orientation.w = 1.0
        self.pose_pub.publish(p)

        # Goal (published several times to guarantee the planner receives it).
        if 2 <= self.ticks <= 8:
            g = PoseStamped()
            g.header.frame_id = "map"
            g.header.stamp = now
            g.pose.position.x = 25.0
            g.pose.position.y = 25.0
            g.pose.position.z = 15.0
            g.pose.orientation.w = 1.0
            self.goal_pub.publish(g)

        # Obstacle wall (PointCloud2) + one small sphere off the direct line.
        self.pc_pub.publish(self.cloud)
        o = Vector3Stamped()
        o.header.frame_id = "map"
        o.header.stamp = now
        o.vector.x = 20.0
        o.vector.y = 20.0
        o.vector.z = 8.0
        self.obs_pub.publish(o)

        # Keep the planner in SAFE mode so it does not emergency-hover.
        d = Float32()
        d.data = 60.0
        self.dist_pub.publish(d)
        z = String()
        z.data = "SAFE"
        self.zone_pub.publish(z)


def main(args=None) -> None:
    """ROS2 entry point for the demo_driver node."""
    if not _ROS_AVAILABLE:
        raise RuntimeError(
            "rclpy is not installed. "
            "Please source your ROS2 workspace before running this node."
        )
    rclpy.init(args=args)
    node = DemoDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
