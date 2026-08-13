"""
PX4MavlinkBridge – ROS2 node bridging ``/planned_path`` to PX4 via MAVLink.

The integrated planner publishes a :class:`nav_msgs.msg.Path` in an ENU
"map" frame (x/y horizontal, z up).  This node converts those waypoints into
a PX4 mission, uploads it over MAVLink and commands the vehicle to fly it in
``AUTO`` (mission) mode.

Requires ``pymavlink`` (``pip install pymavlink``).  Connects to the PX4 SITL
external MAVLink port (default ``udpin:127.0.0.1:14540``) over the shared
host network.

Coordinate conversion
---------------------
PX4 missions only accept global (latitude/longitude) frames, so the local ENU
path is first shifted so that the *first* waypoint (the current pose) becomes
the origin, then projected onto a flat Earth around the PX4 home position::

    N = x - x0
    E = y - y0
    D = -z            (map z is altitude above ground)

The uploaded mission is a take-off to the first waypoint altitude followed by
one ``NAV_WAYPOINT`` per path pose and, optionally, a final ``NAV_LAND``.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Path
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped
    from visualization_msgs.msg import Marker

    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover – ROS2 not installed in CI
    _ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

try:
    from pymavlink import mavutil

    _MAVLINK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MAVLINK_AVAILABLE = False
    mavutil = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# MAVLink constants
# ---------------------------------------------------------------------------
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_MISSION_START = 300
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_GET_HOME_POSITION = 410
MAV_CMD_SET_MESSAGE_INTERVAL = 511

MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_MISSION_ACCEPTED = 0

PX4_CUSTOM_MAIN_MODE_AUTO = 4
PX4_CUSTOM_SUB_MODE_AUTO_MISSION = 4

# PX4 packs custom flight modes as ``(main_mode << 16) | (sub_mode << 24)``.
# See ``src/modules/commander/px4_custom_mode.h`` (union px4_custom_mode).
PX4_CUSTOM_MODE_AUTO_MISSION = (
    (PX4_CUSTOM_MAIN_MODE_AUTO << 16)
    | (PX4_CUSTOM_SUB_MODE_AUTO_MISSION << 24)
)

# MAVLink message IDs used when configuring stream rates.
MAVLINK_MSG_ID_LOCAL_POSITION_NED = 32
MAVLINK_MSG_ID_GLOBAL_POSITION_INT = 33
MAVLINK_MSG_ID_STATUSTEXT = 253

EARTH_RADIUS_M = 6378137.0


# ---------------------------------------------------------------------------
# Pure conversion helpers (unit-tested, no ROS2/MAVLink dependency)
# ---------------------------------------------------------------------------

def enu_to_ned(
    waypoints: List[Tuple[float, float, float]],
    origin: Tuple[float, float],
) -> List[Tuple[float, float, float]]:
    """Convert ENU ``(x, y, z)`` waypoints to local-NED ``(N, E, D)``.

    Args:
        waypoints: List of ``(x, y, z)`` in an ENU "map" frame (z up).
        origin: ``(x0, y0)`` of the reference point (typically the first
            waypoint).  The returned NED coordinates are relative to it.

    Returns:
        List of ``(N, E, D)`` with ``D = -z`` (altitude expressed as down).
    """
    x0, y0 = origin
    return [(x - x0, y - y0, -z) for (x, y, z) in waypoints]


def ned_to_global(
    ned_points: List[Tuple[float, float, float]],
    home: Tuple[float, float, float],
) -> List[Tuple[float, float, float]]:
    """Project local-NED points onto a flat Earth around ``home``.

    Args:
        ned_points: List of ``(N, E, D)`` metres relative to home.
        home: ``(lat_deg, lon_deg, alt_amsl_m)`` of the vehicle home position.

    Returns:
        List of ``(lat_deg, lon_deg, alt_rel_m)`` where ``alt_rel_m`` is the
        altitude relative to home (positive up, ``-D``).
    """
    lat0, lon0, _ = home
    m_per_deg_lat = math.pi * EARTH_RADIUS_M / 180.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))

    out: List[Tuple[float, float, float]] = []
    for (north, east, down) in ned_points:
        lat = lat0 + north / m_per_deg_lat
        lon = lon0 + east / m_per_deg_lon
        alt = -down
        out.append((lat, lon, alt))
    return out


def build_mission_items(
    waypoints: List[Tuple[float, float, float]],
    home: Tuple[float, float, float],
    takeoff_alt: float,
    cruise_speed: float = 0.0,
    acceptance_radius: float = 1.0,
    add_land: bool = True,
) -> List[Dict]:
    """Build PX4 mission items (as plain dicts) from ENU waypoints.

    Args:
        waypoints: List of ``(x, y, z)`` ENU waypoints (z up, metres).
        home: ``(lat_deg, lon_deg, alt_amsl_m)`` home position.
        takeoff_alt: Take-off altitude above home in metres.
        cruise_speed: Ground speed in m/s; ``0`` keeps the PX4 default.
        acceptance_radius: Waypoint acceptance radius in metres.
        add_land: Append a final ``NAV_LAND`` at the last waypoint.

    Returns:
        List of mission-item dicts with keys ``frame``, ``cmd``, ``current``,
        ``autocontinue``, ``p1..p4``, ``x``, ``y``, ``z``.
    """
    if not waypoints:
        raise ValueError("Cannot build a mission from an empty path")

    x0, y0 = waypoints[0][0], waypoints[0][1]
    ned = enu_to_ned(waypoints, (x0, y0))
    global_pts = ned_to_global(ned, home)

    lat0, lon0, _ = home
    items: List[Dict] = []

    # 1) Take-off to the first waypoint altitude.
    items.append({
        "frame": MAV_FRAME_GLOBAL_RELATIVE_ALT,
        "cmd": MAV_CMD_NAV_TAKEOFF,
        "current": 0,
        "autocontinue": 1,
        "p1": 0.0,             # minimum pitch (rotary wing: unused)
        "p2": 0.0,             # unused
        "p3": 0.0,             # unused
        "p4": float("nan"),    # yaw: auto
        "x": lat0,
        "y": lon0,
        "z": takeoff_alt,
    })

    # 2) Optional cruise-speed command.
    if cruise_speed > 0.0:
        items.append({
            "frame": MAV_FRAME_GLOBAL_RELATIVE_ALT,
            "cmd": MAV_CMD_DO_CHANGE_SPEED,
            "current": 0,
            "autocontinue": 1,
            "p1": 1.0,             # speed type: groundspeed
            "p2": cruise_speed,    # m/s
            "p3": -1.0,            # throttle: no change
            "p4": 0.0,             # unused
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        })

    # 3) One NAV_WAYPOINT per path pose.
    for (lat, lon, alt) in global_pts:
        items.append({
            "frame": MAV_FRAME_GLOBAL_RELATIVE_ALT,
            "cmd": MAV_CMD_NAV_WAYPOINT,
            "current": 0,
            "autocontinue": 1,
            "p1": 0.0,                 # hold time (s)
            "p2": acceptance_radius,   # acceptance radius (m)
            "p3": 0.0,                 # pass radius: default
            "p4": float("nan"),        # yaw: auto (point to next waypoint)
            "x": lat,
            "y": lon,
            "z": alt,
        })

    # 4) Optional landing at the final waypoint.
    if add_land:
        last_lat, last_lon, _ = global_pts[-1]
        items.append({
            "frame": MAV_FRAME_GLOBAL_RELATIVE_ALT,
            "cmd": MAV_CMD_NAV_LAND,
            "current": 0,
            "autocontinue": 1,
            "p1": 0.0,             # abort altitude: use default
            "p2": 0.0,             # precision landing: disabled
            "p3": 0.0,             # unused
            "p4": float("nan"),    # yaw: auto
            "x": last_lat,
            "y": last_lon,
            "z": 0.0,
        })

    return items


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class PX4MavlinkBridge(Node):
    """ROS2 node that flies ``/planned_path`` waypoints on PX4.

    Subscribes to :data:`/planned_path` (``nav_msgs/Path``) and, on the first
    valid path, uploads a matching mission to PX4, arms the vehicle, switches
    to ``AUTO`` and starts the mission.  MAVLink I/O runs on a background
    thread so the ROS2 executor stays responsive.

    ROS2 Parameters:
        connection_url (str): pymavlink connection string
            (default ``udpin:127.0.0.1:14540``).
        target_system (int): MAVLink system id to command (default 1).
        target_component (int): MAVLink component id (default 1).
        takeoff_altitude (float): Take-off altitude override in metres;
            ``0`` uses the first waypoint altitude (default 0).
        cruise_speed (float): Mission ground speed in m/s; ``0`` = default.
        acceptance_radius (float): Waypoint acceptance radius (default 1.0).
        add_land (bool): Append a ``NAV_LAND`` at the end (default True).
        fly_once (bool): Fly only the first received path (default True).
        heartbeat_timeout (float): Heartbeat wait timeout in seconds (30).
    """

    def __init__(self) -> None:
        """Initialise node, subscriber, MAVLink thread, and status timer."""
        super().__init__("px4_mavlink_bridge")

        # --- Parameters ----------------------------------------------
        self.declare_parameter("connection_url", "udpin:127.0.0.1:14540")
        self.declare_parameter("target_system", 1)
        self.declare_parameter("target_component", 1)
        self.declare_parameter("takeoff_altitude", 0.0)
        self.declare_parameter("cruise_speed", 0.0)
        self.declare_parameter("acceptance_radius", 1.0)
        self.declare_parameter("add_land", True)
        self.declare_parameter("fly_once", True)
        self.declare_parameter("heartbeat_timeout", 30.0)

        self.connection_url = (
            self.get_parameter("connection_url").get_parameter_value().string_value
        )
        self.target_system = (
            self.get_parameter("target_system").get_parameter_value().integer_value
        )
        self.target_component = (
            self.get_parameter("target_component").get_parameter_value().integer_value
        )
        self.takeoff_altitude = (
            self.get_parameter("takeoff_altitude").get_parameter_value().double_value
        )
        self.cruise_speed = (
            self.get_parameter("cruise_speed").get_parameter_value().double_value
        )
        self.acceptance_radius = (
            self.get_parameter("acceptance_radius").get_parameter_value().double_value
        )
        self.add_land = (
            self.get_parameter("add_land").get_parameter_value().bool_value
        )
        self.fly_once = (
            self.get_parameter("fly_once").get_parameter_value().bool_value
        )
        self.heartbeat_timeout = (
            self.get_parameter("heartbeat_timeout").get_parameter_value().double_value
        )

        # --- State ---------------------------------------------------
        self._path_queue: "queue.Queue" = queue.Queue(maxsize=1)
        self._status_lock = threading.Lock()
        self._status = "waiting for path"
        self._flown = False

        # --- ROS2 interfaces ------------------------------------------
        self._path_sub = self.create_subscription(
            Path, "/planned_path", self._path_callback, 10
        )
        self._status_pub = self.create_publisher(String, "/bridge_status", 10)
        self._status_timer = self.create_timer(0.5, self._publish_status)

        # Drone visualisation for RViz (the *real* PX4 pose, not the
        # synthetic static ``/current_pose`` published by the demo driver).
        self._drone_pose_pub = self.create_publisher(
            PoseStamped, "/px4_drone_pose", 10
        )
        self._drone_marker_pub = self.create_publisher(Marker, "/drone_marker", 10)
        self._drone_trail_pub = self.create_publisher(Path, "/drone_trail", 10)

        # Origin of the planner "map" frame (first waypoint); used to map
        # PX4's LOCAL_POSITION_NED telemetry back into the map frame.
        self._path_origin: Tuple[float, float] = (0.0, 0.0)
        self._trail_points: List[Tuple[float, float, float]] = []

        # --- MAVLink background thread ---------------------------------
        self._mav_thread = threading.Thread(
            target=self._mavlink_loop, name="px4-mavlink", daemon=True
        )
        self._mav_thread.start()

        self.get_logger().info(
            f"PX4MavlinkBridge ready (MAVLink {self.connection_url})."
        )
        if not _MAVLINK_AVAILABLE:
            self.get_logger().error(
                "pymavlink is not installed - run: pip install pymavlink"
            )

    # ------------------------------------------------------------------
    # ROS2 callbacks (main thread)
    # ------------------------------------------------------------------

    def _path_callback(self, msg: "Path") -> None:
        """Store the latest planned path for the MAVLink thread.

        Args:
            msg: Incoming :class:`nav_msgs.msg.Path`.
        """
        waypoints = [
            (p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in msg.poses
        ]
        if len(waypoints) < 2:
            self.get_logger().warning(
                "Ignoring path with fewer than 2 waypoints.",
                throttle_duration_sec=5.0,
            )
            return

        # Remember the planner-frame origin (first waypoint) so the PX4
        # local-NED telemetry can be mapped back into the "map" frame.
        self._path_origin = (waypoints[0][0], waypoints[0][1])
        self._trail_points = []

        # Keep only the newest path in the single-slot queue.
        try:
            self._path_queue.put_nowait(waypoints)
        except queue.Full:
            try:
                self._path_queue.get_nowait()
                self._path_queue.put_nowait(waypoints)
            except queue.Empty:  # pragma: no cover - race, harmless
                pass

    def _publish_status(self) -> None:
        """Publish the current bridge status string (thread-safe)."""
        with self._status_lock:
            status = self._status
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _set_status(self, status: str) -> None:
        """Update the status string under lock."""
        with self._status_lock:
            self._status = status
        self.get_logger().info(status)

    # ------------------------------------------------------------------
    # MAVLink background thread
    # ------------------------------------------------------------------

    def _mavlink_loop(self) -> None:
        """Background thread: connect, wait for paths, and fly them."""
        if not _MAVLINK_AVAILABLE:
            return

        self._set_status(f"connecting MAVLink {self.connection_url}")
        try:
            self.master = mavutil.mavlink_connection(self.connection_url)
        except Exception as exc:  # pylint: disable=broad-except
            self._set_status(f"MAVLink connection error: {exc}")
            return

        self._set_status("waiting for PX4 heartbeat")
        heartbeat = self.master.wait_heartbeat(timeout=self.heartbeat_timeout)
        if heartbeat is None:
            self._set_status(
                f"no heartbeat within {self.heartbeat_timeout:.0f}s "
                f"(is PX4 SITL running on {self.connection_url}?)"
            )
            return

        self.target_system = heartbeat.get_srcSystem()
        self.target_component = heartbeat.get_srcComponent()
        self._set_status(
            f"heartbeat from system {self.target_system} "
            f"component {self.target_component}"
        )
        self._request_streams()

        while rclpy.ok():
            self._poll_telemetry()
            try:
                waypoints = self._path_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._flown and self.fly_once:
                self._set_status("ignoring new path (fly_once=True)")
                continue

            try:
                self._fly_path(waypoints)
                self._flown = True
            except Exception as exc:  # pylint: disable=broad-except
                self._set_status(f"flight failed: {exc}")

    # ------------------------------------------------------------------
    # MAVLink helpers (background thread)
    # ------------------------------------------------------------------

    def _request_streams(self) -> None:
        """Request the telemetry streams used by the bridge."""
        for msg_id, hz in (
            (MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20.0),
            (MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2.0),
            (MAVLINK_MSG_ID_STATUSTEXT, 1.0),
        ):
            interval_us = int(1e6 / hz)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                interval_us,
                0, 0, 0, 0, 0,
            )

    def _poll_telemetry(self) -> None:
        """Drain MAVLink telemetry and publish the drone pose for RViz.

        Reads one ``LOCAL_POSITION_NED`` message (non-blocking) and maps it
        from PX4's local NED frame into the planner's ENU "map" frame.  The
        first waypoint of the current path is the map-frame origin, so::

            map_x = x0 + N
            map_y = y0 + E
            map_z = -D

        Publishes a red marker, a :class:`PoseStamped` and an accumulating
        trail :class:`Path` so RViz can show the vehicle moving.
        """
        if not hasattr(self, "master"):
            return
        msg = self.master.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        if msg is None:
            return

        x0, y0 = self._path_origin
        x = x0 + msg.x
        y = y0 + msg.y
        z = -msg.z

        now = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = now
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self._drone_pose_pub.publish(pose)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = now
        marker.ns = "px4_drone"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.2
        marker.scale.y = 1.2
        marker.scale.z = 0.4
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 1
        self._drone_marker_pub.publish(marker)

        self._trail_points.append((x, y, z))
        trail = Path()
        trail.header.frame_id = "map"
        trail.header.stamp = now
        trail.poses = []
        for (px, py, pz) in self._trail_points:
            p = PoseStamped()
            p.header.frame_id = "map"
            p.header.stamp = now
            p.pose.position.x = px
            p.pose.position.y = py
            p.pose.position.z = pz
            p.pose.orientation.w = 1.0
            trail.poses.append(p)
        self._drone_trail_pub.publish(trail)

    def _get_home(self, timeout: float = 15.0) -> Tuple[float, float, float]:
        """Return the PX4 home position as ``(lat, lon, alt_amsl)``."""
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            MAV_CMD_GET_HOME_POSITION,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(
                type=["HOME_POSITION", "GLOBAL_POSITION_INT"],
                blocking=True,
                timeout=1.0,
            )
            if msg is None:
                continue
            if msg.get_type() == "HOME_POSITION":
                return (
                    msg.latitude * 1e-7,
                    msg.longitude * 1e-7,
                    msg.altitude * 1e-3,
                )
            # GLOBAL_POSITION_INT fallback.
            return msg.lat * 1e-7, msg.lon * 1e-7, msg.alt * 1e-3

        raise TimeoutError("could not obtain PX4 home position")

    def _upload_mission(self, items: List[Dict]) -> None:
        """Upload mission items using the MAVLink mission protocol.

        PX4 runs its mission manager in ``_int_mode`` and therefore requests
        items with ``MISSION_REQUEST_INT``; the bridge answers with
        ``MISSION_ITEM_INT`` (lat/lon scaled by 1e7) in that case.  It still
        falls back to the float ``MISSION_ITEM`` form for GCS implementations
        that use ``MISSION_REQUEST``.
        """
        mav = self.master.mav
        ts, tc = self.target_system, self.target_component
        count = len(items)

        self._set_status(f"uploading {count} mission items")
        mav.mission_count_send(ts, tc, count)

        sent = set()
        while True:
            msg = self.master.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"],
                blocking=True,
                timeout=10.0,
            )
            if msg is None:
                raise TimeoutError("mission upload timed out")

            msg_type = msg.get_type()

            if msg_type == "MISSION_ACK":
                if msg.type == MAV_MISSION_ACCEPTED:
                    self._set_status("mission accepted")
                    return
                raise RuntimeError(
                    f"mission upload rejected (MAV_MISSION_RESULT={msg.type})"
                )

            # MISSION_REQUEST(_INT): send the requested item (handles
            # re-requests).
            seq = msg.seq
            if seq in sent or not (0 <= seq < count):
                continue
            it = items[seq]
            if msg_type == "MISSION_REQUEST_INT":
                # Integer frame: lat/lon are int32 scaled by 1e7.
                mav.mission_item_int_send(
                    ts, tc, seq,
                    it["frame"], it["cmd"], it["current"], it["autocontinue"],
                    it["p1"], it["p2"], it["p3"], it["p4"],
                    int(round(it["x"] * 1e7)),
                    int(round(it["y"] * 1e7)),
                    it["z"],
                )
            else:
                mav.mission_item_send(
                    ts, tc, seq,
                    it["frame"], it["cmd"], it["current"], it["autocontinue"],
                    it["p1"], it["p2"], it["p3"], it["p4"],
                    it["x"], it["y"], it["z"],
                )
            sent.add(seq)

    def _arm(self, timeout: float = 10.0) -> bool:
        """Arm the vehicle and wait for the armed heartbeat flag."""
        self._set_status("arming")
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            hb = self.master.recv_match(
                type="HEARTBEAT", blocking=True, timeout=1.0
            )
            if hb is not None and (hb.base_mode & MAV_MODE_FLAG_SAFETY_ARMED):
                self._set_status("armed")
                return True
        return False

    def _set_mode_auto(self) -> None:
        """Switch the vehicle to AUTO (mission) mode."""
        self._set_status("switching to AUTO mode")
        self.master.mav.set_mode_send(
            self.target_system,
            MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            PX4_CUSTOM_MODE_AUTO_MISSION,
        )

    def _start_mission(self) -> None:
        """Start the uploaded mission from item 0."""
        self._set_status("starting mission")
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            MAV_CMD_MISSION_START,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    # ------------------------------------------------------------------
    # Flight orchestration (background thread)
    # ------------------------------------------------------------------

    def _fly_path(self, waypoints: List[Tuple[float, float, float]]) -> None:
        """Convert a path to a mission and fly it.

        Args:
            waypoints: List of ``(x, y, z)`` ENU waypoints.
        """
        self._set_status(f"planning flight for {len(waypoints)} waypoints")

        home = self._get_home()

        takeoff_alt = self.takeoff_altitude
        if takeoff_alt <= 0.0:
            takeoff_alt = waypoints[0][2]
        if takeoff_alt < 1.0:
            self.get_logger().warning(
                f"Take-off altitude {takeoff_alt:.2f} m is very low."
            )

        items = build_mission_items(
            waypoints,
            home,
            takeoff_alt=takeoff_alt,
            cruise_speed=self.cruise_speed,
            acceptance_radius=self.acceptance_radius,
            add_land=self.add_land,
        )

        self._upload_mission(items)

        if not self._arm():
            self._set_status("arming failed - vehicle not armed")
            return

        # Small pause so PX4 latches the armed state before the mode switch.
        time.sleep(1.0)
        self._set_mode_auto()
        time.sleep(1.0)
        self._start_mission()

        self._set_status(
            f"mission started: take-off to {takeoff_alt:.1f} m, "
            f"then {len(waypoints)} waypoints"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    """ROS2 entry point for the px4_mavlink_bridge node."""
    if not _ROS_AVAILABLE:
        raise RuntimeError(
            "rclpy is not installed. "
            "Please source your ROS2 workspace before running this node."
        )
    rclpy.init(args=args)
    node = PX4MavlinkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
