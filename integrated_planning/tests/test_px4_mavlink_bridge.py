"""Unit tests for the PX4 MAVLink bridge conversion helpers.

These tests only exercise the pure conversion helpers, so they run without
ROS2 or pymavlink installed.
"""

from __future__ import annotations

import math
import threading

import pytest

from integrated_planning.ros_integration.px4_mavlink_bridge import (
    EARTH_RADIUS_M,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    MAV_MISSION_ACCEPTED,
    PX4MavlinkBridge,
    build_mission_items,
    enu_to_ned,
    ned_to_global,
)


def test_enu_to_ned_origin_shift() -> None:
    """The first waypoint becomes the NED origin; z becomes -z."""
    wps = [(5.0, 5.0, 12.0), (25.0, 25.0, 15.0)]
    ned = enu_to_ned(wps, (5.0, 5.0))
    assert ned[0] == (0.0, 0.0, -12.0)
    assert ned[1] == (20.0, 20.0, -15.0)


def test_ned_to_global_flat_earth() -> None:
    """Flat-Earth projection around lat=0 maps metres to degrees correctly."""
    home = (0.0, 0.0, 100.0)
    pts = ned_to_global([(0.0, 0.0, -12.0), (20.0, 20.0, -15.0)], home)
    m_per_deg = math.pi * EARTH_RADIUS_M / 180.0
    assert pts[0][2] == pytest.approx(12.0)
    assert pts[1][0] == pytest.approx(20.0 / m_per_deg)
    assert pts[1][2] == pytest.approx(15.0)


def test_build_mission_items_structure() -> None:
    """Mission = takeoff + waypoints + land, with takeoff altitude applied."""
    home = (0.0, 0.0, 100.0)
    wps = [(5.0, 5.0, 12.0), (25.0, 25.0, 15.0)]
    items = build_mission_items(wps, home, takeoff_alt=12.0)

    assert items[0]["cmd"] == MAV_CMD_NAV_TAKEOFF
    assert items[0]["z"] == 12.0
    assert items[-1]["cmd"] == MAV_CMD_NAV_LAND
    wp_cmds = [it["cmd"] for it in items[1:-1]]
    assert wp_cmds == [MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_WAYPOINT]
    # First waypoint sits at the take-off altitude (the origin).
    assert items[1]["z"] == pytest.approx(12.0)


def test_build_mission_items_without_land() -> None:
    """add_land=False leaves the last waypoint as NAV_WAYPOINT."""
    home = (0.0, 0.0, 100.0)
    wps = [(5.0, 5.0, 12.0), (25.0, 25.0, 15.0)]
    items = build_mission_items(wps, home, takeoff_alt=12.0, add_land=False)
    assert items[-1]["cmd"] == MAV_CMD_NAV_WAYPOINT


# ---------------------------------------------------------------------------
# Mission upload protocol (regression: PX4 requests MISSION_REQUEST_INT)
# ---------------------------------------------------------------------------


class _FakeMav:
    """Records mission-related MAVLink calls without any pymavlink dependency."""

    def __init__(self) -> None:
        self.count_sent = None
        self.int_sent = []
        self.float_sent = []

    def mission_count_send(self, ts, tc, count):
        self.count_sent = count

    def mission_item_send(self, ts, tc, seq, frame, cmd, current,
                          autocontinue, p1, p2, p3, p4, x, y, z):
        self.float_sent.append((seq, frame, cmd, x, y, z))

    def mission_item_int_send(self, ts, tc, seq, frame, cmd, current,
                              autocontinue, p1, p2, p3, p4, x, y, z):
        self.int_sent.append((seq, frame, cmd, x, y, z))


class _FakeMaster:
    """Scripted ``mavutil``-like connection returning canned messages."""

    def __init__(self, messages):
        self.mav = _FakeMav()
        self._messages = list(messages)
        self._idx = 0

    def recv_match(self, type=None, blocking=True, timeout=None):
        if self._idx < len(self._messages):
            msg = self._messages[self._idx]
            self._idx += 1
            return msg
        return None


class _FakeMsg:
    """Minimal MAVLink message with ``get_type()`` and field attributes."""

    def __init__(self, msg_type, **fields):
        self._type = msg_type
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self._type


class _FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _make_bridge() -> "PX4MavlinkBridge":
    """Build a bridge instance without invoking the ROS2 constructor."""
    bridge = object.__new__(PX4MavlinkBridge)
    bridge.target_system = 1
    bridge.target_component = 1
    bridge._status = ""
    bridge._status_lock = threading.Lock()
    bridge.get_logger = lambda: _FakeLogger()
    return bridge


def _sample_items() -> list:
    return [
        {"frame": 3, "cmd": MAV_CMD_NAV_TAKEOFF, "current": 0, "autocontinue": 1,
         "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": float("nan"),
         "x": 47.3977423, "y": 8.5455938, "z": 12.0},
        {"frame": 3, "cmd": MAV_CMD_NAV_WAYPOINT, "current": 0, "autocontinue": 1,
         "p1": 0.0, "p2": 1.0, "p3": 0.0, "p4": float("nan"),
         "x": 47.3979220, "y": 8.5457800, "z": 15.0},
    ]


def test_upload_mission_replies_int_to_px4() -> None:
    """PX4 requests items with MISSION_REQUEST_INT -> MISSION_ITEM_INT replies."""
    items = _sample_items()
    master = _FakeMaster([
        _FakeMsg("MISSION_REQUEST_INT", seq=0),
        _FakeMsg("MISSION_REQUEST_INT", seq=1),
        _FakeMsg("MISSION_ACK", type=MAV_MISSION_ACCEPTED),
    ])
    bridge = _make_bridge()
    bridge.master = master

    bridge._upload_mission(items)

    assert master.mav.count_sent == 2
    assert master.mav.float_sent == []
    assert len(master.mav.int_sent) == 2

    seq0, frame, cmd, x0, y0, z0 = master.mav.int_sent[0]
    assert (seq0, cmd) == (0, MAV_CMD_NAV_TAKEOFF)
    assert x0 == int(round(47.3977423 * 1e7))
    assert y0 == int(round(8.5455938 * 1e7))
    assert z0 == 12.0

    seq1, _, cmd1, x1, y1, z1 = master.mav.int_sent[1]
    assert (seq1, cmd1) == (1, MAV_CMD_NAV_WAYPOINT)
    assert x1 == int(round(47.3979220 * 1e7))
    assert z1 == 15.0


def test_upload_mission_float_fallback() -> None:
    """A GCS using MISSION_REQUEST still receives MISSION_ITEM (float)."""
    items = _sample_items()[:1]
    master = _FakeMaster([
        _FakeMsg("MISSION_REQUEST", seq=0),
        _FakeMsg("MISSION_ACK", type=MAV_MISSION_ACCEPTED),
    ])
    bridge = _make_bridge()
    bridge.master = master

    bridge._upload_mission(items)

    assert master.mav.int_sent == []
    assert len(master.mav.float_sent) == 1
    seq, _, cmd, x, y, z = master.mav.float_sent[0]
    assert (seq, cmd) == (0, MAV_CMD_NAV_TAKEOFF)
    assert x == 47.3977423
    assert y == 8.5455938
    assert z == 12.0
