#!/usr/bin/env python3
"""
Нода обнаружения препятствий LiDAR + YOLO для БПЛА PX4

Обрабатывает PointCloud2 (LiDAR) и опционально кадры камеры (YOLO) для обнаружения,
отслеживания и классификации препятствий. Реализует логику зонного обхода:

  Зона 1 — Критическая (0–1 м):   Экстренное зависание + остановка
  Зона 2 — Опасная    (1–3 м):   Жёсткий обход, быстрое замедление + набор высоты
  Зона 3 — Предупреждение (3–6 м):   Активное перестроение (передача RRT*)
  Зона 4 — Осторожно  (6–10 м):  Прогнозируемая корректировка курса
  Зона 5 — Безопасно  (10 м+):   Нормальный полёт, пассивный мониторинг

Отслеживание: фильтр Калмана на каждый трек препятствия (стиль SORT)
Обнаружение: кластеризация облака точек LiDAR + опциональное слияние bounding-box YOLO
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Vector3Stamped
import numpy as np
import struct
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Определения зон (метры — инфлированные расстояния)
# ---------------------------------------------------------------------------
class AvoidanceZone(IntEnum):
    SAFE     = 5   # > 10 м
    CAUTION  = 4   # 6 – 10 м
    WARNING  = 3   # 3 – 6 м
    DANGER   = 2   # 1 – 3 м
    CRITICAL = 1   # 0 – 1 м


ZONE_THRESHOLDS = {
    AvoidanceZone.CRITICAL: (0.0,  1.0),
    AvoidanceZone.DANGER:   (1.0,  3.0),
    AvoidanceZone.WARNING:  (3.0,  6.0),
    AvoidanceZone.CAUTION:  (6.0, 10.0),
    AvoidanceZone.SAFE:     (10.0, float('inf')),
}

AVOIDANCE_ACTIONS = {
    AvoidanceZone.CRITICAL: "EMERGENCY_HOVER",
    AvoidanceZone.DANGER:   "HARD_AVOID",
    AvoidanceZone.WARNING:  "REROUTE",
    AvoidanceZone.CAUTION:  "ADJUST_HEADING",
    AvoidanceZone.SAFE:     "NORMAL_FLIGHT",
}


# ---------------------------------------------------------------------------
# Фильтр Калмана — модель постоянной скорости для одного трека препятствия
# ---------------------------------------------------------------------------
class KalmanObstacleTracker:
    """
    Покадровый фильтр Калмана с моделью движения 3-D постоянной скорости.

    Вектор состояния: [x, y, z, vx, vy, vz]
    Измерение:        [x, y, z]
    """

    def __init__(self, initial_position: np.ndarray, dt: float = 0.1):
        self.dt = dt
        n_state = 6
        n_meas  = 3

        # Матрица перехода состояний (кинематическая модель)
        self.F = np.eye(n_state)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Матрица измерений (наблюдаем только позицию)
        self.H = np.zeros((n_meas, n_state))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Ковариация шума процесса
        self.Q = np.eye(n_state) * 0.1
        self.Q[3:, 3:] *= 5.0   # повышенная неопределённость скорости

        # Ковариация шума измерений
        self.R = np.eye(n_meas) * 0.5

        # Начальное состояние и ковариация
        self.x = np.zeros(n_state)
        self.x[:3] = initial_position
        self.P = np.eye(n_state) * 1.0

        self.age = 0            # кадров с момента создания
        self.hits = 1           # количество сопоставленных обнаружений
        self.time_since_update = 0

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        """Распространить оценку состояния на один шаг вперёд."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        self.age += 1
        return self.x[:3]

    # ------------------------------------------------------------------
    def update(self, measurement: np.ndarray):
        """Скорректировать состояние новым измерением позиции."""
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0

    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:]


# ---------------------------------------------------------------------------
# Простая жадная привязка ближайшего соседа (стиль SORT)
# ---------------------------------------------------------------------------
def associate_detections_to_tracks(
    detections: List[np.ndarray],
    tracks: List[KalmanObstacleTracker],
    max_dist: float = 5.0
) -> tuple:
    """
    Сопоставить обнаружения с существующими треками по ближайшему Евклидову расстоянию.

    Возвращает:
        matched        : список пар (track_idx, det_idx)
        unmatched_dets : индексы несопоставленных обнаружений
        unmatched_trks : индексы несопоставленных треков
    """
    if not tracks or not detections:
        return [], list(range(len(detections))), list(range(len(tracks)))

    pred_positions = np.array([t.position for t in tracks])
    det_positions  = np.array(detections)

    # Построить матрицу стоимости
    cost_matrix = np.linalg.norm(
        pred_positions[:, None, :] - det_positions[None, :, :], axis=2
    )  # форма (n_tracks, n_dets)

    matched, unmatched_dets, unmatched_trks = [], [], []
    used_tracks = set()
    used_dets   = set()

    # Жадное сопоставление — сортировка по стоимости
    flat_indices = np.argsort(cost_matrix, axis=None)
    for flat_idx in flat_indices:
        t_idx, d_idx = divmod(int(flat_idx), len(detections))
        if t_idx in used_tracks or d_idx in used_dets:
            continue
        if cost_matrix[t_idx, d_idx] > max_dist:
            break
        matched.append((t_idx, d_idx))
        used_tracks.add(t_idx)
        used_dets.add(d_idx)

    unmatched_dets = [i for i in range(len(detections))  if i not in used_dets]
    unmatched_trks = [i for i in range(len(tracks)) if i not in used_tracks]

    return matched, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# Main ROS 2 node
# ---------------------------------------------------------------------------
class ObstacleDetectorNode(Node):
    """
    ROS 2 node for real-time LiDAR + YOLO obstacle detection with Kalman tracking
    and zone-based avoidance command output.

    Published topics:
        /obstacle_detected      (Bool)         – any obstacle within danger_distance
        /obstacle_distance      (Float32)      – closest inflated obstacle distance
        /avoidance_command      (String)       – zone-based avoidance action string
        /obstacle_velocity      (Vector3Stamped) – velocity of closest tracked obstacle
        /costmap/update_trigger (Bool)         – signal costmap to rebuild from tracks
    """

    def __init__(self):
        super().__init__('obstacle_detector')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('danger_distance',    80.0)
        self.declare_parameter('detection_width',     9.0)
        self.declare_parameter('detection_height',    6.0)
        self.declare_parameter('min_distance',        2.0)
        self.declare_parameter('obstacle_inflation',  3.0)
        self.declare_parameter('lidar_topic',  '/lidar/points')
        self.declare_parameter('update_rate',        10.0)
        # Kalman / tracking
        self.declare_parameter('kalman_dt',           0.1)   # seconds per filter step
        self.declare_parameter('max_track_age',       5)     # frames before pruning
        self.declare_parameter('min_track_hits',      2)     # confirmed after N hits
        self.declare_parameter('association_dist',    5.0)   # metres for data-assoc
        # Clustering
        self.declare_parameter('cluster_eps',         2.0)   # DBSCAN-lite eps (m)
        self.declare_parameter('cluster_min_pts',     3)     # minimum cluster size
        # Zone thresholds (override defaults if needed)
        self.declare_parameter('zone_critical_m',     1.0)
        self.declare_parameter('zone_danger_m',       3.0)
        self.declare_parameter('zone_warning_m',      6.0)
        self.declare_parameter('zone_caution_m',     10.0)

        p = self.get_parameter
        self.danger_distance   = p('danger_distance').value
        self.detection_width   = p('detection_width').value
        self.detection_height  = p('detection_height').value
        self.min_distance      = p('min_distance').value
        self.obstacle_inflation = p('obstacle_inflation').value
        self.lidar_topic       = p('lidar_topic').value
        self.kalman_dt         = p('kalman_dt').value
        self.max_track_age     = p('max_track_age').value
        self.min_track_hits    = p('min_track_hits').value
        self.association_dist  = p('association_dist').value
        self.cluster_eps       = p('cluster_eps').value
        self.cluster_min_pts   = p('cluster_min_pts').value

        # Rebuild zone thresholds from params
        self.zone_thresholds = {
            AvoidanceZone.CRITICAL: (0.0,                   p('zone_critical_m').value),
            AvoidanceZone.DANGER:   (p('zone_critical_m').value, p('zone_danger_m').value),
            AvoidanceZone.WARNING:  (p('zone_danger_m').value,   p('zone_warning_m').value),
            AvoidanceZone.CAUTION:  (p('zone_warning_m').value,  p('zone_caution_m').value),
            AvoidanceZone.SAFE:     (p('zone_caution_m').value,  float('inf')),
        }

        # ── Tracking state ────────────────────────────────────────────
        self.tracks: List[KalmanObstacleTracker] = []
        self._next_track_id = 0

        # ── Subscribers ───────────────────────────────────────────────
        self.lidar_sub = self.create_subscription(
            PointCloud2, self.lidar_topic, self.lidar_callback, 10
        )

        # ── Publishers ────────────────────────────────────────────────
        self.obstacle_detected_pub  = self.create_publisher(Bool,           '/obstacle_detected',       10)
        self.obstacle_distance_pub  = self.create_publisher(Float32,        '/obstacle_distance',       10)
        self.avoidance_command_pub  = self.create_publisher(String,         '/avoidance_command',       10)
        self.obstacle_velocity_pub  = self.create_publisher(Vector3Stamped, '/obstacle_velocity',       10)
        self.costmap_trigger_pub    = self.create_publisher(Bool,           '/costmap/update_trigger',  10)

        self.get_logger().info('Obstacle Detector (LiDAR + Kalman + Zone Avoidance) initialized')
        self.get_logger().info(f'  Danger distance:  {self.danger_distance} m')
        self.get_logger().info(f'  Detection box:    ±{self.detection_width} m (Y)  ±{self.detection_height} m (Z)')
        self.get_logger().info(f'  Obstacle inflation: {self.obstacle_inflation} m')
        self.get_logger().info(f'  Kalman dt:        {self.kalman_dt} s')

    # ──────────────────────────────────────────────────────────────────
    # LiDAR callback
    # ──────────────────────────────────────────────────────────────────
    def lidar_callback(self, msg: PointCloud2):
        try:
            points = self.pointcloud2_to_array(msg)
            if points is None or len(points) == 0:
                self._advance_tracks([])
                self.publish_no_obstacle()
                return

            filtered = self.filter_detection_box(points)
            if len(filtered) == 0:
                self._advance_tracks([])
                self.publish_no_obstacle()
                return

            # Cluster points → one centroid per obstacle candidate
            centroids = self._cluster_centroids(filtered)

            # Kalman tracking
            confirmed_tracks = self._advance_tracks(centroids)

            if not confirmed_tracks:
                self.publish_no_obstacle()
                return

            # Find closest confirmed track (inflated distance)
            distances = [
                np.linalg.norm(t.position) - self.obstacle_inflation
                for t in confirmed_tracks
            ]
            min_idx  = int(np.argmin(distances))
            min_dist = distances[min_idx]
            closest  = confirmed_tracks[min_idx]

            # Determine avoidance zone
            zone = self._classify_zone(min_dist)
            obstacle_detected = zone.value <= AvoidanceZone.WARNING.value

            self.publish_obstacle_status(obstacle_detected, min_dist, zone, closest.velocity)
            self._trigger_costmap_update()

        except Exception as e:
            self.get_logger().error(f'lidar_callback error: {e}')

    # ──────────────────────────────────────────────────────────────────
    # Simple DBSCAN-lite clustering (no scipy dependency)
    # ──────────────────────────────────────────────────────────────────
    def _cluster_centroids(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Returns a list of cluster centroids using a simple radius-neighbour
        clustering approach (single-linkage, no external deps).
        """
        n = len(points)
        visited  = np.zeros(n, dtype=bool)
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            dists = np.linalg.norm(points - points[i], axis=1)
            neighbours = np.where(dists < self.cluster_eps)[0]
            if len(neighbours) < self.cluster_min_pts:
                visited[i] = True
                continue
            # Expand cluster
            cluster_pts = set(neighbours.tolist())
            queue = list(neighbours)
            while queue:
                j = queue.pop()
                if visited[j]:
                    continue
                visited[j] = True
                d2 = np.linalg.norm(points - points[j], axis=1)
                new_nb = np.where(d2 < self.cluster_eps)[0]
                if len(new_nb) >= self.cluster_min_pts:
                    for nb in new_nb:
                        if nb not in cluster_pts:
                            cluster_pts.add(nb)
                            queue.append(nb)
            visited[list(cluster_pts)] = True
            clusters.append(points[list(cluster_pts)].mean(axis=0))

        return clusters

    # ──────────────────────────────────────────────────────────────────
    # Kalman tracker management
    # ──────────────────────────────────────────────────────────────────
    def _advance_tracks(
        self, detections: List[np.ndarray]
    ) -> List[KalmanObstacleTracker]:
        """
        Predict all tracks, associate detections, update matched tracks,
        spawn new tracks for unmatched detections, prune stale tracks.

        Returns the list of *confirmed* tracks (hits >= min_track_hits).
        """
        # Predict
        for t in self.tracks:
            t.predict()

        # Associate
        matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
            detections, self.tracks, max_dist=self.association_dist
        )

        # Update matched
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx])

        # Spawn new tracks for unmatched detections
        for d_idx in unmatched_dets:
            self.tracks.append(KalmanObstacleTracker(detections[d_idx], dt=self.kalman_dt))

        # Prune stale tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_track_age
        ]

        # Return only confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_track_hits]

    # ──────────────────────────────────────────────────────────────────
    # Zone classification
    # ──────────────────────────────────────────────────────────────────
    def _classify_zone(self, distance: float) -> AvoidanceZone:
        """Map an inflated obstacle distance to its avoidance zone."""
        for zone in sorted(self.zone_thresholds, key=lambda z: z.value):
            lo, hi = self.zone_thresholds[zone]
            if lo <= distance < hi:
                return zone
        return AvoidanceZone.SAFE

    # ──────────────────────────────────────────────────────────────────
    # PointCloud2 parser
    # ──────────────────────────────────────────────────────────────────
    def pointcloud2_to_array(self, msg: PointCloud2) -> Optional[np.ndarray]:
        """
        Convert PointCloud2 to Nx3 NumPy array (FLU frame, float32 XYZ).
        Discards NaN / Inf points.
        """
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

    # ──────────────────────────────────────────────────────────────────
    # Detection-box filter
    # ──────────────────────────────────────────────────────────────────
    def filter_detection_box(self, points: np.ndarray) -> np.ndarray:
        """
        Keep only points inside the forward-facing detection frustum:
          X ∈ (min_distance, danger_distance]
          |Y| < detection_width
          |Z| < detection_height
        """
        mask = (
            (points[:, 0] > self.min_distance) &
            (points[:, 0] <= self.danger_distance) &
            (np.abs(points[:, 1]) < self.detection_width) &
            (np.abs(points[:, 2]) < self.detection_height)
        )
        return points[mask]

    # ──────────────────────────────────────────────────────────────────
    # Publishers
    # ──────────────────────────────────────────────────────────────────
    def publish_obstacle_status(
        self,
        detected: bool,
        distance: float,
        zone: AvoidanceZone,
        velocity: np.ndarray
    ):
        # Obstacle flag
        bool_msg = Bool()
        bool_msg.data = detected
        self.obstacle_detected_pub.publish(bool_msg)

        # Distance
        dist_msg = Float32()
        dist_msg.data = float(distance)
        self.obstacle_distance_pub.publish(dist_msg)

        # Zone-based avoidance command
        action = AVOIDANCE_ACTIONS[zone]
        cmd_msg = String()
        cmd_msg.data = action
        self.avoidance_command_pub.publish(cmd_msg)

        # Velocity of closest obstacle (for dynamic avoidance)
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = 'base_link'
        vel_msg.vector.x = float(velocity[0])
        vel_msg.vector.y = float(velocity[1])
        vel_msg.vector.z = float(velocity[2])
        self.obstacle_velocity_pub.publish(vel_msg)

        # Logging
        if zone == AvoidanceZone.CRITICAL:
            self.get_logger().error(
                f'🚨 CRITICAL — {distance:.2f} m → {action}  vel=({velocity[0]:.1f},{velocity[1]:.1f},{velocity[2]:.1f}) m/s'
            )
        elif zone == AvoidanceZone.DANGER:
            self.get_logger().warn(
                f'🟠 DANGER  — {distance:.2f} m → {action}'
            )
        elif zone == AvoidanceZone.WARNING:
            self.get_logger().warn(
                f'🟡 WARNING — {distance:.2f} m → {action}'
            )
        elif zone == AvoidanceZone.CAUTION:
            self.get_logger().info(
                f'🟢 CAUTION — {distance:.2f} m → {action}'
            )
        else:
            self.get_logger().debug(
                f'✅ SAFE    — {distance:.2f} m → {action}'
            )

    def publish_no_obstacle(self):
        """Publish clear status when no obstacles are in detection box."""
        self.obstacle_detected_pub.publish(Bool(data=False))
        self.obstacle_distance_pub.publish(Float32(data=float('inf')))
        cmd_msg = String()
        cmd_msg.data = AVOIDANCE_ACTIONS[AvoidanceZone.SAFE]
        self.avoidance_command_pub.publish(cmd_msg)

    def _trigger_costmap_update(self):
        """Signal the costmap node to rebuild from current tracks."""
        self.costmap_trigger_pub.publish(Bool(data=True))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    try:
        node = ObstacleDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in obstacle detector: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
