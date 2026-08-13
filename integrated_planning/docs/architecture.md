# Система интегрированного планирования – Архитектура

## Обзор

Система интегрированного планирования соединяет существующий конвейер **обнаружения препятствий**
с **трёхмерным автономным планированием пути** для БПЛА Griphonyx.

```
┌─────────────────────────────────────────────────────────────┐
│                     Griphonyx UAV System                     │
│                                                             │
│  ┌──────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │  LiDAR   │───►│ obstacle_       │───►│ obstacle_map_ │  │
│  │ VLP-16   │    │ detector.py     │    │ bridge.py     │  │
│  └──────────┘    └────────┬────────┘    └───────┬───────┘  │
│                           │                     │           │
│               /obstacle_detected          Updates           │
│               /obstacle_distance          VoxelMap          │
│               /avoidance_zone                │              │
│                           │                  │              │
│                           ▼                  ▼              │
│                  ┌─────────────────────────────────┐        │
│                  │   integrated_planner_node.py    │        │
│                  │                                 │        │
│                  │  ┌──────────────────────────┐  │        │
│                  │  │  HybridAStarPlanner3D    │  │        │
│                  │  │  (5-DOF, HOVER + CRUISE) │  │        │
│                  │  └──────────────────────────┘  │        │
│                  │  ┌──────────────────────────┐  │        │
│                  │  │       VoxelMap3D          │  │        │
│                  │  │  (3D occupancy grid)      │  │        │
│                  │  └──────────────────────────┘  │        │
│                  └────────────────┬────────────────┘        │
│                                   │                         │
│                            /planned_path                    │
│                                   │                         │
│                                   ▼                         │
│                          ┌────────────────┐                 │
│                          │  PX4 Autopilot │                 │
│                          └────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Компоненты

### 1. Детектор препятствий (пакет `obstacle_detection`)

Существующая нода, обрабатывающая облака точек LiDAR и публикующая:

| Topic | Type | Description |
|-------|------|-------------|
| `/obstacle_detected` | `std_msgs/Bool` | Флаг наличия препятствия |
| `/obstacle_distance` | `std_msgs/Float32` | Расстояние до ближайшего препятствия (м) |
| `/avoidance_zone` | `std_msgs/String` | Имя зоны (CRITICAL/DANGER/…) |
| `/lidar/points` | `sensor_msgs/PointCloud2` | Сырой скан LiDAR |
| `/obstacle_position` | `geometry_msgs/Vector3Stamped` | 3D позиция препятствия |

### 2. ObstacleMapBridge (пакет `integrated_planning`)

Преобразует данные препятствий в воксельную занятость:

```
/lidar/points ──────────────► pointcloud_callback()
                                   │
                                   ▼ pointcloud2_to_array()
                                   │
                                   ▼ VoxelMap.update_from_pointcloud()

/obstacle_position ─────────► obstacle_position_callback()
                                   │
                                   ▼ VoxelMap.add_sphere_obstacle()

Timer (10 Hz) ──────────────► publish_map_visualization()
                                   │
                                   ▼ /voxel_map_viz (MarkerArray)
```

### 3. VoxelMap (`integrated_planning.maps`)

Плотная трёхмерная сетка занятости:

```
VoxelMap(size_x=100, size_y=100, size_z=50, resolution=1.0)
    │
    ├── _grid: np.ndarray[uint8, (nx, ny, nz)]
    │
    ├── add_box_obstacle(x, y, z, sx, sy, sz)
    ├── add_cylinder_obstacle(x, y, z, radius, height)
    ├── add_sphere_obstacle(x, y, z, radius)
    │
    ├── is_occupied_xyz(x, y, z) → bool
    ├── aabb_collision(x, y, z, yaw, pitch, vehicle) → bool
    │
    └── update_from_pointcloud(points: ndarray)
```

### 4. HybridAStarPlanner3D (`integrated_planning.planners`)

Гибридный A* с 5 степенями свободы и двумя режимами полёта:

```
State: (x, y, z, yaw, pitch)

HOVER mode:
    26-направленное голономное движение
    step = vehicle.hover_step (по умолчанию 1.0 м)

CRUISE mode:
    Неголономный полёт вперёд
    step = vehicle.cruise_step (по умолчанию 2.0 м)
    Углы поворота: ±arcsin(step / 2r)

Cost function:
    f(n) = g(n) + h(n)
    g(n) = Σ(step_dist + W_STEER·|Δyaw| + W_PITCH·|pitch|
              + W_PITCH_CHANGE·|Δpitch| + W_MODE_SWITCH·[смена режима]
              + W_ALTITUDE_FLOOR·[z < min_alt+1])
    h(n) = Euclidean3D(n, goal)
```

### 5. IntegratedPlannerNode (`integrated_planning.ros_integration`)

Центральная координационная нода:

```
Subscriptions → State updates → Zone-based action selection → Plan/Replan

/obstacle_detected ──► obstacle_detected flag
/obstacle_distance ──► obstacle_distance (м)
                           │
                           ▼ if dist ≤ emergency_replan_distance
                           └──► emergency_replan() [20k итераций]
/avoidance_zone ────► zone string
                           │
                           ├── CRITICAL → emergency_hover()
                           ├── DANGER   → emergency_replan()
                           └── WARNING  → replan() [80k итераций]
/goal_pose ─────────► goal_pose tuple → replan()
/current_pose ──────► current_pose tuple

Timer (1 Hz) ───────► replan_timer_callback()
                           └── if obstacle_detected → replan()
```

## Потоки данных

### Нормальный режим работы

```
1. БПЛА получает /goal_pose
2. IntegratedPlannerNode сохраняет цель, вызывает replan()
3. HybridAStarPlanner3D.plan(current_pose, goal) выполняется
4. Сглаженный путь публикуется в /planned_path (nav_msgs/Path)
5. PX4 следует по путевым точкам
```

### Избегание препятствий

```
1. LiDAR обнаруживает препятствие
2. obstacle_detector публикует /obstacle_detected=True
3. ObstacleMapBridge отмечает воксели в VoxelMap
4. IntegratedPlannerNode получает /obstacle_detected
5. Вызывает replan() с обновлённой картой
6. Новый путь обходит препятствие, публикуется в /planned_path
```

### Аварийное избегание

```
1. Препятствие входит в зону DANGER/CRITICAL
2. /avoidance_zone = "CRITICAL"
3. Вызывается IntegratedPlannerNode.emergency_hover()
4. БПЛА набирает 5 м высоты и зависает
5. Попытка перепланирования с уменьшенным бюджетом итераций
```

## Конфигурация

Все параметры настраиваются через ROS2 в `config/planner_params.yaml`:

| Параметр | По умолчанию | Описание |
|-----------|---------|-------------|
| `map_size_x/y/z` | 100/100/50 м | Размеры воксельной карты |
| `map_resolution` | 1.0 м | Длина стороны вокселя |
| `vehicle_min_altitude` | 5.0 м | Минимум AGL (FAA) |
| `vehicle_max_altitude` | 120.0 м | Потолок Part 107 (FAA) |
| `max_iterations` | 80 000 | Бюджет A* в обычном режиме |
| `emergency_replan_distance` | 3.0 м | Триггер аварийного режима |
| `preferred_mode` | `"hover"` | Режим полёта по умолчанию |

## Зависимости

### Python (≥ 3.8)
- `numpy ≥ 1.21.0`
- `scipy ≥ 1.7.0` (опционально, для продвинутых функций)

### ROS2 (Humble / Iron / Jazzy)
- `rclpy`
- `std_msgs`, `sensor_msgs`, `nav_msgs`
- `geometry_msgs`, `visualization_msgs`

## Производительность

| Метрика | Значение |
|--------|-------|
| Время обычного планирования | < 0.5 с (80k итераций) |
| Время аварийного планирования | < 0.1 с (20k итераций) |
| Частота обновления карты | 10 Гц |
| Период проверки перепланирования | 1 Гц |
