# Система интегрированного планирования – Справочник API

## Модуль: `integrated_planning`

Пакет верхнего уровня, предоставляющий основные публичные классы.

```python
from integrated_planning import HybridAStarPlanner3D, VehicleConfig, FlightMode, VoxelMap
```

---

## `integrated_planning.maps.voxel_map_3d`

### `class VoxelMap`

Трёхмерная сетка занятости на основе numpy-массива `uint8`.

```python
VoxelMap(size_x, size_y, size_z, resolution=1.0)
```

**Параметры:**

| Имя | Тип | Описание |
|------|------|-------------|
| `size_x` | `float` | Размер карты по X в метрах |
| `size_y` | `float` | Размер карты по Y в метрах |
| `size_z` | `float` | Размер карты по Z в метрах |
| `resolution` | `float` | Длина стороны вокселя в метрах (по умолчанию 1.0) |

**Атрибуты:**

| Имя | Тип | Описание |
|------|------|-------------|
| `nx, ny, nz` | `int` | Количество вокселей по каждой оси |
| `size_x, size_y, size_z` | `float` | Размеры карты |
| `resolution` | `float` | Длина стороны вокселя |

#### Методы

##### `world_to_voxel(x, y, z) → Tuple[int, int, int]`

Преобразование мировых координат в индексы вокселей.

##### `voxel_to_world(ix, iy, iz) → Tuple[float, float, float]`

Преобразование индексов вокселей в мировые координаты (центр вокселя).

##### `add_box_obstacle(x, y, z, size_x, size_y, size_z)`

Отметить выровненный по осям прямоугольный параллелепипед как занятый.
- `(x, y, z)` — минимальный угол.
- Размеры — `(size_x, size_y, size_z)`.

##### `add_cylinder_obstacle(x, y, z, radius, height)`

Отметить вертикальный цилиндр как занятый.
- `(x, y)` — горизонтальный центр, `z` — базовая высота.

##### `add_sphere_obstacle(x, y, z, radius)`

Отметить сферу как занятую.

##### `clear_map()`

Сбросить все воксели в свободное состояние.

##### `is_occupied_voxel(ix, iy, iz) → bool`

Вернуть `True`, если воксель `(ix, iy, iz)` занят.
Индексы вне границ возвращают `True` (консервативно).

##### `is_occupied_xyz(x, y, z) → bool`

Вернуть `True`, если мировая точка `(x, y, z)` занята.

##### `aabb_collision(x, y, z, yaw, pitch, vehicle) → bool`

Проверить столкновение с помощью консервативного AABB-сканирования контура аппарата.

**Параметры:**
- `x, y, z` – центр аппарата
- `yaw` – курс в радианах
- `pitch` – угол тангажа в радианах
- `vehicle` – экземпляр `VehicleConfig`

##### `update_from_pointcloud(points: np.ndarray)`

Отметить воксели из массива `(N, 3)` float-значений точек XYZ.
- Точки вне карты молча игнорируются.
- Точки NaN / Inf фильтруются.

##### `get_occupied_voxels() → List[Tuple[int, int, int]]`

Вернуть список всех кортежей индексов занятых вокселей.

##### `get_occupancy_rate() → float`

Вернуть долю занятых вокселей в `[0.0, 1.0]`.

##### `save_map(filename: str)`

Сохранить карту в сжатый файл `.npz`.

##### `classmethod load_map(filename: str) → VoxelMap`

Загрузить карту из файла `.npz`, созданного `save_map`.

---

## `integrated_planning.planners.hybrid_astar_3d`

### `class FlightMode(IntEnum)`

```python
class FlightMode(IntEnum):
    HOVER = 1   # 26-направленное голономное движение
    CRUISE = 2  # Неголономный полёт вперёд
```

---

### `class VehicleConfig`

Датакласс, описывающий размеры и ограничения БПЛА.

```python
@dataclass
class VehicleConfig:
    length: float = 1.8
    width: float = 1.2
    height: float = 0.6
    rotor_diameter: float = 0.5
    min_altitude: float = 5.0
    max_altitude: float = 120.0
    min_turn_radius: float = 3.0
    hover_step: float = 1.0
    cruise_step: float = 2.0
    inflation: float = 0.35
```

---

### `class Node3D`

Внутренний узел поиска (обычно не используется напрямую).

| Атрибут | Тип | Описание |
|-----------|------|-------------|
| `x, y, z` | `float` | Позиция |
| `yaw, pitch` | `float` | Ориентация (радианы) |
| `parent` | `Optional[str]` | Ключ родительского узла |
| `g` | `float` | Стоимость-к-приходу |
| `h` | `float` | Эвристическая стоимость-до-цели |
| `f` | `float` | Общая стоимость (g + h) |
| `mode` | `FlightMode` | Текущий режим полёта |
| `key` | `str` | Дискретный ключ сетки (свойство) |

---

### `class HybridAStarPlanner3D(BasePlanner)`

3D гибридный A\* планировщик пути.

```python
HybridAStarPlanner3D(voxel_map, vehicle=None, preferred_mode=FlightMode.HOVER)
```

**Параметры:**

| Имя | Тип | Описание |
|------|------|-------------|
| `voxel_map` | `VoxelMap` | Общая карта занятости |
| `vehicle` | `VehicleConfig` | Ограничения аппарата (по умолчанию `VehicleConfig()`) |
| `preferred_mode` | `FlightMode` | Начальный режим полёта |

#### Методы

##### `plan(start, goal, max_iter=80000) → Optional[List[Tuple]]`

Найти путь без столкновений от `start` до `goal`.

**Параметры:**
- `start` – кортеж `(x, y, z, yaw, pitch)`
- `goal` – кортеж `(x, y, z, yaw, pitch)`
- `max_iter` – Максимальное количество расширений A* (используйте ~20 000 для аварийного перепланирования)

**Возвращает:** Упорядоченный список кортежей `(x, y, z, yaw, pitch)` или `None`.

##### `validate_pose(pose) → bool`

Вернуть `True`, если высота позы находится в пределах `[min_altitude, max_altitude]`.

##### `static smooth_path(path, iterations=150) → List[Tuple]`

Сгладить сырой путь с помощью итеративного градиентного спуска.
- Начальная и конечная позы сохраняются.
- Сглаживаются только координаты XYZ; yaw/pitch берутся из исходного пути.

##### `_get_motion_primitives(mode) → List[Tuple]`

Вернуть примитивы движения для заданного `FlightMode`.
- HOVER: 26 единичных шагов, масштабированных на `vehicle.hover_step`
- CRUISE: 9 примитивов вперёд (прямо, крен влево/вправо × набор/уровень/снижение)

---

## `integrated_planning.planners.base_planner`

### `class BasePlanner(ABC)`

Абстрактный интерфейс для всех планировщиков.

#### Абстрактные методы

- `plan(start, goal) → Optional[List[Tuple]]`
- `validate_pose(pose) → bool`

#### Вспомогательные методы

- `static path_length(path) → float` – Полная длина дуги в 3D
- `static validate_path(path, max_segment_length=10.0) → bool` – Проверка корректности
- `static euclidean_distance_3d(a, b) → float` – Расстояние в 3D

---

## `integrated_planning.ros_integration.obstacle_map_bridge`

### `class ObstacleMapBridge(Node)`

Нода ROS2, соединяющая данные обнаружения препятствий с `VoxelMap`.

```python
# Запускается через ROS2
ros2 run integrated_planning obstacle_map_bridge
```

#### Параметры

| Параметр | Тип | По умолчанию | Описание |
|-----------|------|---------|-------------|
| `map_size_x` | `float` | 100.0 | Размер карты по X |
| `map_size_y` | `float` | 100.0 | Размер карты по Y |
| `map_size_z` | `float` | 50.0 | Размер карты по Z |
| `map_resolution` | `float` | 1.0 | Разрешение вокселя |
| `obstacle_inflation` | `float` | 0.5 | Радиус сферического препятствия |
| `update_rate` | `float` | 10.0 | Частота публикации визуализации (Гц) |

#### Подписанные топики

| Топик | Тип | Описание |
|-------|------|-------------|
| `/lidar/points` | `sensor_msgs/PointCloud2` | Скан LiDAR |
| `/obstacle_position` | `geometry_msgs/Vector3Stamped` | Позиция препятствия |

#### Публикуемые топики

| Топик | Тип | Описание |
|-------|------|-------------|
| `/voxel_map_viz` | `visualization_msgs/MarkerArray` | Занятые воксели для RViz2 |

#### Ключевые методы

- `pointcloud_callback(msg)` – Обработать PointCloud2 → обновить VoxelMap
- `obstacle_position_callback(msg)` – Добавить сферу в позиции препятствия
- `publish_map_visualization()` – Опубликовать MarkerArray
- `static pointcloud2_to_array(msg) → np.ndarray` – Преобразовать сообщение ROS в массив `(N, 3)`

---

## `integrated_planning.ros_integration.integrated_planner_node`

### `class IntegratedPlannerNode(Node)`

Основная интеграционная нода, соединяющая обнаружение препятствий с планированием пути.

```python
# Запускается через ROS2
ros2 run integrated_planning integrated_planner_node
```

#### Параметры

| Параметр | Тип | По умолчанию | Описание |
|-----------|------|---------|-------------|
| `map_size_x/y/z` | `float` | 100/100/50 | Размеры карты |
| `map_resolution` | `float` | 1.0 | Разрешение вокселя |
| `vehicle_length/width/height` | `float` | 1.8/1.2/0.6 | Размер аппарата |
| `vehicle_inflation` | `float` | 0.35 | Запас безопасности |
| `vehicle_min/max_altitude` | `float` | 5.0/120.0 | Пределы высоты |
| `max_iterations` | `int` | 80 000 | Бюджет A* в обычном режиме |
| `preferred_mode` | `str` | `"hover"` | Режим полёта |
| `replan_on_obstacle` | `bool` | `true` | Авто-перепланирование при обнаружении |
| `emergency_replan_distance` | `float` | 3.0 | Триггер аварийного режима (м) |

#### Подписанные топики

| Топик | Тип | Описание |
|-------|------|-------------|
| `/obstacle_detected` | `std_msgs/Bool` | Флаг препятствия |
| `/obstacle_distance` | `std_msgs/Float32` | Расстояние (м) |
| `/avoidance_zone` | `std_msgs/String` | Имя зоны |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Цель навигации |
| `/current_pose` | `geometry_msgs/PoseStamped` | Текущая поза БПЛА |

#### Публикуемые топики

| Топик | Тип | Описание |
|-------|------|-------------|
| `/planned_path` | `nav_msgs/Path` | Путевые точки для PX4 |
| `/planner_status` | `std_msgs/String` | Сообщения статуса |

#### Ключевые методы

| Метод | Описание |
|--------|-------------|
| `replan()` | Обычное перепланирование (80k итераций) |
| `emergency_replan()` | Быстрое перепланирование (20k итераций) |
| `emergency_hover()` | Опубликовать путь: набор высоты + зависание |
| `publish_path(waypoints)` | Преобразовать и опубликовать `nav_msgs/Path` |
| `publish_status(status)` | Опубликовать строку статуса |
| `static quaternion_to_yaw(quat)` | Извлечь yaw из кватерниона |
| `static yaw_to_quaternion(yaw)` | Построить кватернион из yaw |

---

## Веса стоимости

Определены как константы уровня модуля в `hybrid_astar_3d.py`:

| Константа | Значение | Назначение |
|----------|-------|---------|
| `W_STEER` | 1.2 | Штраф за боковое руление |
| `W_PITCH` | 1.0 | Штраф за угол тангажа |
| `W_PITCH_CHANGE` | 2.0 | Штраф за быстрые изменения тангажа (комфорт) |
| `W_MODE_SWITCH` | 3.0 | Штраф за переключения режимов |
| `W_ALTITUDE_FLOOR` | 50.0 | Серьёзный штраф вблизи минимальной высоты |
