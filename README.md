# PX4 ROS 2 — Обнаружение препятствий и планирование пути A*

Модульный пакет ROS 2 для обнаружения препятствий на основе LiDAR в реальном времени и **планирования пути A\***, разработанный для больших БПЛА с фиксированным крылом/VTOL (размах 4,57 метра). Реализация протестирована в симуляции и готова к работе на оборудовании.

##  Цель миссии

**Обнаруживать препятствия с помощью LiDAR и генерировать выполнимые бесконфликтные пути для больших летательных аппаратов с фиксированным крылом/VTOL с учётом кинодинамических ограничений.**

##  Обзор архитектуры

### **Глобальный планировщик A\* + Зонный обход** (Текущий)

Система использует взвешенный A\*
для глобального планирования пути в сочетании с существующим зонным обходом на основе отслеживания Калманом
для локального реактивного управления.

**Ключевые преимущества:**
-  **Детерминированность** — одинаковые входные данные всегда дают одинаковый путь
-  **Оптимальные пути** (w=1.0) или настраиваемый компромисс скорость/качество (w>1.0)
-  **Прямая интеграция с costmap** — читает `nav_msgs/OccupancyGrid` нативно
-  **Быстрота** — 10–100 мс для типичных сред (против 100–300 мс для RRT*)
-  **Сохраняет всё обнаружение препятствий** — отслеживание Калманом и зонные команды всё ещё активны

**Поток системы:**
```
LiDAR → obstacle_detector → /avoidance_command, /obstacle_velocity
                          → /costmap/update_trigger → costmap_node → /costmap/grid
                          → astar_planner           → /planned_path
```

### ~~RRT* с кинодинамическими ограничениями~~ (Устарело)

Предыдущий планировщик RRT* перемещён в `deprecated/rrt_star_planner/`.
См. [docs/ASTAR_INTEGRATION.md](docs/ASTAR_INTEGRATION.md) для обоснования миграции.

##  Возможности

### Обнаружение препятствий
- Обработка PointCloud2 в реальном времени
- Настраиваемые параметры обнаружения (расстояние, FOV, высота)
- Радиус инфляции препятствий для структурного зазора (концы крыльев, пропеллеры)
- Настроено для БПЛА с размахом крыла 4,57 метра и опасной дистанцией 80 м

### Планирование пути (A\*)
- Взвешенный A\* на 2D OccupancyGrid
- 8-связная сетка с диагональным перемещением
- Настраиваемый вес эвристики (1.0 = оптимально, >1.0 = быстрее/жаднее)
- Обход препятствий на основе costmap с настраиваемым летальным порогом
- Перепланирование 2 Гц, запускаемое обнаружением препятствий или отклонением от пути
- Совместимость со всеми существующими потребителями `/planned_path`

##  Структура проекта

```
Griphonyx/
├── obstacle_detection/          # Пакет ROS 2 обнаружения препятствий
│   ├── obstacle_detection/
│   │   ├── __init__.py
│   │   └── obstacle_detector.py # Основная нода обнаружения
│   ├── config/
│   │   └── params.yaml          # Параметры обнаружения
│   ├── launch/
│   │   └── detection.launch.py  # Файл запуска
│   ├── package.xml
│   ├── setup.py
│   └── setup.cfg
├── astar_planner/               # Пакет ROS 2 планировщика A* (заменяет RRT*)
│   ├── astar_planner/
│   │   ├── __init__.py
│   │   ├── astar_node.py        # Основная ROS 2 нода A*
│   │   ├── astar_algorithm.py   # Ядро A*
│   │   └── costmap_interface.py # Мост OccupancyGrid → numpy
│   ├── config/
│   │   └── astar_params.yaml    # Конфигурация A*
│   ├── launch/
│   │   └── astar_planner.launch.py
│   ├── package.xml
│   ├── setup.py
│   └── setup.cfg
├── costmap/                     # Пакет ROS 2 карты стоимости
├── integrated_planning/         # Модуль интегрированного планирования
│   └── launch/
│       └── full_system.launch.py    # Запуск полного стека
├── deprecated/
│   ├── README.md                # Уведомление об устаревании
│   └── rrt_star_planner/        # Заархивированный RRT* (не активен)
├── config/
│   └── integrated_system.yaml   # Единая конфигурация системы
├── context/
│   └── Autonomy Trade Study.xlsx
├── docs/
│   ├── ASTAR_INTEGRATION.md     # Руководство по интеграции A* (новое)
│   ├── RRT_star_tuning.md       # Заметки по настройке RRT* (исторические)
│   ├── architecture_information.md
│   └── testing_guide.md
├── README.md
└── requirements.txt
```

### Необходимые компоненты

- ROS 2 Humble (или новее)
- PX4 Autopilot + симуляция Gazebo Classic
- MicroXRCEAgent (для моста PX4 ↔ ROS2)
- Python 3.8+
- X11 (для Gazebo/RViz2), опционально NVIDIA GPU

### Установка

1. **Клонирование репозитория:**
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/Flex278/Griphonyx
   ```

2. **Установка зависимостей:**
   ```bash
   source /opt/ros/humble/setup.bash
   cd ~/ros2_ws
   pip3 install --upgrade pip
   pip3 install -r src/Griphonyx/requirements.txt
   ```

   На Ubuntu 22.04 (Humble) pip может выдать ошибку `externally-managed-environment` — тогда добавь `--break-system-packages` или используй venv.

3. **Сборка пакетов:**
   ```bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

   Выборочная сборка (опционально):
   ```bash
   colcon build --packages-select obstacle_detection costmap astar_planner integrated_planning
   ```

### Запуск ноды

1. **Запуск XRCE-DDS агента (мост PX4 ↔ ROS2):**
   ```bash
   MicroXRCEAgent udp4 -p 8888
   ```

2. **Запуск PX4 + симуляции Gazebo:**
   ```bash
   cd ~/PX4-Autopilot
   make px4_sitl gazebo-classic
   ```

3. **Запуск полного стека (A* + обнаружение препятствий):**
   ```bash
   ros2 launch integrated_planning full_system.launch.py
   ```

   Или запуск отдельных компонентов:
   ```bash
   ros2 launch obstacle_detection detection.launch.py
   ros2 launch costmap costmap.launch.py
   ros2 launch astar_planner astar_planner.launch.py
   ```

4. **Визуализация в RViz2:**
   ```bash
   rviz2
   ```

   Для Gazebo и RViz2 нужен X11-доступ (`$DISPLAY`, `/tmp/.X11-unix`). При наличии NVIDIA GPU можно ускорить рендеринг драйвером NVIDIA.

##  Конфигурация

### Параметры обнаружения препятствий

Редактируйте [obstacle_detection/config/params.yaml](obstacle_detection/config/params.yaml) для обнаружения препятствий:

```yaml
obstacle_detector:
  ros__parameters:
    danger_distance: 80.0      # Порог расстояния (метры) — настроен для большого БПЛА
    detection_width: 9.0       # Горизонтальная ширина ±Y (метры) — учитывает размах крыла 4,572 метра
    detection_height: 6.0      # Вертикальная высота ±Z (метры)
    min_distance: 2.0          # Игнорировать точки ближе этого расстояния (метры)
    obstacle_inflation: 3.0    # Запас безопасности вокруг препятствий (метры)
    lidar_topic: "/lidar/points"
```

**Характеристики БПЛА:**
- Размах крыла: 4,57 метра
- Длина: 4,42 м
- Высота: 1,22 м
- Полезная нагрузка: ящик 6×0,75×0,75 футов

### Параметры планирования пути

Редактируйте [astar_planner/config/astar_params.yaml](astar_planner/config/astar_params.yaml) для планировщика A*:

```yaml
astar_planner:
  ros__parameters:
    grid_resolution: 0.5       # метров на ячейку
    heuristic_weight: 1.0      # 1.0=оптимально, >1.0=быстрее/жаднее
    max_iterations: 10000      # жёсткое ограничение на раскрытие узлов
    planning_frequency: 2.0    # Гц — частота перепланирования
    lethal_cost_threshold: 70  # ячейки >= этого значения непроходимы
    cost_penalty_factor: 15.0  # штраф за прохождение через инфлированные ячейки
    goal_tolerance_m: 3.0      # радиус приёма цели (метры)
    safety_margin: 3.0         # минимальный зазор (размах крыла 4,572 метра)
    min_altitude_m: 5.0
    max_altitude_m: 120.0
```

##  Топики

**Подписан obstacle_detector:**
- `/lidar/points` (sensor_msgs/msg/PointCloud2) — данные облака точек LiDAR

**Публикует obstacle_detector:**
- `/obstacle_detected` (std_msgs/msg/Bool) — True, когда препятствие в опасной дистанции
- `/obstacle_distance` (std_msgs/msg/Float32) — расстояние до ближайшего препятствия в метрах
- `/avoidance_command` (std_msgs/msg/String) — зонная команда (EMERGENCY_HOVER / HARD_AVOID / REROUTE / ADJUST_HEADING / NORMAL_FLIGHT)
- `/obstacle_velocity` (geometry_msgs/msg/Vector3Stamped) — оценка скорости препятствия фильтром Калмана

**Публикует costmap_node:**
- `/costmap/grid` (nav_msgs/msg/OccupancyGrid) — 2D карта стоимости для планирования A*

**Публикует astar_planner:**
- `/planned_path` (nav_msgs/msg/Path) — глобальный путь, сгенерированный A* (заменяет путь RRT*)
- `/astar_planner/status` (std_msgs/msg/String) — статус планировщика в JSON

##  Тестирование в симуляции

1. **Проверка топика LiDAR:**
   ```bash
   ros2 topic list | grep cloud
   ros2 topic info /lidar/points
   ```

2. **Размещение препятствий в Gazebo** и мониторинг обнаружения:
   ```bash
   ros2 topic echo /obstacle_detected
   ros2 topic echo /obstacle_distance
   ```

##  Развёртывание на оборудовании

1. **Обновление драйвера LiDAR:**
   - Замените Gazebo LiDAR на драйвер реального сенсора (Velodyne, Ouster, Livox и т.д.)
   - Обновите параметр `lidar_topic` при необходимости

2. **Проверка размеров БПЛА:**
   - Убедитесь, что размах крыла, длина и высота соответствуют параметрам
   - Настройте `obstacle_inflation` в соответствии с требованиями структурного зазора
   - Обновите `min_turn_radius` на основе лётных испытаний

3. **Настройка для реальных условий:**
   - Настройте `danger_distance` под скорость полёта и время реакции
   - Уменьшите `detection_height` для фильтрации шума земли
   - Увеличьте `obstacle_clearance` в загромождённых средах

4. **Изменения кода не требуются** в основной логике обнаружения/планирования

##  Чек-лист разработки

См. [docs/CHECKLIST.md](docs/CHECKLIST.md) для полного поэтапного руководства по реализации.

##  Интеграция с PX4

Топики `/obstacle_detected` и `/obstacle_distance` могут использоваться:
- Системой команды планирования пути для запуска перестроения маршрута
- Сообщением PX4 `ObstacleDistance` UORB для бортового предотвращения столкновений
- Пользовательской логикой действий (удержание позиции, набор высоты, возврат домой)

**Принцип проектирования:** Восприятие отделено от планирования. Этот пакет не зависит от планировщика.

> См. [docs/ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md) для полного исследования компромиссов архитектуры и обоснования.

##  Известные ограничения

- Только обнаружение вперёд (X > 0)
- Фиксированная зона обнаружения (не адаптируется к скорости)
- Без удаления плоскости земли (предполагается горизонтальный полёт)
- Базовая логика минимального расстояния (без кластеризации)

##  Будущие улучшения

- [ ] Кластеризация DBSCAN для отслеживания нескольких препятствий
- [ ] Удаление плоскости земли RANSAC
- [ ] Динамическое масштабирование опасной зоны в зависимости от воздушной скорости
- [ ] Слияние глубины стереокамеры

##  Благодарности

Создано с соблюдением лучших практик PX4 + ROS 2 для непрерывности от симуляции к реальности.
