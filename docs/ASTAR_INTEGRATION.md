# Интеграция A* Планирования Пути

## Обзор

Данный документ описывает интеграцию алгоритма поиска пути A\* в стек
обнаружения препятствий, заменяющего предыдущий планировщик RRT\*.

---

## Обоснование: A\* против RRT\*

| Критерий | RRT\* (предыдущий) | A\* (текущий) |
|-----------|-------------------|---------------|
| Оптимальность пути | ~105% от оптимального | Оптимальный (w=1.0) |
| Время планирования | 100–300 мс | 10–100 мс |
| Детерминизм | Недетерминированный | Детерминированный |
| Интеграция с картой стоимости | Своя проверка занятости | Прямая `OccupancyGrid` |
| Связность сетки | Непрерывное пространство состояний | 8-связная сетка |
| Перепланирование | По событиям | Периодическое + по событиям |
| Источник | Локальная реализация | Пакет `astar_planner` |

Планировщик A\* лучше подходит к существующему `costmap_node`, который уже
публикует `nav_msgs/OccupancyGrid` — A\* работает напрямую с этой сеткой
без накладных расходов на преобразование.

---

## Детали алгоритма

Основная реализация A\* (`astar_planner/astar_algorithm.py`) — это
**взвешенный A\*** на 2D-сетке занятости:

- **Сетка:** `nav_msgs/OccupancyGrid` из `costmap_node` (60×60 м, 0.5 м/ячейка)
- **Связность:** 8-связная (диагональное перемещение включено по умолчанию)
- **Эвристика:** Октильное расстояние — допустимое для 8-связных сеток
- **Вес эвристики:** Настраиваемый (`heuristic_weight`, по умолчанию 1.0 = оптимальный)
- **Функция стоимости:**
  - Свободная ячейка (cost=0): стоимость перемещения = 1.0 (ортогонально) или √2 (диагонально)
  - Раздутая ячейка (cost 1–89): стоимость перемещения, масштабированная на `cost_penalty_factor`
  - Неизвестная ячейка (cost=-1): стоимость перемещения × 1.5 (мягкий штраф)
  - Летальная ячейка (cost≥70): непроходимая
- **Ограничение итераций:** `max_iterations` (по умолчанию 10 000) предотвращает бесконечные циклы
  в несвязных картах

---

## Структура пакета

```
astar_planner/
├── astar_planner/
│   ├── __init__.py
│   ├── astar_algorithm.py     # Ядро поиска A*
│   ├── astar_node.py          # Нода ROS 2, оборачивающая алгоритм
│   └── costmap_interface.py   # Мост OccupancyGrid → numpy
├── config/
│   └── astar_params.yaml      # Настраиваемые параметры
├── launch/
│   └── astar_planner.launch.py
├── package.xml
├── setup.py
└── setup.cfg
```

---

## Интерфейсы топиков

### Подписки (сохранены от RRT\*)

| Топик | Тип | Источник | Описание |
|-------|------|--------|-------------|
| `/costmap/grid` | `nav_msgs/OccupancyGrid` | `costmap_node` | 2D-карта стоимости для планирования |
| `/obstacle_detected` | `std_msgs/Bool` | `obstacle_detector` | Триггерит перепланирование |
| `/obstacle_distance` | `std_msgs/Float32` | `obstacle_detector` | Расстояние до ближайшего препятствия |
| `/avoidance_command` | `std_msgs/String` | `obstacle_detector` | Команда зоны → перепланирование |
| `/obstacle_velocity` | `geometry_msgs/Vector3Stamped` | `obstacle_detector` | Скорость по Калману |
| `/vehicle/pose` | `geometry_msgs/PoseStamped` | БПЛА | Текущая позиция |
| `/planning/goal` | `geometry_msgs/PoseStamped` | Планировщик миссии | Цель навигации |

### Публикации

| Топик | Тип | Описание |
|-------|------|-------------|
| `/planned_path` | `nav_msgs/Path` | Глобальный путь, сгенерированный A\* |
| `/astar_planner/status` | `std_msgs/String` | Строка статуса в формате JSON |

---

## Конфигурация

Ключевые параметры в `astar_planner/config/astar_params.yaml`:

```yaml
astar_planner:
  ros__parameters:
    heuristic_weight: 1.0         # 1.0=оптимальный, 1.2=немного жадный
    max_iterations: 10000         # увеличьте для больших сред
    planning_frequency: 2.0       # Гц
    lethal_cost_threshold: 70     # должно совпадать с costmap_lethal_cost
    cost_penalty_factor: 15.0     # выше → пути обходят раздутые зоны
    goal_tolerance_m: 3.0         # метры
    safety_margin: 3.0            # метры (БПЛА с размахом крыльев 15 футов)
```

Значение `lethal_cost_threshold` (по умолчанию 70) должно совпадать с настройкой летальной стоимости карты стоимости.
Это значение соответствует параметру `costmap_lethal_cost` в
`costmap/config/costmap_params.yaml` для обеспечения согласованной обработки препятствий.

---

## Сохранённые возможности

Следующие функции исходной системы **полностью сохранены**:

- ✅ `obstacle_detection/` — обработка облака точек LiDAR
- ✅ `KalmanObstacleTracker` — фильтр Калмана на каждое препятствие (модель постоянной скорости 3D)
- ✅ Зональное избегание (CRITICAL / DANGER / WARNING / CAUTION / SAFE)
- ✅ Топик `/avoidance_command` — команды зон по-прежнему публикуются
- ✅ Топик `/obstacle_velocity` — скорость по Калману по-прежнему публикуется
- ✅ `costmap/` — карта стоимости OccupancyGrid с зональным раздутием
- ✅ `integrated_planning/` — 3D гибридный A\* для локального планирования (без изменений)

---

## Руководство по миграции с RRT\*

Если вам нужно вернуться к RRT\*:

1. Скопируйте `deprecated/rrt_star_planner/` обратно в корень репозитория:
   ```bash
   cp -r deprecated/rrt_star_planner ./
   ```

2. Обновите `costmap/launch/full_stack.launch.py` — замените ноду `astar_planner`
   на ноду `rrt_star_planner`.

3. Обновите `launch/full_system.launch.py` — замените ноду `astar_planner`.

4. Формат вывода топика `/planned_path` идентичен (`nav_msgs/Path`),
   поэтому нижестоящим потребителям не требуется никаких изменений.

---

## Тестирование

Запуск модульных тестов A\* на чистом Python (ROS не требуется):

```bash
cd astar_planner
python -m pytest tests/ -v
```

Полный системный тест (требуется ROS 2 + Gazebo):

```bash
ros2 launch launch/full_system.launch.py
ros2 topic echo /planned_path
ros2 topic echo /astar_planner/status
```
