#!/usr/bin/env bash
#
# run_px4_demo.sh - sync, build and launch the PX4 bridge demo.
#
# Copies the integrated_planning package into the ros2_px4_dev container,
# rebuilds it and launches the full planning + MAVLink bridge stack.
#
# Prerequisites:
#   * PX4 SITL running in the px4_gazebo container (Gazebo Classic).
#   * ros2_px4_dev container running.
#
# Usage:
#   ./scripts/run_px4_demo.sh
#
set -euo pipefail

HOST_SRC="${GRIPHONYX_SRC:-$HOME/Рабочий стол/Griphonyx}"
CONTAINER="ros2_px4_dev"
WS="/ros2_ws"

echo "==> Syncing integrated_planning into ${CONTAINER}"
docker cp "${HOST_SRC}/integrated_planning" "${CONTAINER}:${WS}/src/griphonyx/"

echo "==> Building integrated_planning"
docker exec "${CONTAINER}" bash -lc "
    set -e
    source /opt/ros/humble/setup.bash
    cd ${WS}
    colcon build --packages-select integrated_planning
"

echo "==> Launching PX4 demo (Ctrl-C to stop)"
docker exec -it "${CONTAINER}" bash -lc "
    source /opt/ros/humble/setup.bash
    source ${WS}/install/setup.bash
    ros2 launch integrated_planning px4_demo.launch.py
"
