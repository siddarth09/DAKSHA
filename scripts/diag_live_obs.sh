#!/usr/bin/env bash
set -eo pipefail
WS=/home/sid/projects25
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export PYTHONPATH="$WS/src/ZERO/zero_control:${PYTHONPATH:-}"
exec python3 "$WS/src/ZERO/scripts/diag_live_obs.py"
