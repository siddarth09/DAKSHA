#!/usr/bin/env bash
# Play a trained checkpoint closed-loop in the sim.
#
#   1) start the sim as usual:  ros2 launch zero_bringup rebot.launch.py can_x:=0.47 can_y:=0.24
#   2) then, in another terminal:  bash scripts/run_policy.sh
#   3) press X on the gamepad to start/stop the policy (same button as the recorder)
#
# Args: $1 = checkpoint dir (default: the r32 LoRA run), $2 = replan horizon in steps (default 25).
#
# `set -u` is deliberately NOT used: the ROS setup scripts reference unbound variables.
set -eo pipefail

CKPT="${1:-$HOME/zero_runs/base_lora_r32/checkpoints/last/pretrained_model}"
STEPS="${2:-25}"
WS=/home/sid/projects25

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

# The policy needs lerobot_env (torch cu128 for sm_120, lerobot 0.5.1 for PEFT loading). That venv
# is built with include-system-site-packages=false, so ROS's site-packages must be added by hand.
# zero_control comes from SOURCE because the install tree holds only an egg-link, which the venv's
# interpreter does not process -- this also means node edits apply without a colcon build.
export PYTHONPATH="$WS/src/ZERO/zero_control:${PYTHONPATH:-}"

exec /home/sid/lerobot_env/bin/python -m zero_control.policy_node --ros-args \
  --params-file "$WS/install/zero_bringup/share/zero_bringup/config/rebot_control.yaml" \
  -p checkpoint:="$CKPT" \
  -p dataset_root:="$HOME/zero_data/zero_base" \
  -p n_action_steps:="$STEPS"
