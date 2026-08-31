#!/usr/bin/env bash
# Play a trained checkpoint closed-loop in the sim.
#
#   1) start the sim:  ros2 launch zero_bringup rebot.launch.py can_x:=0.47 can_y:=0.24
#      ...or the Panda:  ros2 launch zero_bringup panda.launch.py can_x:=0.47 can_y:=0.24
#                        then run this with  ROBOT=panda
#   2) then, in another terminal:  bash scripts/run_policy.sh
#   3) press X on the gamepad to start/stop the policy (same button as the recorder)
#
# Args: $1 = checkpoint dir (default: the r32 LoRA run), $2 = replan horizon in steps (default 25).
#
# `set -u` is deliberately NOT used: the ROS setup scripts reference unbound variables.
set -eo pipefail

# xargs trims whitespace: a trailing space after a line-continuation backslash makes the shell
# pass a single-space argument, which `${1:-default}` treats as SET. That reached rcl as
# `-p checkpoint:=" "`, which it parses as unset, and the node then died with a confusing
# "parameter 'checkpoint' is not initialized".
CKPT="$(printf '%s' "${1:-}" | xargs || true)"
CKPT="${CKPT:-$HOME/zero_runs/crossv2_full_c25/checkpoints/last/pretrained_model}"
STEPS="$(printf '%s' "${2:-}" | xargs || true)"
STEPS="${STEPS:-25}"

# Which robot to drive. The policy itself is embodiment-agnostic -- it emits the 20-dim absolute
# EEF pose from action.py, and each robot's own eef_control_node turns that into joint commands
# through its own IK. Only the params file differs. This is the cross-embodiment claim, so it is
# worth stating plainly: NOTHING about the checkpoint changes here.
ROBOT="${ROBOT:-rebot}"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint dir not found: '$CKPT'" >&2
  echo "  usage: bash scripts/run_policy.sh [CHECKPOINT_DIR] [N_ACTION_STEPS]" >&2
  echo "  available:" >&2
  ls -d "$HOME"/zero_runs/*/checkpoints/*/pretrained_model 2>/dev/null | sed 's/^/    /' >&2
  exit 1
fi
case "$STEPS" in (''|*[!0-9]*) echo "ERROR: N_ACTION_STEPS must be an integer, got '$STEPS'" >&2; exit 1;; esac
WS=/home/sid/projects25

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

# The policy needs lerobot_env (torch cu128 for sm_120, lerobot 0.5.1 for PEFT loading). That venv
# is built with include-system-site-packages=false, so ROS's site-packages must be added by hand.
# zero_control comes from SOURCE because the install tree holds only an egg-link, which the venv's
# interpreter does not process -- this also means node edits apply without a colcon build.
export PYTHONPATH="$WS/src/ZERO/zero_control:${PYTHONPATH:-}"

# The X button comes from /joy, which is published by the `joy` driver -- NOT by zero_control's
# teleop node. rebot_teleop.launch.py starts both, so using it here would leave teleop publishing
# /zero/eef_target at 50 Hz and holding the home pose, which silently overrides every policy
# command (measured: policy asked for 162 mm, arm moved 2.3 mm). So start joy_node alone, and only
# if nothing is already publishing /joy. autorepeat_rate matches the teleop launch: joy_node
# otherwise publishes only on CHANGE, so a button press can be missed.
if ! ros2 topic info /joy 2>/dev/null | grep -q "Publisher count: [1-9]"; then
  echo "starting joy_node (nothing is publishing /joy)"
  ros2 run joy joy_node --ros-args \
    -p deadzone:=0.05 -p autorepeat_rate:=50.0 -p coalesce_interval_ms:=5 &
  JOY_PID=$!
  # NOT a trap + exec: `exec` replaces this shell, so the trap is discarded and Ctrl-C never runs
  # it. That leaked five joy_node processes over one debugging session. Run the node as a CHILD
  # instead and clean up after it returns.
  sleep 2
else
  echo "/joy already has a publisher -- reusing it"
fi

# Anything publishing /zero/eef_target will fight the policy for the arm; the node itself also
# refuses to start in that case, but say so here where the fix is obvious.
if ros2 node list 2>/dev/null | grep -q '^/zero_teleop$'; then
  echo
  echo "ERROR: zero_teleop is running and publishes /zero/eef_target at 50 Hz, which will"
  echo "       override every policy command. Stop it first:"
  echo "         pkill -f 'zero_control/teleop'"
  exit 1
fi

cleanup() { [ -n "${JOY_PID:-}" ] && kill "$JOY_PID" 2>/dev/null; pkill -f 'lib/joy/joy_node' 2>/dev/null; true; }
trap cleanup EXIT INT TERM

/home/sid/lerobot_env/bin/python -m zero_control.policy_node --ros-args \
  --params-file "$WS/install/zero_bringup/share/zero_bringup/config/${ROBOT}_control.yaml" \
  -p checkpoint:="$CKPT" \
  -p repo_id:=zero/cross \
  -p dataset_root:="$HOME/zero_data/cross_base" \
  -p n_action_steps:="$STEPS" \
  ${TRACE:+-p trace_path:="$TRACE"}
