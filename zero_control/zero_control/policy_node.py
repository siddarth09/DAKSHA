"""Trained checkpoint -> /zero/eef_target. The teleop node's slot, driven by the policy.

    bash scripts/run_policy.sh [CHECKPOINT] [N_ACTION_STEPS]

The topology is unchanged from teleoperation: this node publishes the same 20-dim absolute
SE(3)+grip target the gamepad did, and `eef_control_node` still owns all the IK. Swapping the
operator for a policy therefore touches nothing else in the stack. X toggles run/stop, as in the
recorder.

Runs under lerobot_env because the checkpoint needs torch 2.12+cu128 (sm_120) and lerobot 0.5.1,
which live in /home/sid/lerobot_env. That venv is built with `include-system-site-packages =
false`, so run_policy.sh prepends ROS's site-packages to PYTHONPATH. rclpy imports cleanly
against the venv's numpy 2.4.4, and images are decoded with np.frombuffer so cv_bridge is never
needed.

It does not compute its own FK. ROS's pinocchio is compiled against numpy 1.x and refuses to
load under numpy 2.4.4, so this node cannot run ArmIK at all. Instead `eef_control_node`
publishes the measured 20-dim observation on /zero/eef_state. That is the better design
regardless: the controller already does this FK every tick, so there is one source of kinematic
truth rather than a second implementation that could drift from the one the dataset was recorded
with.

One fidelity detail matters. `observation.state` grip channels are the last commanded grip, not
a sensor reading (`record_node._measured_state` reads `self.action[9]` / `[19]`). That is
reproduced inside `eef_control_node._publish_state`, so what arrives here already matches what
the policy trained on. Images are RGB in [0,1] CHW, matching the recorder's bgr8 -> RGB flip.

`eef_control_node` does not clamp targets to a workspace box; an unreachable target just makes
the arm stall short with a large IK residual (its own docstring warns this "looks exactly like
the policy"). This node watches /zero/ik_status and warns, so a kinematic stall is not misread
as a policy failure.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float64MultiArray

from zero_control.action import DIM


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("zero_policy")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("dataset_root", "")
        self.declare_parameter("repo_id", "zero/base")
        # Empty means "take it from the dataset". The language instruction is part of the conditioning,
        # so a mismatch between training and inference is silent degradation that looks like a policy
        # failure. The default here was v1's string ("pick up the can and place it in the tray") while
        # cross_v1 was recorded with a longer one that names the handover. Reading it from the metadata
        # removes the whole class of bug.
        self.declare_parameter("task", "")
        self.declare_parameter("rate_hz", 10.0)          # must equal the dataset fps
        self.declare_parameter("n_action_steps", 25)     # replan horizon; <= chunk_size
        self.declare_parameter("cameras", ["front", "left_wrist", "right_wrist"])
        self.declare_parameter("autostart", False)
        self.declare_parameter("residual_warn_m", 0.02)
        # Dump every tick to an npz for offline analysis. The closed loop cannot be reproduced offline,
        # since the images depend on where the arm actually went, so when a rollout misbehaves this trace
        # is the only way to see the measured state and the commanded action side by side over time.
        self.declare_parameter("trace_path", "")
        # Mirage uses a "high-gain or blocking controller on the target robot" and requires the norm of
        # the error from the desired pose to be < 0.015 m at every timestep. Our servo is a one-step DLS
        # chase: a 70-86 mm move needs ~20 ticks (200 ms) to settle but gets 100 ms at 10 Hz, so it never
        # catches up, which is the 22-41 mm residual seen in rollout. Blocking decouples policy rate from
        # settling time. It makes the rollout slower than real time, which is fine in sim, and it is what
        # makes `action` equal to the achieved pose, so no forward dynamics model is needed.
        self.declare_parameter("block_until_settled", True)
        self.declare_parameter("settle_m", 0.015)      # Mirage's position tolerance
        # Mirage bounds the norm of the pose error including the quaternion, so orientation has to gate
        # the wait too. /zero/ik_status carries rot_err at indices 1 and 4.
        self.declare_parameter("settle_rad", 0.03)     # ~1.7 deg
        self.declare_parameter("settle_timeout_s", 1.0)
        # Blocking and a short n_action_steps work against each other: at 5 steps the target moves every
        # 0.5 s and the arm cannot settle inside the tolerance, giving 64 stalls in 106 s. The short
        # horizon did not help anyway (closest approach 61.5 mm against 57.0 mm at 25 steps), so prefer 25
        # and leave blocking on.

        ckpt = self.get_parameter("checkpoint").value
        if not ckpt or not Path(ckpt).exists():
            raise SystemExit(f"checkpoint '{ckpt}' not found. Pass -p checkpoint:=<dir>")
        self.task = self.get_parameter("task").value or None
        self.cams = list(self.get_parameter("cameras").value)
        self.warn_m = float(self.get_parameter("residual_warn_m").value)
        self.blocking = bool(self.get_parameter("block_until_settled").value)
        self.settle_m = float(self.get_parameter("settle_m").value)
        self.settle_rad = float(self.get_parameter("settle_rad").value)
        self.settle_timeout = float(self.get_parameter("settle_timeout_s").value)
        self.residual = 0.0
        self.residual_rot = 0.0
        self.waiting_since = None
        self.stalls = 0
        self._last_warn = 0.0
        self.trace_path = self.get_parameter("trace_path").value
        self.trace: list[np.ndarray] = []

        self.policy, self.pre, self.post, horizon = self._load(ckpt)

        self.state: np.ndarray | None = None
        self.rgb: dict[str, np.ndarray] = {}
        self.running = bool(self.get_parameter("autostart").value)
        self.prev_x = 0
        self.infer_ms = 0.0
        self.replans = 0
        self.ticks = 0

        for c in self.cams:
            self.create_subscription(Image, f"/zero/{c}/image_raw",
                                     lambda m, c=c: self._on_rgb(m, c), 10)
        self.create_subscription(Float64MultiArray, "/zero/eef_state", self._on_state, 10)
        self.create_subscription(Float64MultiArray, "/zero/ik_status", self._on_status, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/zero/eef_target", 10)

        # Refuse to run against a competing publisher. teleop_node publishes the same topic at 50 Hz
        # and, with no stick input, holds the home pose, so a teleop left running from a recording session
        # overwrites every policy command 5x per tick. The policy asked for 162 mm of motion and the arm
        # moved 2.3 mm in 106 s, while `resid` read 1 mm because the arm was tracking teleop's home target
        # perfectly. That is invisible from this node's own logs, which is why it needs a guard rather
        # than a comment.
        others = [info.node_name
                  for info in self.get_publishers_info_by_topic("/zero/eef_target")
                  if info.node_name != self.get_name()]
        if others:
            raise SystemExit(
                f"\n/zero/eef_target already has publisher(s): {others}.\n"
                f"They will fight this node for control of the arm; teleop publishes at 50 Hz\n"
                f"against this node's {self.get_parameter('rate_hz').value:.0f} Hz and holds the "
                f"home pose when idle.\n  Stop it first:  pkill -f 'zero_control/teleop'\n")

        hz = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f"policy ready: {Path(ckpt).parents[2].name}/{Path(ckpt).parents[0].name} | "
            f"{len(self.cams)} cameras | {hz:.0f} Hz "
            f"| replan every {horizon} steps ({horizon/hz:.1f} s) | task={self.task!r} | "
            f"{'RUNNING' if self.running else 'press X to start'}")

    # ------------------------------------------------------------------ policy
    def _load(self, ckpt: str):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        cfg.device = "cuda"
        # A shorter horizon than training is an inference choice, and a good one: open-loop chunk error
        # grows from 31 mm at k=0 to 47 mm at k=49, so replanning sooner is strictly better. See
        # scripts/eval_chunk_error.py.
        cfg.n_action_steps = min(int(self.get_parameter("n_action_steps").value), cfg.chunk_size)

        meta = LeRobotDatasetMetadata(self.get_parameter("repo_id").value,
                                      root=self.get_parameter("dataset_root").value or None)
        if self.task is None:
            tasks = list(meta.tasks.index)
            if len(tasks) != 1:
                raise SystemExit(f"dataset has {len(tasks)} tasks {tasks}; pass -p task:=<one>")
            self.task = tasks[0]
            self.get_logger().info(f"task taken from the dataset: {self.task!r}")
        policy = make_policy(cfg=cfg, ds_meta=meta)
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=ckpt)
        policy.eval()
        return policy, pre, post, cfg.n_action_steps

    def _tick(self) -> None:
        import torch
        if not self.running or self.state is None or len(self.rgb) < len(self.cams):
            return
        if self.blocking and self.ticks and (self.residual > self.settle_m
                                             or self.residual_rot > self.settle_rad):
            # Hold the current target until the arm arrives, rather than issuing a new one the servo has no
            # chance of reaching. Timeout so an unreachable pose cannot deadlock.
            if self.waiting_since is None:
                self.waiting_since = time.perf_counter()
            if time.perf_counter() - self.waiting_since < self.settle_timeout:
                return
            self.stalls += 1
            self.get_logger().warn(
                f"target not settled after {self.settle_timeout:.1f}s "
                f"({self.residual*1000:.0f} mm, {np.degrees(self.residual_rot):.1f} deg), "
                f"advancing anyway (stall #{self.stalls})")
        self.waiting_since = None
        batch = {"observation.state": torch.from_numpy(self.state).unsqueeze(0)}
        for c in self.cams:
            img = self.rgb[c].astype(np.float32) / 255.0                  # HWC RGB [0,1]
            batch[f"observation.images.{c}"] = torch.from_numpy(
                np.transpose(img, (2, 0, 1))).unsqueeze(0)                # -> 1,C,H,W
        batch["task"] = [self.task]

        t0 = time.perf_counter()
        with torch.no_grad():
            action = self.post(self.policy.select_action(self.pre(batch)))
        # Only 1 tick in n_action_steps runs the flow-matching integration; the rest pop a queued action
        # in ~0.1 ms. Averaging both together reported "9 ms" for what is really a ~225 ms inference, so
        # track the expensive calls only.
        dt = (time.perf_counter() - t0) * 1000
        if dt > 20.0:
            self.infer_ms = dt
            self.replans += 1

        a = action[0, :DIM].float().cpu().numpy().astype(float)
        self.pub.publish(Float64MultiArray(data=a.tolist()))
        if self.trace_path:
            self.trace.append(np.concatenate([[self.ticks, self.replans, self.residual,
                                               self.residual_rot], self.state, a]))
        self.ticks += 1
        if self.ticks % 50 == 0:
            self.get_logger().info(
                f"step {self.ticks}  replans {self.replans} (last {self.infer_ms:.0f} ms)  "
                f"L {np.round(a[0:3], 3)}  "
                f"R {np.round(a[10:13], 3)}  grip {a[9]:.2f}/{a[19]:.2f}  "
                f"resid {self.residual*1000:.0f} mm/{np.degrees(self.residual_rot):.1f}deg  "
                f"stalls {self.stalls}")

    # ------------------------------------------------------------------ ROS in
    def _on_rgb(self, msg: Image, cam: str) -> None:
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.rgb[cam] = img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()

    def _on_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == DIM:
            self.state = np.asarray(msg.data, dtype=np.float32)

    def _on_status(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 6:
            return
        self.residual = max(msg.data[0], msg.data[3])       # worst arm gates the wait
        self.residual_rot = max(msg.data[1], msg.data[4])
        # Throttle on time, not on self.ticks: ticks stops incrementing while blocking waits, so a
        # tick-based gate fires on every 100 Hz status message and floods the log.
        now = time.perf_counter()
        if now - self._last_warn < 2.0:
            return
        self._last_warn = now
        for side, err in (("left", msg.data[0]), ("right", msg.data[3])):
            if err > self.warn_m:
                self.get_logger().warn(
                    f"{side} IK residual {err*1000:.0f} mm; the arm is stalling short of the "
                    f"commanded pose; this is NOT the policy failing")

    def _on_joy(self, msg: Joy) -> None:
        x = msg.buttons[2] if len(msg.buttons) > 2 else 0
        if x and not self.prev_x:
            self.running = not self.running
            if self.running:
                self.policy.reset()          # clear the action queue between attempts
                self.ticks = 0
            self.get_logger().info("RUNNING" if self.running else "STOPPED")
        self.prev_x = x


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.trace_path and node.trace:
            np.savez(node.trace_path, trace=np.array(node.trace))
            print(f"wrote {node.trace_path}  ({len(node.trace)} ticks)")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
