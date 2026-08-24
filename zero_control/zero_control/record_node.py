"""Record teleop demos straight into a LeRobotDataset.

    ros2 run zero_control record --ros-args \
        --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml \
        -p root:=/home/sid/zero_data/rebot_pick_place \
        -p task:="pick up the can and place it in the tray"

    BACK  start / stop an episode        B  discard the episode in progress

WHAT THE POLICY SEES vs WHAT IS STORED -- the distinction this whole dataset turns on:

    observation.state   20-dim: per hand pos(3) + rot6d(6) + grip(1), MEASURED via FK
    action              the same 20-dim, COMMANDED (what the operator asked for)
    observation.images  front + both wrists

Those are embodiment-invariant, and they are the only keys a transferable policy may read. Joint
angles must NOT be among them: the reBot has 6 per arm and Panda 7, so a policy trained on one
cannot even be fed the other -- the vector length differs. This is why Mirage drives its transfer
through Cartesian control rather than joint control.

Everything else is stored for repair and analysis, NOT for the policy:

    observation.joint_pos / joint_vel   makes the dataset REPAIRABLE. The tool point was wrong
                                        three times while this rig was being built (113 mm, then
                                        39 mm, then right). EEF poses recorded against a wrong
                                        TCP are permanently mislabelled -- unless joints are
                                        stored, in which case every pose can be recomputed by FK.
                                        It is also what a Mirage-style re-render of the other
                                        robot needs, since the image is embodiment-specific too.
    observation.ft                      per-finger wrench, 4 x 6. Asymmetric loading is what a
                                        bad grasp looks like numerically.
    observation.ik_residual             tracking error. A frame where the arm was NOT following
                                        the operator is worse than useless: it is mislabelled,
                                        and this is what lets you filter it out afterwards.
    observation.grip_cmd                the latched gripper command, to separate "operator asked
                                        for closed" from "gripper reached closed".
    observation.depth.*                 float32 metres, one per camera.

FRAMES ARE ASSEMBLED FROM LATEST-VALUE CACHES, not time-synchronised. The streams run at
different rates (joints 100 Hz, cameras ~10 Hz) and a hard sync would drop most frames. The
dataset fps is therefore capped by the SLOWEST stream -- the cameras -- which is why they were
cut to three at 224x224 to get 10 Hz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, Joy
from std_msgs.msg import Float64MultiArray

from zero_control.action import DIM, SIDES, pack
from zero_control.ik import ArmIK


class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("zero_record")
        self.declare_parameter("robot", "rebot")
        self.declare_parameter("root", "")
        self.declare_parameter("task", "pick up the can and place it in the tray")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("depth", True)
        self.declare_parameter("cameras", ["front", "left_wrist", "right_wrist"])
        self.declare_parameter("ft_sensors", [""])
        self.declare_parameter("eef_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("btn_toggle", 6)     # BACK
        self.declare_parameter("btn_discard", 1)    # B
        for side in SIDES:
            self.declare_parameter(f"{side}_arm_joints", [""])
            self.declare_parameter(f"{side}_eef_frame", "")

        self.key = self.get_parameter("robot").value
        root = self.get_parameter("root").value
        if not root:
            raise SystemExit("pass -p root:=<dataset dir>")
        self.root = Path(root)
        self.task = str(self.get_parameter("task").value)
        self.fps = float(self.get_parameter("fps").value)
        self.cams = list(self.get_parameter("cameras").value)
        self.fts = [s for s in self.get_parameter("ft_sensors").value if s]
        self.use_depth = bool(self.get_parameter("depth").value)

        urdf = str(Path(get_package_share_directory("zero_description"))
                   / "urdf" / f"zero_{self.key}.urdf")
        off = tuple(self.get_parameter("eef_offset").value)
        self.ik = {s: ArmIK(urdf, self.get_parameter(f"{s}_eef_frame").value,
                            list(self.get_parameter(f"{s}_arm_joints").value), off)
                   for s in SIDES}
        self.arm_joints = {s: list(self.get_parameter(f"{s}_arm_joints").value) for s in SIDES}

        self.q = np.zeros(self.ik["left"].model.nq)
        self.joint_names: list[str] = []
        self.joint_pos = np.zeros(0, np.float32)
        self.joint_vel = np.zeros(0, np.float32)
        self.rgb: dict[str, np.ndarray | None] = {c: None for c in self.cams}
        self.dep: dict[str, np.ndarray | None] = {c: None for c in self.cams}
        self.ft: dict[str, np.ndarray] = {s: np.zeros(6, np.float32) for s in self.fts}
        self.action = np.zeros(DIM, np.float32)
        self.have_action = False
        self.ik_res = np.zeros(6, np.float32)
        self.have_js = False
        self.prev_buttons: list[int] = []
        self.joy: Joy | None = None

        self.ds = None                      # created lazily, once shapes are known
        self.recording = False
        self.n_frames = 0
        self.n_episodes = 0

        for c in self.cams:
            self.create_subscription(Image, f"/zero/{c}/image_raw",
                                     lambda m, k=c: self._on_rgb(m, k), 5)
            if self.use_depth:
                self.create_subscription(Image, f"/zero/{c}/depth",
                                         lambda m, k=c: self._on_depth(m, k), 5)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        for s in self.fts:
            self.create_subscription(WrenchStamped, f"/{s}_broadcaster/wrench",
                                     lambda m, k=s: self._on_ft(m, k), 10)
        self.create_subscription(Float64MultiArray, "/zero/eef_target", self._on_action, 10)
        self.create_subscription(Float64MultiArray, "/zero/ik_status", self._on_status, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.create_timer(1.0 / self.fps, self._tick)
        self.get_logger().info(
            f"recorder ready: {self.fps:.0f} fps, {len(self.cams)} cameras, "
            f"{len(self.fts)} F/T, depth={'on' if self.use_depth else 'off'}\n"
            f"  dataset root: {self.root}\n"
            f"  task: {self.task!r}\n"
            "  BACK = start/stop episode, B = discard")

    # ---------------------------------------------------------------- callbacks
    def _on_rgb(self, msg: Image, cam: str) -> None:
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.rgb[cam] = img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()

    def _on_depth(self, msg: Image, cam: str) -> None:
        if msg.encoding == "32FC1":
            d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        else:                                     # 16UC1 millimetres -> metres
            d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width) / 1000.0
        self.dep[cam] = d.astype(np.float32).copy()

    def _on_js(self, msg: JointState) -> None:
        model = self.ik["left"].model
        for name, pos in zip(msg.name, msg.position):
            jid = model.getJointId(name)
            if jid < model.njoints:
                self.q[model.joints[jid].idx_q] = pos
        if not self.joint_names:
            self.joint_names = list(msg.name)
        self.joint_pos = np.asarray(msg.position, np.float32)
        self.joint_vel = (np.asarray(msg.velocity, np.float32) if len(msg.velocity)
                          else np.zeros_like(self.joint_pos))
        self.have_js = True

    def _on_ft(self, msg: WrenchStamped, sensor: str) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        self.ft[sensor] = np.array([f.x, f.y, f.z, t.x, t.y, t.z], np.float32)

    def _on_action(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == DIM:
            self.action = np.asarray(msg.data, np.float32)
            self.have_action = True

    def _on_status(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == 6:
            self.ik_res = np.asarray(msg.data, np.float32)

    def _on_joy(self, msg: Joy) -> None:
        self.joy = msg

    def _pressed(self, which: str) -> bool:
        if self.joy is None:
            return False
        idx = int(self.get_parameter(f"btn_{which}").value)
        if not 0 <= idx < len(self.joy.buttons):
            return False
        was = self.prev_buttons[idx] if idx < len(self.prev_buttons) else 0
        return bool(self.joy.buttons[idx]) and not was

    # ---------------------------------------------------------------- dataset
    def _features(self) -> dict:
        n_joint = len(self.joint_pos)
        feats: dict = {
            "observation.state": {"dtype": "float32", "shape": (DIM,), "names": None},
            "action": {"dtype": "float32", "shape": (DIM,), "names": None},
            "observation.joint_pos": {"dtype": "float32", "shape": (n_joint,),
                                      "names": self.joint_names},
            "observation.joint_vel": {"dtype": "float32", "shape": (n_joint,),
                                      "names": self.joint_names},
            "observation.ft": {"dtype": "float32", "shape": (6 * len(self.fts),),
                               "names": [f"{s}.{a}" for s in self.fts
                                         for a in ("fx", "fy", "fz", "tx", "ty", "tz")]},
            "observation.ik_residual": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.grip_cmd": {"dtype": "float32", "shape": (2,), "names": list(SIDES)},
        }
        for c in self.cams:
            h, w = self.rgb[c].shape[:2]
            feats[f"observation.images.{c}"] = {"dtype": "video", "shape": (h, w, 3),
                                               "names": ["height", "width", "channel"]}
            if self.use_depth:
                feats[f"observation.depth.{c}"] = {"dtype": "float32", "shape": (h, w),
                                                  "names": ["height", "width"]}
        return feats

    def _ensure_dataset(self) -> None:
        if self.ds is not None:
            return
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.ds = LeRobotDataset.create(
            repo_id=f"zero/{self.key}", fps=int(self.fps), features=self._features(),
            root=self.root, robot_type=self.key,
            # Images go through worker threads; the control loop must not block on PNG/video
            # encoding or the dataset silently drops to whatever rate the writer can sustain.
            image_writer_processes=0, image_writer_threads=2 * len(self.cams))
        self.get_logger().info(f"created dataset at {self.root}")

    def _measured_state(self) -> np.ndarray:
        """MEASURED end-effector pose, by FK from the joints -- not the commanded target."""
        poses, grips = {}, {}
        for side in SIDES:
            T = self.ik[side].fk(self.q)
            poses[side] = (T.translation, T.rotation)
            # gripper opening, normalised, read from the measured joints
            grips[side] = float(self.action[9 if side == "left" else 19])
        return pack(poses, grips).astype(np.float32)

    # ---------------------------------------------------------------- loop
    def _tick(self) -> None:
        if self.joy is not None:
            if self._pressed("toggle"):
                if self.recording:
                    self._stop(save=True)
                else:
                    self._start()
            elif self._pressed("discard") and self.recording:
                self._stop(save=False)
            self.prev_buttons = list(self.joy.buttons)

        if not self.recording:
            return
        if not self.have_js or any(self.rgb[c] is None for c in self.cams):
            return
        if self.use_depth and any(self.dep[c] is None for c in self.cams):
            return

        frame = {
            "observation.state": self._measured_state(),
            "action": self.action.copy(),
            "observation.joint_pos": self.joint_pos.copy(),
            "observation.joint_vel": self.joint_vel.copy(),
            "observation.ft": np.concatenate([self.ft[s] for s in self.fts])
                              if self.fts else np.zeros(0, np.float32),
            "observation.ik_residual": self.ik_res.copy(),
            "observation.grip_cmd": np.array(
                [self.action[9], self.action[19]], np.float32),
            "task": self.task,
        }
        for c in self.cams:
            frame[f"observation.images.{c}"] = self.rgb[c]
            if self.use_depth:
                frame[f"observation.depth.{c}"] = self.dep[c]
        self.ds.add_frame(frame)
        self.n_frames += 1
        if self.n_frames % int(self.fps) == 0:
            self.get_logger().info(f"  recording... {self.n_frames} frames "
                                   f"({self.n_frames / self.fps:.1f} s)")

    def _start(self) -> None:
        if not self.have_js or any(self.rgb[c] is None for c in self.cams):
            self.get_logger().warn("not all streams up yet -- not starting")
            return
        # ⚠️ REFUSE TO RECORD WITHOUT AN ACTION STREAM. Nothing else catches this: with the teleop
        # node down, /zero/eef_target is simply never published, `action` stays at its zero
        # initialiser, and the recorder happily writes a perfectly well-formed episode in which
        # every action is the zero vector. It looks like a real dataset, trains to nothing, and
        # the only clue is a column of zeros nobody thinks to plot. Caught it in the first test
        # run of this node.
        if not self.have_action:
            self.get_logger().error(
                "no /zero/eef_target seen -- is the teleop node running? "
                "Refusing to record: every action would be zero.")
            return
        self._ensure_dataset()
        self.recording = True
        self.n_frames = 0
        self.get_logger().info(f"=== episode {self.n_episodes} START ===")

    def _stop(self, save: bool) -> None:
        self.recording = False
        if save and self.n_frames:
            self.ds.save_episode()
            self.n_episodes += 1
            self.get_logger().info(
                f"=== episode SAVED: {self.n_frames} frames, "
                f"{self.n_frames / self.fps:.1f} s  (total {self.n_episodes}) ===")
        else:
            if self.ds is not None:
                self.ds.clear_episode_buffer()
            self.get_logger().info(f"=== episode DISCARDED ({self.n_frames} frames) ===")
        self.n_frames = 0


def main() -> None:
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.recording:
            node._stop(save=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
