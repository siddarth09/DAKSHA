# ZERO — cross-embodiment manipulation transfer

**VLFA — Vision · Language · Force · Action.**

ZERO asks a narrow question: **if you teleoperate one robot arm, can the resulting policy drive a
completely different arm?** Not fine-tuned on the new arm — driven directly.

The bet is that it can, provided the policy never learns anything about the arm it was recorded
on. So the action and the proprioceptive state are expressed only as **end-effector poses in the
table frame** — `pos(3) + rot6d(6) + grip(1)` per hand, 20 values for two hands. No joint angles
anywhere the policy can see them. A 6-DoF arm and a 7-DoF arm produce the *same* 20 numbers for
the same physical motion, so a policy trained on one is at least well-defined on the other. This
is the same reasoning behind [Mirage](https://arxiv.org/abs/2402.19249), which shows that
Cartesian-space policies transfer zero-shot between arms where joint-space policies cannot.

The policy is **VLFA**: it reads vision (three cameras), language (the task instruction) and
**force** (fingertip load), and emits those EEF actions. Force is a first-class input, not a
diagnostic — contact is most of what separates a grasp that will hold from one that will slip, and
it is information the cameras cannot supply. Keeping force *transferable* takes the same care as
the action space, and is described below.

Demonstrations are collected by hand on the **Seeed reBot** (source), then replayed and evaluated
on the **Franka Panda** (target). Both work the identical scene: the same table, the same can, the
same tray, the same physics.

### Source — Seeed reBot, 6 DoF per arm

![reBot workstation](docs/rebot_scene.png)

### Target — Franka Panda, 7 DoF per arm

![Panda workstation](docs/panda_scene.png)

The task is **pick → hand over → place**: the left arm lifts the can, passes it to the right arm
mid-table, and the right arm sets it in the tray. It is a handover by construction, not by choice —
the base separation (0.90 m) was solved so that the can is reachable by the left arm *only* and
the tray by the right arm *only*, so an arm cannot quietly skip the handover and turn the task
into pick-and-place.

---

## What the policy sees, and what is merely stored

This distinction is the whole design, and it is easy to get wrong because LeRobot puts both in one
dataset.

**Policy-visible** — must be embodiment-invariant:

| key | shape | meaning |
| --- | --- | --- |
| `observation.state` | 20 | per hand `pos(3) + rot6d(6) + grip(1)`, **measured** via forward kinematics |
| `action` | 20 | the same 20 values, **commanded** by the operator |
| `observation.images.front` | 224×224×3 | scene view, roughly a humanoid's viewpoint |
| `observation.images.left_wrist` | 224×224×3 | left gripper |
| `observation.images.right_wrist` | 224×224×3 | right gripper |
| `observation.force` | 14 | per hand: net 6-D wrench **in the tool frame** + normalised squeeze |
| `task` | string | the language instruction — the **L** in VLFA |

**Auxiliary** — recorded for repair and analysis, *not* for the policy:

| key | shape | why it exists |
| --- | --- | --- |
| `observation.joint_pos` / `joint_vel` | 16 | makes the dataset **repairable** — see below |
| `observation.ft` | 24 | RAW per-finger wrench, each in its own sensor frame — kept for analysis, not fed to the policy |
| `observation.ik_residual` | 6 | tracking error; lets you drop frames where the arm was not following the operator |
| `observation.grip_cmd` | 2 | latched gripper command, so "asked to close" is separable from "reached closed" |
| `observation.depth.*` | 224×224 | float32 metres, one per camera |

### Making force transferable

Force cannot be handed to the policy raw. Each fingertip sensor reports in its own frame, and those
frames differ between the arms — measured, the reBot's finger frames sit 90° from its tool frame
while Panda's first finger is identity. The raw 24-dim vector is therefore *not the same physical
measurement* on the two robots.

`observation.force` resolves it into one shared quantity, per hand:

- **net wrench (6)** — the finger wrenches rotated into the **tool frame** and summed. External
  load: the object's weight, or the gripper pressing on something.
- **squeeze (1)** — the *mean* of the per-finger force magnitudes, divided by that robot's own
  grip-force cap (reBot 15 N, Panda 40 N), so it is dimensionless and comparable.

Both are needed, because the sum destroys the one that matters most. On a symmetric pinch the pads
push against each other and cancel: gripping the can gives 15.6 N and 14.5 N per pad, so the
**sum is ~1 N while the squeeze is ~15 N**. A first version reported only the sum and read 0.098
during a firm grasp. Measured on the fixed version: squeeze goes 0.05 open → 0.88 gripping, while
the net wrench stays flat.

Sum and mean are each defined for *any* number of fingers, so a two-finger jaw and the G1's
three-finger Dex3 present the same feature. The sensor-to-tool rotations are constant — every
gripper joint is prismatic, so fingers translate and never rotate (verified: 0.00e+00 change
between fully open and fully closed) — so they are measured once at generation time.

Joint angles, by contrast, **cannot** be made policy-visible: the reBot has 6 per arm and Panda 7,
so a policy trained on one cannot even be fed the other — the vector length differs.

But they are recorded anyway, because they make the dataset repairable. The tool-centre point of
this rig was wrong three times during development (113 mm out, then 39 mm out, then correct). Any
EEF pose recorded against a wrong TCP is permanently mislabelled — *unless* the joint angles are
stored, in which case every pose can be recomputed by forward kinematics with the corrected value.
The same data is what a Mirage-style re-render of the other robot needs, since the wrist images
are embodiment-specific too.

---

## Setup

```bash
cd ~/projects25
colcon build --symlink-install --packages-select zero_description zero_bringup zero_control
source install/setup.bash
```

Requires ROS 2 Jazzy, `mujoco_ros2_control`, `mujoco`, `pinocchio`, and (for recording)
`lerobot` and `rerun-sdk`.

> **Where to run things.** Every `ros2` command below runs from the **workspace root**
> (`~/projects25`) — the `install/...` paths in them are relative to it. Every generator script
> runs from the **package directory** (`~/projects25/src/ZERO`), because they import
> `scripts/zero_layout.py`. Build with `colcon` from the workspace root only; there is a stray
> `build/ install/ log/` inside `src/ZERO` from an accidental build, and sourcing that one gives
> you stale descriptions.

## Recording demonstrations

Four terminals, in order. Swap `rebot` for `panda` to drive the other arm.

```bash
# 1. simulator + controllers + IK
#    can_x / can_y / can_yaw move the can; defaults are the solved pick position
ros2 launch zero_bringup rebot.launch.py
ros2 launch zero_bringup rebot.launch.py can_x:=0.30 can_y:=0.42 can_yaw:=0.5

# 2. gamepad teleoperation
ros2 launch zero_bringup rebot_teleop.launch.py

# 3. live monitor: cameras, joint plots, fingertip forces
ros2 run zero_control rerun_viewer --ros-args \
    --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml \
    -p depth:=true

# 4. dataset recorder
ros2 run zero_control record --ros-args \
    --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml \
    -p root:=$HOME/zero_data/rebot_pick_place \
    -p task:="pick up the can and place it in the tray"
```

Each episode is announced out loud, the way `lerobot_record` does — "Recording episode 3",
"Episode saved. 4 recorded", "Reset the environment", "Episode discarded. Re-record". Teleoperating
means both hands on the pad and eyes on the sim, so the terminal is the one place you are
guaranteed not to be looking. It uses lerobot's own `log_say` (`spd-say` on Linux) and is
non-blocking on purpose: a blocking utterance inside the 10 Hz recording timer would drop frames
from the episode it is announcing. Silence it with `-p play_sounds:=false`.

> **⚠️ Let the recorder finish.** LeRobotDataset only stamps the parquet footer when its writers
> are closed, so the recorder calls `finalize()` on the way out. Kill it before that completes and
> you get a directory that looks perfect — right `meta/info.json`, right episode and frame counts,
> a multi-megabyte data file — that pyarrow refuses to open with *"Parquet magic bytes not found in
> footer"*. The frames are gone. A 2-episode take was lost that way while this was being built.
> **"Dataset saved" is the signal that it is safe to exit** — not "Stop recording", which is spoken
> before the close.

Only one simulator may run at a time. Launching a second gives you two `controller_manager`s on
one ROS graph, both answering each other's service calls, and every controller spawner fails with
a misleading message — so `rebot.launch.py` refuses to start if one is already up.

### Moving the can

`can_x`, `can_y` and `can_yaw` are launch arguments on the sim, in table-frame metres/radians.
Defaults come from the solved pick position, so a bare launch is unchanged.

The can's pose lives in the MJCF's `home` keyframe, so the launch reads that keyframe, substitutes
the can's free-joint block, and writes a `<key .../>` file that
mujoco_ros2_control's `override_start_position_file` hardware parameter picks up. Everything else
— arm poses, gripper, tray — is copied verbatim, so only the can moves. Off-table values are
rejected at launch rather than spawning the can somewhere unrecordable.

There is no `can_z`: the resting height is a function of the can's own geometry, and hand-setting
it invites a can that floats or is half-buried.

**This is per-launch, not per-episode.** Randomising the can between episodes of one session needs
the pose changed at runtime, which the installed mujoco_ros2_control cannot do — it exposes only
`reset_world`, `set_pause` and `step_simulation`. Upstream has a `SetFreeJointState` service that
would do it; this build predates it. Until then, per-position batches mean restarting the sim.

### Resuming a dataset

Point the recorder at an existing `root` and it **appends** — it reports
`RESUMING dataset ...: N episodes already recorded`, says "Resuming, N episodes on disk", and
numbers the next episode N. So a 30-episode session can be split across restarts, breaks, or a
crash.

`LeRobotDataset.create` raises `FileExistsError` on an existing root, so the recorder loads instead
of creating when it finds one.

**A directory with `meta/info.json` is not necessarily a usable dataset.** That file is written up
front; `meta/episodes/*.parquet` — the episode index — is only written when the writers close. A
killed recorder therefore leaves `info.json` claiming N episodes with no index. Handing that to
LeRobotDataset does not fail locally: it reads the missing index as "not downloaded yet" and goes
to the Hugging Face Hub, which 404s for a dataset that exists only on disk, and the traceback then
points at huggingface_hub instead of at the half-written directory. The recorder checks for the
index first and says so in one sentence, including how to check whether the frame data survived.

To open a finished dataset for training:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="zero/rebot", root="~/zero_data/rebot_pick_place")
```

Before appending, the recorder compares the dataset's feature set against what it would write and
refuses if they differ — the schema here has changed repeatedly (force promoted into the
observation, depth added, joint velocities added), and mixing two schemas in one directory
produces a dataset that is quietly two incompatible halves.

### Gamepad

| control | action |
| --- | --- |
| **LB** / **RB** | hold to drive the left / right arm — nothing held, nothing moves |
| left stick | X / Y in the table plane |
| right stick | Z (up/down) and yaw |
| **Y** | toggle the selected gripper — *latches*, so it stays closed while you carry |
| **A** | return both arms home |
| **START** | resync the target to the measured pose |
| **X** | start / stop recording an episode |
| **B** | discard the episode in progress — the index is reused, so nothing is skipped |

Or from the keyboard, using `lerobot_record`'s own bindings so muscle memory carries over:
**SPACE** start · **RIGHT** end and save · **LEFT** discard and re-record · **ESC** stop and exit.
`-p keyboard:=false` disables them.

The dead-man doubles as the arm selector deliberately: the teleop node integrates stick velocity
into an absolute pose, and an integrator with no dead-man accumulates stick noise while you are
looking at the screen. It also makes "which arm am I driving" unambiguous.

Verify the axis map on a fresh gamepad with `ros2 run zero_control joy_probe`, which names each
control as you move it. Clones keep the XInput button indices but not always the axis *signs*;
flip `sign_x/y/z/yaw` in `zero_bringup/config/teleop.yaml`.

---

## Regenerating the robot descriptions

Every description is generated from one file, `scripts/zero_layout.py`. Nothing is hand-edited —
the MJCF, the URDF, the controller YAML and the launch files all come from that single registry,
so the two descriptions cannot drift apart.

```bash
python3 scripts/gen_scene.py   rebot   # MJCF: arms, table, can, tray, cameras, sensors
python3 scripts/gen_urdf.py    rebot   # URDF + <ros2_control> block
python3 scripts/gen_bringup.py rebot   # controller YAML + launch files
python3 scripts/check_parity.py rebot  # assert everything still agrees
```

**Run `check_parity.py` after any change.** `mujoco_ros2_control` binds joints, cameras and
sensors across the two descriptions **by name**, and a mismatch does not raise — the parts that
match keep working, so the symptom is "the gripper does nothing" or "no images", which reads as a
tuning or networking fault. The check asserts joint names, actuator transmissions, camera names,
force/torque sensor pairs, URDF-vs-MJCF forward kinematics (0.003 mm), and that the IK's tool
point coincides with where the fingers actually close (0.05 mm).

Supporting tools:

| script | purpose |
| --- | --- |
| `measure_tcp.py` | measure where the pads meet; prints the `eef_offset` to paste into the registry |
| `solve_home.py` | solve a home pose with IK — reports honestly when one is unreachable |
| `findhome.py` | older random search for a home pose, kept for comparison |
| `solve_handover.py` | solve the base separation that forces a genuine handover |
| `hero_shot.py` | render the images in this README |

---

## Known constraints

Things that are measured, not guessed, and that will bite if you forget them.

- **10 Hz, not 30.** The camera rate is capped by MuJoCo's offscreen rendering: three cameras at
  224×224 gives 10.17 Hz, five gave 5.00. Resolution is nearly free; each additional camera is
  not. `top` and `side` are commented out in `SCENE_CAMS` and cost frame rate to restore.
- **The reBot is at its workspace limit.** It cannot achieve a top-down approach at the can's
  position, so grasps there are diagonal and less reliable than Panda's. The base separation was
  solved against an older, incorrect tool point and is worth re-solving.
- **Grasp the can in its upper half.** Both arms hold it when grasped 40–70 % up its height.
- **Depth reads out to ~88 m** (the skybox). Clip it before training.
- **~170 MB per minute** of recording, mostly depth. Fifty 30-second demos is roughly 1.4 GB.
- **Object poses are not recorded**, so success cannot be auto-labelled — episodes need
  annotating, or add the poses and re-record.
- **A 404 from huggingface_hub when opening a local dataset** means the local metadata is
  incomplete, not that you need credentials. The Hub is only consulted when `load_metadata()`
  fails — usually a missing `meta/episodes/` index from a killed recorder.
- **Post-compile MuJoCo edits are silently ignored.** Writing `body_gravcomp`, `geom_friction` or
  `actuator_forcerange` on an already-compiled model measures byte-identical to not doing it.
  Everything must be set on the `MjSpec` at generation time.

---

## Where this is going

The reBot → Panda transfer is the *experiment*, not the destination. Two fixed-base arms on a
table are the cleanest possible test of the claim: identical task, identical scene, identical
action space, and nothing different except the arm.

The destination is the **Unitree G1** — a humanoid, doing the same bimanual pick-and-handover with
its own arms and three-finger Dex3 hands. A humanoid changes what "the same action space" means: the base can
move, the cameras move with the head, and the workspace is defined by the whole body rather than a
bolted-down shoulder. Getting reBot → Panda to work first means that when the G1 arrives, the only
new variable is the embodiment — the action representation, the recording pipeline, the dataset
schema and the evaluation are all already settled and already known to transfer once.

If a policy recorded on a 6-DoF hobby arm can drive a Panda without ever seeing one, the same
argument should carry to a humanoid. That is the thing worth finding out.
