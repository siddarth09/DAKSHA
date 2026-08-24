# ZERO — cross-embodiment manipulation transfer

ZERO asks a narrow question: **if you teleoperate one robot arm, can the resulting policy drive a
completely different arm?** Not fine-tuned on the new arm — driven directly.

The bet is that it can, provided the policy never learns anything about the arm it was recorded
on. So the action and the proprioceptive state are expressed only as **end-effector poses in the
table frame** — `pos(3) + rot6d(6) + grip(1)` per hand, 20 values for two hands. No joint angles
anywhere the policy can see them. A 6-DoF arm and a 7-DoF arm produce the *same* 20 numbers for
the same physical motion, so a policy trained on one is at least well-defined on the other. This
is the same reasoning behind [Mirage](https://arxiv.org/abs/2402.19249), which shows that
Cartesian-space policies transfer zero-shot between arms where joint-space policies cannot.

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

**Auxiliary** — recorded for repair and analysis, *not* for the policy:

| key | shape | why it exists |
| --- | --- | --- |
| `observation.joint_pos` / `joint_vel` | 16 | makes the dataset **repairable** — see below |
| `observation.ft` | 24 | fingertip wrench, 4 fingers × (force 3 + torque 3) |
| `observation.ik_residual` | 6 | tracking error; lets you drop frames where the arm was not following the operator |
| `observation.grip_cmd` | 2 | latched gripper command, so "asked to close" is separable from "reached closed" |
| `observation.depth.*` | 224×224 | float32 metres, one per camera |

Joint angles **cannot** be policy-visible: the reBot has 6 per arm and Panda 7, so a policy
trained on one cannot even be fed the other — the vector length differs.

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
ros2 launch zero_bringup rebot.launch.py

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

Only one simulator may run at a time. Launching a second gives you two `controller_manager`s on
one ROS graph, both answering each other's service calls, and every controller spawner fails with
a misleading message — so `rebot.launch.py` refuses to start if one is already up.

### Gamepad

| control | action |
| --- | --- |
| **LB** / **RB** | hold to drive the left / right arm — nothing held, nothing moves |
| left stick | X / Y in the table plane |
| right stick | Z (up/down) and yaw |
| **Y** | toggle the selected gripper — *latches*, so it stays closed while you carry |
| **A** | return both arms home |
| **START** | resync the target to the measured pose |
| **BACK** | start / stop recording an episode |
| **B** | discard the episode in progress |

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
- **Loading a local dataset** with `LeRobotDataset(repo_id=...)` tries to reach HuggingFace for
  version info and 404s on a repo that only exists locally. Writing is unaffected and the parquet
  is directly readable, but this needs solving before training.
- **Post-compile MuJoCo edits are silently ignored.** Writing `body_gravcomp`, `geom_friction` or
  `actuator_forcerange` on an already-compiled model measures byte-identical to not doing it.
  Everything must be set on the `MjSpec` at generation time.

---

## Where this is going

The reBot → Panda transfer is the *experiment*, not the destination. Two fixed-base arms on a
table are the cleanest possible test of the claim: identical task, identical scene, identical
action space, and nothing different except the arm.

The destination is the **Unitree G1** — a humanoid, doing the same bimanual pick-and-handover with
its own arms and grippers. A humanoid changes what "the same action space" means: the base can
move, the cameras move with the head, and the workspace is defined by the whole body rather than a
bolted-down shoulder. Getting reBot → Panda to work first means that when the G1 arrives, the only
new variable is the embodiment — the action representation, the recording pipeline, the dataset
schema and the evaluation are all already settled and already known to transfer once.

If a policy recorded on a 6-DoF hobby arm can drive a Panda without ever seeing one, the same
argument should carry to a humanoid. That is the thing worth finding out.
