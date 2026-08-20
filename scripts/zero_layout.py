"""Shared layout constants for project ZERO's bimanual reBot workstation.

THE POINT OF THIS FILE: the URDF and the MJCF are two independent descriptions of the same
robot, and `mujoco_ros2_control` binds them together BY JOINT NAME across two topics
(`/robot_description` for the URDF, `/mujoco_robot_description` for the MJCF). If the names
or the mount poses drift apart, nothing throws -- the plugin reports "Joint '%s' not found in
the URDF joint data" for the lucky cases and silently ignores the rest, so a gripper can end
up never moving while everything looks fine. Both generators import from here so that a drift
is impossible rather than merely unlikely.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- sources ---------------------------------------------------------------------------
MENAGERIE_ARM = ROOT / "robots" / "seeed_rebot_devarm" / "seeed_rebot_devarm.xml"
# Seeed's official RS (RobStride) URDF -- the variant menagerie models, per its README's
# RS-06 / RS-00 motor torque limits. The DM variant uses Damiao actuators and different specs.
SEEED_URDF = Path.home() / "rebotarm_ros2/src/rebotarm_bringup/description/urdf/00-arm-rs_asm-v3.urdf"
SEEED_MESHES = Path.home() / "rebotarm_ros2/src/rebotarm_bringup/description/meshes_rs"

# --- outputs ---------------------------------------------------------------------------
PKG = ROOT / "zero_description"
OUT_URDF = PKG / "urdf" / "zero_bimanual.urdf"
OUT_MJCF = PKG / "mjcf" / "zero_bimanual.xml"
OUT_PNG = ROOT / "scenes" / "bimanual.png"

# --- joint names -----------------------------------------------------------------------
# `gripper_joint1/2` come from Seeed's official URDF. Menagerie ships the same two joints as
# `joint_left`/`joint_right`, so OUR COPY of the MJCF was renamed to match the URDF rather
# than the other way round -- upstream's MoveIt config and ros2_controllers.yaml already
# reference the URDF names, and keeping them lets those configs work unmodified.
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS
SIDES = ("left", "right")


def prefixed(side: str, name: str) -> str:
    return f"{side}_{name}"


def all_prefixed_joints() -> list[str]:
    return [prefixed(s, j) for s in SIDES for j in ALL_JOINTS]


# --- workstation geometry --------------------------------------------------------------
# Measured from 20k random configurations of one arm (scripts/measure_reach.py):
#   points along +x at qpos=0; gripper_end at (0.302, 0, 0.218) from base
#   max reach 0.909 m, p95 0.828 m, reaches to z = -0.37 below its own base
# BASE_SEP 0.44 m puts the whole centreline inside both envelopes, which is what makes this
# genuinely bimanual instead of two arms working independently.
TABLE_TOP_Z = 0.75
TABLE_HALF = (0.60, 0.70, 0.02)  # widened 2026-08-20: two Pandas crowded the old 0.9x0.9 top
TABLE_CENTER_XY = (0.05, 0.0)
BASE_SEP = 0.44
# ⚠️ MEASURED, not chosen. Phase-0.5 sensitivity sweep (scripts/sens.py) over table height x
# mount position: table height barely matters, BASE_X dominates. Shared reBot+G1 table volume,
# at G1 standoffs of 0.10 / 0.20 / 0.30 m:
#     BASE_X -0.25 : 11.6 /  5.4 / 1.3 L   <- original, G1 must press against the table
#     BASE_X -0.15 : 19.7 / 10.9 / 4.7 L
#     BASE_X -0.05 : 27.7 / 16.3 / 7.6 L   <- chosen; 5.6x the margin at 0.30 m standoff
# Moving the arms forward costs work-area depth in front of them (still ~37 cm to the front
# edge) and buys the G1 room to stand without leaning over the table.
BASE_X = -0.05

# Mount pose per side, in world coordinates. Used by BOTH generators.
MOUNTS = {
    "left": (BASE_X, +BASE_SEP / 2, TABLE_TOP_Z),
    "right": (BASE_X, -BASE_SEP / 2, TABLE_TOP_Z),
}

# menagerie's "raised" keyframe, SIGN-FLIPPED to match the URDF convention that
# fix_mjcf_conventions.py now enforces. menagerie ships it as (0, -0.7, -1.1); with the six arm
# axes negated those same numbers would drive the arm DOWN into the table instead of clear of it.
HOME_QPOS = (0.315, 1.139, 0.570, 0.734, -0.432, 3.047)  # aimed at LOOK_AT, see scripts/findhome.py
GRIPPER_OPEN = (0.0, 0.0)


# ── CAMERAS ────────────────────────────────────────────────────────────────────────────
# Named to match the convention in siddarth09/Panda_mujoco, which uses RealSense D435i bodies
# at top / table / side / wrist. Scaled to this table: top at z=0.75, spans x[-0.40,0.50],
# y[-0.45,0.45]; the Phase-0.5 task volume is x[0.12,0.52] y[±0.36] z[0.76,1.00], centroid
# ~(0.36, 0, 0.90). LOOK_AT is the task centroid, so every scene camera frames the work area
# rather than the whole table.
#
# ⚠️ WRIST CAMS ARE THE PRIMARY VIEW for cross-embodiment, not the scene cams. A wrist view puts
# the base/torso out of frame, so the manipulator-vs-humanoid difference stops being visible;
# scene cams show two completely different robots and cannot transfer (this is why Mirage-style
# cross-painting was dropped -- see research/README.md).
# NB: overwritten at the bottom of this file once PICK_POS/PLACE_POS exist -- the cameras and
# the home-pose search must aim at the actual task, not at a hand-picked point above it.
LOOK_AT = (0.36, 0.0, 0.90)
SCENE_CAMS = {
    # name:       (eye position,            fovy)
    "top":        ((0.30, 0.00, 1.60), 58),   # straight down over the work area
    "front":      ((1.15, 0.00, 1.15), 58),   # across the table, roughly the G1's viewpoint
    "side":       ((0.30, -1.05, 1.10), 58),  # side elevation, for depth disambiguation
}
WRIST_CAM_FOVY = 70          # wide: the object is only ~10 cm from the lens
WRIST_CAM_POS = (0.0, 0.0, 0.035)   # on gripper_end, just behind the finger roots
CAM_RES = (640, 480)


def all_cameras() -> list[str]:
    """Every camera name in the MJCF, which is also every <sensor> name the URDF must declare.

    ⚠️ mujoco_ros2_control matches cameras to sensors BY NAME ("For cameras, the sensor name
    _must_ match the camera name in the MJCF" -- upstream's own demo URDF). A mismatch does not
    error: the camera simply never publishes, so RViz shows nothing and it looks like a
    networking or QoS problem. Same trap as the joints, same fix -- one source of truth here,
    asserted by scripts/check_parity.py.
    """
    return list(SCENE_CAMS) + [f"{s}_wrist" for s in SIDES]


# Topic namespace per camera: /zero/<cam>/{image_raw,camera_info,depth}
CAM_NS = "/zero"


ROBOCASA = Path("/home/sid/projects25/src/robocasa/robocasa/models/assets/objects/lightwheel")
PICK_OBJECT = ROBOCASA / "lemon_wedge" / "LemonWedge001" / "model.xml"
PLACE_TARGET = ROBOCASA / "tray" / "Tray001" / "model.xml"

# ── HANDOVER GEOMETRY, solved not guessed (scripts/solve_handover.py) ──────────────────
# The task is PICK -> HANDOVER -> PLACE, so it is only a handover if the picking arm CANNOT
# reach the tray. Otherwise the policy skips the handover and the task silently degenerates
# into pick-and-place. Constraints: pick reachable by LEFT only, place by RIGHT only, the
# handover pose by BOTH (6 cm margin on the "cannot reach" side).
# Solved separations:  reBot 0.75-1.20 m  ·  Panda 0.55-1.05 m
# The TASK must be identical across embodiments -- only the mounts may differ - so we take the
# INTERSECTION [0.75, 1.05] and use its midpoint for both robots.
HANDOVER_SEP = 0.90

PICK_POS = (0.34, +HANDOVER_SEP / 2, TABLE_TOP_Z + 0.03)    # in front of the LEFT arm
PLACE_POS = (0.34, -HANDOVER_SEP / 2, TABLE_TOP_Z + 0.015)  # in front of the RIGHT arm
HANDOVER_POS = (0.34, 0.0, TABLE_TOP_Z + 0.13)              # centreline, above the table

# Each arm aims at ITS OWN role's target. Aiming both at one shared point made them face each
# other, which is wrong for a handover and was visible immediately in the viewer.
HOME_TARGET = {"left": PICK_POS, "right": PLACE_POS}

LOOK_AT = HANDOVER_POS

# ══════════════════════════════════════════════════════════════════════════════════════
# EMBODIMENT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════════════
# The cross-embodiment chain is reBot (source) -> Panda -> G1. Everything above this line is
# reBot-specific and is what the URDF generator / parity check / ros2_control path use, because
# only the SOURCE embodiment needs teleoperation and ros2_control. Targets only need to be
# stepped and IK'd, so they live here as plain MJCF specs.
#
# MEASURED per robot (do not edit by hand -- see the printouts in gen_scene.py):
#   reBot : reach max 0.909 m (p95 0.83) · gripper 100 mm open · 6 arm DoF
#   Panda : reach max 1.189 m (p95 1.12) · gripper  80 mm open · 7 arm DoF
#   Dex3  : gripper 124 mm open (thumb-tip to index-tip)
# ⚠️ THE BINDING APERTURE IS PANDA'S 80 mm, so the shared object must be ~44 mm to leave grasp
# margin on all three. That is tighter than the 55 mm the reBot-only gate suggested.
OBJECT_WIDTH = 0.044

ROBOTS = {
    "rebot": {
        "mjcf": ROOT / "robots" / "seeed_rebot_devarm" / "seeed_rebot_devarm.xml",
        "arm_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        "gripper_joints": ("gripper_joint1", "gripper_joint2"),
        "eef_body": "gripper_end",
        "eef_offset": (0.0, 0.0, 0.10),
        "wrist_cam_pos": (0.0, 0.0, 0.035),
        "wrist_cam_xyaxes": (1, 0, 0, 0, -1, 0),   # look along the body's +z (approach axis)
        "base_sep": HANDOVER_SEP,
        "base_x": -0.05,
        # Per-side, searched (scripts/findhome.py <robot> <side>). NOT mirrored from one pose:
        # the correct mirror is robot-specific and copying left->right aimed the arm 0.43 m away.
        "home": {
            "left": (-0.352, 1.314, 0.706, 0.067, -1.185, 2.366),
            "right": (0.422, 1.269, 0.446, 0.649, 1.006, -3.073),
        },
        "grip_open": (0.0, 0.0),
        # menagerie names each actuator after its joint, so the ros2_control parity check can
        # compare the two sets directly.
        "actuators_match_joints": True,
        # ── ros2_control / URDF side (source embodiment: needs teleop) ──
        "urdf": SEEED_URDF,
        "urdf_root": "base_link",
        "urdf_skip_links": (),
        "mesh_src": SEEED_MESHES,
        "mesh_glob": ("*.STL",),
        "mesh_remap": None,
        # One command interface PER ACTUATOR. The reBot has TWO real gripper actuators plus an
        # <equality> coupling the fingers, so both are declared; Panda has ONE actuator driving
        # both fingers, so it declares one. Copying either pattern blindly is wrong.
        "ros2_control_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                                "gripper_joint1", "gripper_joint2"),
        "arm_ctrl_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        "grip_ctrl_joints": ("gripper_joint1", "gripper_joint2"),
    },
    "panda": {
        # ⚠️ Sid's ros2-ready copy, NOT menagerie's panda.xml. menagerie drives the gripper through a
        # TENDON actuator (`tendon="split"`, ctrlrange remapped to 0-255), which ros2_control cannot
        # command by joint name -- check_parity caught it as "finger_joint1 has no actuator". His
        # version replaces it with a direct joint actuator on finger_joint1 and drops the tendon
        # (ntendon 1 -> 0), ctrlrange 0-0.4.
        "mjcf": ROOT / "robots" / "franka_emika_panda" / "panda_ros2.xml",
        "arm_joints": tuple(f"joint{i}" for i in range(1, 8)),
        "gripper_joints": ("finger_joint1", "finger_joint2"),
        "eef_body": "hand",
        "eef_offset": (0.0, 0.0, 0.10),
        "wrist_cam_pos": (0.0, 0.0, 0.02),
        "wrist_cam_xyaxes": (1, 0, 0, 0, -1, 0),
        # Panda reaches 1.19 m vs the reBot's 0.91, and its base is bulkier, so it sits wider
        # apart. ⚠️ NOT yet put through the sensitivity sweep that set the reBot's numbers --
        # run the Panda leg of scripts/reach_gate.py before trusting these.
        "base_sep": HANDOVER_SEP,
        # Panda sits further back than the reBot: its links are much bulkier and at base_x=-0.05
        # the two arms crowded the work area. Per-robot on purpose -- what must stay IDENTICAL
        # across embodiments is the TASK (object + plate poses), not where the arms are bolted.
        "base_x": -0.30,
        # Searched, not menagerie keyframe 0 -- that pose aimed the gripper off to the side and the
        # wrist cameras saw only table legs. scripts/findhome.py panda: aim 0.989, down 0.964.
        "home": {
            "left": (0.207, 0.486, -0.147, -1.985, -0.542, 2.558, -0.688),
            "right": (0.277, 0.612, -0.336, -1.788, 0.422, 2.137, 0.005),
        },
        "grip_open": (0.04, 0.04),
        # ⚠️ Panda actuators are `actuator1..8`, NOT joint names, and there are 8 for 9 joints
        # (one actuator drives both fingers, with an <equality> coupling them -- same topology as
        # the reBot's gripper). Any name-based joint<->actuator check must skip this robot.
        "actuators_match_joints": False,
        # Sid's own URDF from siddarth09/Panda_mujoco, already proven with mujoco_ros2_control.
        # MJCF-derived, so it agrees with menagerie exactly: FK checked over 400 configs, 0.0000 mm
        # (unlike the reBot, whose independently-derived URDF had every joint axis inverted).
        "urdf": ROOT / "robots" / "franka_emika_panda" / "panda_ros2.urdf",
        # `link0` is Panda's real base. Its URDF also ships its own `world` link, which must be
        # dropped -- our generated file supplies a single shared `world` root for both arms, and
        # two copies of `world` would make the tree a forest.
        "urdf_root": "link0",
        "urdf_skip_links": ("world",),
        "mesh_src": ROOT / "robots" / "franka_emika_panda" / "assets",
        "mesh_glob": ("*.obj", "*.stl"),
        # The URDF references converter output (`converted_link0_0_e6ebedff.obj`) that was never
        # committed. Menagerie ships the same meshes as `link0_0.obj`, so remap rather than
        # regenerate: converted_<part>_<idx>_<hex>.obj -> <part>_<idx>.obj
        "mesh_remap": r"converted_(.+)_([0-9a-f]{8})\.obj",
        "ros2_control_joints": tuple(f"joint{i}" for i in range(1, 8)) + ("finger_joint1",),
        "arm_ctrl_joints": tuple(f"joint{i}" for i in range(1, 8)),
        "grip_ctrl_joints": ("finger_joint1",),
    },
}


def robot_all_joints(key: str) -> tuple[str, ...]:
    r = ROBOTS[key]
    return tuple(r["arm_joints"]) + tuple(r["gripper_joints"])


def robot_prefixed_ros2_joints(key: str) -> list[str]:
    """Prefixed joint names for the <ros2_control> block -- one per ACTUATOR, both arms."""
    return [prefixed(s, j) for s in SIDES for j in ROBOTS[key]["ros2_control_joints"]]


def robot_mounts(key: str) -> dict[str, tuple[float, float, float]]:
    r = ROBOTS[key]
    return {
        "left": (r["base_x"], +r["base_sep"] / 2, TABLE_TOP_Z),
        "right": (r["base_x"], -r["base_sep"] / 2, TABLE_TOP_Z),
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# THE TASK  — identical for every embodiment, which is the whole point
# ══════════════════════════════════════════════════════════════════════════════════════
# Robocasa lightwheel assets, measured from mesh vertices (NOT geom_rbound, which reports
# bounding-sphere radii and made a flat tray read as 460 mm cubed):
#     lemon_wedge  18 x  23 x  54 mm   -> pick object; 18-23 mm grasp width, comfortably inside
#                                         the 44 mm ceiling set by Panda's 80 mm gripper
#     tray         28 x 220 x 396 mm   -> the "plate" to place onto
# Alternative if the task proves too easy: `marshmallow` at 42 x 45 x 60 mm sits right ON the
# 44 mm limit, so it exercises the aperture constraint for all three grippers.
def robot_all_joints(key: str) -> tuple[str, ...]:
    r = ROBOTS[key]
    return tuple(r["arm_joints"]) + tuple(r["gripper_joints"])


def robot_prefixed_ros2_joints(key: str) -> list[str]:
    """Prefixed joint names for the <ros2_control> block -- one per ACTUATOR, both arms."""
    return [prefixed(s, j) for s in SIDES for j in ROBOTS[key]["ros2_control_joints"]]


def robot_mounts(key: str) -> dict[str, tuple[float, float, float]]:
    r = ROBOTS[key]
    return {
        "left": (r["base_x"], +r["base_sep"] / 2, TABLE_TOP_Z),
        "right": (r["base_x"], -r["base_sep"] / 2, TABLE_TOP_Z),
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# THE TASK  — identical for every embodiment, which is the whole point
# ══════════════════════════════════════════════════════════════════════════════════════
# Robocasa lightwheel assets, measured from mesh vertices (NOT geom_rbound, which reports
# bounding-sphere radii and made a flat tray read as 460 mm cubed):
#     lemon_wedge  18 x  23 x  54 mm   -> pick object; 18-23 mm grasp width, comfortably inside
#                                         the 44 mm ceiling set by Panda's 80 mm gripper
#     tray         28 x 220 x 396 mm   -> the "plate" to place onto
# Alternative if the task proves too easy: `marshmallow` at 42 x 45 x 60 mm sits right ON the
# 44 mm limit, so it exercises the aperture constraint for all three grippers.
def robot_all_joints(key: str) -> tuple[str, ...]:
    r = ROBOTS[key]
    return tuple(r["arm_joints"]) + tuple(r["gripper_joints"])


def robot_prefixed_ros2_joints(key: str) -> list[str]:
    """Prefixed joint names for the <ros2_control> block -- one per ACTUATOR, both arms."""
    return [prefixed(s, j) for s in SIDES for j in ROBOTS[key]["ros2_control_joints"]]


def robot_mounts(key: str) -> dict[str, tuple[float, float, float]]:
    r = ROBOTS[key]
    return {
        "left": (r["base_x"], +r["base_sep"] / 2, TABLE_TOP_Z),
        "right": (r["base_x"], -r["base_sep"] / 2, TABLE_TOP_Z),
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# THE TASK  — identical for every embodiment, which is the whole point
# ══════════════════════════════════════════════════════════════════════════════════════
# Robocasa lightwheel assets, measured from mesh vertices (NOT geom_rbound, which reports
# bounding-sphere radii and made a flat tray read as 460 mm cubed):
#     lemon_wedge  18 x  23 x  54 mm   -> pick object; 18-23 mm grasp width, comfortably inside
#                                         the 44 mm ceiling set by Panda's 80 mm gripper
#     tray         28 x 220 x 396 mm   -> the "plate" to place onto
# Alternative if the task proves too easy: `marshmallow` at 42 x 45 x 60 mm sits right ON the
# 44 mm limit, so it exercises the aperture constraint for all three grippers.