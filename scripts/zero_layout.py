"""Shared layout constants for project ZERO's bimanual reBot workstation.

The URDF and the MJCF are two independent descriptions of the same robot, and
`mujoco_ros2_control` binds them together by joint name across two topics
(`/robot_description` for the URDF, `/mujoco_robot_description` for the MJCF). If the names or
the mount poses drift apart nothing throws: the plugin reports "Joint '%s' not found in the URDF
joint data" for the lucky cases and ignores the rest, so a gripper can end up never moving while
everything looks fine. Both generators import from here so drift is impossible.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- sources ---
MENAGERIE_ARM = ROOT / "robots" / "seeed_rebot_devarm" / "seeed_rebot_devarm.xml"
# Seeed's official RS (RobStride) URDF, the variant menagerie models, per its README's RS-06 /
# RS-00 motor torque limits. The DM variant uses Damiao actuators and different specs.
SEEED_URDF = Path.home() / "rebotarm_ros2/src/rebotarm_bringup/description/urdf/00-arm-rs_asm-v3.urdf"
SEEED_MESHES = Path.home() / "rebotarm_ros2/src/rebotarm_bringup/description/meshes_rs"

# --- outputs ---
PKG = ROOT / "zero_description"
OUT_URDF = PKG / "urdf" / "zero_bimanual.urdf"
OUT_MJCF = PKG / "mjcf" / "zero_bimanual.xml"
OUT_PNG = ROOT / "scenes" / "bimanual.png"

# `gripper_joint1/2` come from Seeed's official URDF. Menagerie ships the same two joints as
# `joint_left`/`joint_right`, so our copy of the MJCF was renamed to match the URDF rather than
# the other way round: upstream's MoveIt config and ros2_controllers.yaml already reference the
# URDF names, and keeping them lets those configs work unmodified.
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS
SIDES = ("left", "right")


def prefixed(side: str, name: str) -> str:
    return f"{side}_{name}"


def all_prefixed_joints() -> list[str]:
    return [prefixed(s, j) for s in SIDES for j in ALL_JOINTS]


# Measured from 20k random configurations of one arm (scripts/measure_reach.py):
#   points along +x at qpos=0; gripper_end at (0.302, 0, 0.218) from base
#   max reach 0.909 m, p95 0.828 m, reaches to z = -0.37 below its own base
# BASE_SEP 0.44 m puts the whole centreline inside both envelopes, which is what makes this
# genuinely bimanual instead of two arms working independently.
TABLE_TOP_Z = 0.75
TABLE_HALF = (0.60, 0.70, 0.02)  # widened 2026-08-20: two Pandas crowded the old 0.9x0.9 top
TABLE_CENTER_XY = (0.05, 0.0)
BASE_SEP = 0.44
# Measured, not chosen. Phase-0.5 sensitivity sweep (scripts/sens.py) over table height x mount
# position: table height barely matters, BASE_X dominates. Shared reBot+G1 table volume, at G1
# standoffs of 0.10 / 0.20 / 0.30 m:
#     BASE_X -0.25 : 11.6 /  5.4 / 1.3 L   original; the G1 must press against the table
#     BASE_X -0.15 : 19.7 / 10.9 / 4.7 L
#     BASE_X -0.05 : 27.7 / 16.3 / 7.6 L   chosen; 5.6x the margin at 0.30 m standoff
# Moving the arms forward costs work-area depth in front of them (still ~37 cm to the front edge)
# and buys the G1 room to stand without leaning over the table.
BASE_X = -0.05

# Mount pose per side, in world coordinates. Used by BOTH generators.
MOUNTS = {
    "left": (BASE_X, +BASE_SEP / 2, TABLE_TOP_Z),
    "right": (BASE_X, -BASE_SEP / 2, TABLE_TOP_Z),
}

# menagerie's "raised" keyframe, sign-flipped to match the URDF convention that
# fix_mjcf_conventions.py enforces. menagerie ships it as (0, -0.7, -1.1); with the six arm axes
# negated those same numbers would drive the arm down into the table instead of clear of it.
HOME_QPOS = (0.315, 1.139, 0.570, 0.734, -0.432, 3.047)  # aimed at LOOK_AT, see scripts/findhome.py
GRIPPER_OPEN = (0.0, 0.0)


# Cameras. Named to match the convention in siddarth09/Panda_mujoco, which uses RealSense D435i
# bodies at top / table / side / wrist. Scaled to this table: top at z=0.75, spans x[-0.40,0.50],
# y[-0.45,0.45]; the Phase-0.5 task volume is x[0.12,0.52] y[+/-0.36] z[0.76,1.00], centroid
# ~(0.36, 0, 0.90). LOOK_AT is the task centroid, so every scene camera frames the work area
# rather than the whole table.
#
# The wrist cams are the primary view for cross-embodiment, not the scene cams. A wrist view puts
# the base and torso out of frame, so the manipulator-vs-humanoid difference stops being visible;
# scene cams show two completely different robots and cannot transfer. That is why Mirage-style
# cross-painting was dropped, see research/README.md.
#
# Overwritten at the bottom of this file once PICK_POS/PLACE_POS exist: the cameras and the
# home-pose search must aim at the actual task, not at a hand-picked point above it.
LOOK_AT = (0.36, 0.0, 0.90)
# Every camera costs frame rate, and the dataset can never be faster than its slowest
# observation. With 5 cameras: 5.00 Hz at 640x480, 5.60 at 320x240, 6.50 at 224x224, so
# resolution is almost free and each extra camera is not. Down to front plus two wrists.
# `top` and `side` are one line each to restore if a policy turns out to need them, but they
# have to be paid for in fps, and 5 Hz demos do not contain an operator's corrections.
SCENE_CAMS = {
    # name:       (eye position,            fovy)
    "front":      ((1.15, 0.00, 1.15), 58),   # across the table, roughly the G1's viewpoint
    # "top":      ((0.30, 0.00, 1.60), 58),
    # "side":     ((0.30, -1.05, 1.10), 58),
}
WRIST_CAM_FOVY = 70          # wide: the object is only ~10 cm from the lens
WRIST_CAM_POS = (0.0, 0.0, 0.035)   # on gripper_end, just behind the finger roots
# 224x224, the native input size for essentially every vision backbone a policy would use, so
# nothing is resized at train time. Larger buys no fps back anyway, see SCENE_CAMS.
CAM_RES = (224, 224)

# Camera publish rate, which has to be set explicitly. mujoco_ros2_control defaults to 5 Hz
# (measured 4.6), fine for eyeballing the sim and far too slow to imitate from, since an
# operator's corrections are not in the data at 5 Hz. Read as a hardware parameter from the URDF's
# <ros2_control><hardware> block (mujoco_system_interface.cpp reads `camera_publish_rate` there),
# so this is ours to set without touching that package. It also caps the LeRobot dataset fps: the
# dataset can never be faster than its slowest observation.
CAM_RATE_HZ = 30.0


def all_cameras() -> list[str]:
    """Every camera name in the MJCF, which is also every <sensor> name the URDF must declare.

    mujoco_ros2_control matches cameras to sensors by name ("For cameras, the sensor name _must_
    match the camera name in the MJCF", from upstream's own demo URDF). A mismatch does not
    error: the camera never publishes, so RViz shows nothing and it looks like a networking or
    QoS problem. Same trap as the joints, same fix: one source of truth here, asserted by
    scripts/check_parity.py.
    """
    return list(SCENE_CAMS) + [f"{s}_wrist" for s in SIDES]


# Topic namespace per camera: /zero/<cam>/{image_raw,camera_info,depth}
CAM_NS = "/zero"


ROBOCASA = Path("/home/sid/projects25/src/robocasa/robocasa/models/assets/objects/lightwheel")
# An upright can, and specifically a uniform cylinder. Two separate requirements:
#
#   Height. The original lemon wedge lies flat, 23 mm tall with its grasp centre ~11 mm off the
#   table, so a top-down gripper has to reach almost to the surface, which put the reBot's left
#   arm on its workspace boundary. An upright object carries its grasp centre ~75 mm up and the
#   reBot reaches it comfortably.
#
#   Shape. Juice006 met the height requirement and still failed in teleop, because it is a bottle:
#   its diameter is a uniform 49 mm up to 58% of its height and then breaks into a shoulder and a
#   35 mm cap. Any grasp above mid-body closes on the neck, so the operator gets a hold on the cap
#   instead of the body. A summary "uniformity" statistic over the whole grasp band hid this; the
#   diameter profile showed it immediately.
#
# Paprika012 measures 64-67 mm across from 5% to 75% of its height: a real cylinder, so any grasp
# in that band is on the body. Checked on both embodiments (grasp at 75% height, then lifted and
# translated 80 mm to stress the hold):
#   reBot  lift 100.2 mm, in-jaw slip 6.2 mm     Panda  lift 87.7 mm, slip 12.6 mm
# The near-cylinders that are cleaner in profile are not usable: Cinnamon007/005 tip Panda over at
# every grasp height, GlassCup025 tips the reBot. This is the only candidate both arms hold.
PICK_OBJECT = ROBOCASA / "paprika" / "Paprika012" / "model.xml"
PLACE_TARGET = ROBOCASA / "tray" / "Tray001" / "model.xml"

# Handover geometry, solved rather than guessed (scripts/solve_handover.py). The task is
# pick -> handover -> place, so it is only a handover if the picking arm cannot reach the tray.
# Otherwise the policy skips the handover and the task degenerates into pick-and-place.
# Constraints: pick reachable by left only, place by right only, the handover pose by both, with a
# 6 cm margin on the "cannot reach" side.
# Solved separations:  reBot 0.75-1.20 m,  Panda 0.55-1.05 m
# The task must be identical across embodiments and only the mounts may differ, so take the
# intersection [0.75, 1.05] and use its midpoint for both robots.
HANDOVER_SEP = 0.90

PICK_POS = (0.34, +HANDOVER_SEP / 2, TABLE_TOP_Z + 0.03)    # in front of the LEFT arm
PLACE_POS = (0.34, -HANDOVER_SEP / 2, TABLE_TOP_Z + 0.015)  # in front of the RIGHT arm
HANDOVER_POS = (0.34, 0.0, TABLE_TOP_Z + 0.13)              # centreline, above the table

# Each arm aims at its own role's target. Aiming both at one shared point made them face each
# other, which is wrong for a handover and was visible immediately in the viewer.
HOME_TARGET = {"left": PICK_POS, "right": PLACE_POS}

LOOK_AT = HANDOVER_POS

# ======================================================================================
# EMBODIMENT REGISTRY
# ======================================================================================
# The cross-embodiment chain is reBot (source) -> UR5e -> G1. Everything above this line is
# reBot-specific and is what the URDF generator, parity check and ros2_control path use, because
# only the source embodiment needs teleoperation and ros2_control. Targets only need to be stepped
# and IK'd, so they live here as plain MJCF specs.
#
# Measured per robot; do not edit by hand, see the printouts in gen_scene.py:
#   reBot : reach max 0.909 m (p95 0.83), gripper 100 mm open, 6 arm DoF
#   Panda : reach max 1.189 m (p95 1.12), gripper  80 mm open, 7 arm DoF
#   reBot : gripper 100 mm open (two 50 mm slide joints, coupled by an equality)
#   Dex3  : gripper 124 mm open (thumb-tip to index-tip)
# The binding aperture is Panda's 80 mm; the reBot opens wider, not narrower. Panda's throat at
# full open is 74 mm, so a 66 mm object leaves ~4 mm per side. It grasps reliably in test, but
# that is the tight constraint to respect when swapping the object, not the reBot.
#
# Gripper force limits, because the source models effectively have none: the reBot's gripper
# actuators ship forcerange +/-1904 N and Panda's is [0, 0], which in MuJoCo means unlimited. A
# position-controlled gripper commanded fully closed then drives straight through whatever it is
# holding: 11 mm of penetration into a 66 mm can, which crushes it, squirts it out of the jaws and
# makes contact forces meaningless. Real values make the grasp compliant, so it squeezes until the
# limit and then holds by friction, which is what the recorded demo should show. Applied by
# gen_scene to the gripper actuators only; the arm joints keep their own limits.
OBJECT_WIDTH = 0.066

# Mass of the pick object, forced rather than inherited. The robocasa assets are meshes with a
# default density, so a spice tin came out at 0.763 kg, roughly twice a full 330 ml can and heavy
# enough on a small arm that the grasp keeps failing under load. A real coke can is ~0.39 kg full
# and ~0.015 kg empty. Set in the MJCF at generation time: writing mass or friction onto an
# already-compiled model measures byte-identical, the same trap as body_gravcomp.
#
# This is an empty can (~15 g of aluminium), not a full one (~350 g). The asset it replaced weighed
# 1.19-1.51 kg (the mass is split across a nested body, so reading one body's own mass understates
# it), which is too much payload to keep hold of on a small arm.
#
# Being this light costs nothing in stability. The 7.6 g lemon mesh rocked through 24-58 deg
# because MuJoCo's convex-convex narrowphase gave it a single contact point, whereas this cylinder
# is a primitive and collides analytically: drift over 20 s is 0.00 mm / 0.00 deg at every mass
# from 15 g to 350 g. Set PICK_MASS = 0.35 for a full can if a heavier payload is ever wanted.
PICK_MASS = 0.015

# The pick object is a built primitive, not a robocasa asset. There is no drinks can in the
# 69-object lightwheel set, and every asset near the right size is the wrong shape or proportions:
#   Juice006      49 x 114  a bottle: uniform to 58% of height, then a shoulder and a 35 mm cap,
#                           so any grasp above mid-body closes on the neck
#   Paprika012    66 x 150  a true cylinder, but at a realistic 0.39 kg it is top-heavy; the
#                           usable grasp band shrank to the top 15% and below that it tips
#   Cinnamon007   48 x 125  the cleanest cylinder of all, and it tips Panda at every height
#   GlassCup025   48 x 114  tips the reBot
# A real 330 ml can is 66 x 115 mm, aspect 1.74, squatter than anything available, which is why it
# stands up to being grasped. Two further wins over a mesh: a cylinder primitive collides
# analytically instead of through a convex hull (no multiccd contact-count games), and its friction
# is ours to set, where every robocasa asset ships torsional friction 0.005 and a post-compile
# write to change that is ignored.
#
# PICK_SKIN is a mesh used purely for the can's appearance; collision and mass come from the
# cylinder in PICK_PRIMITIVE. Scaled onto that cylinder at generation time, so swapping it changes
# only how the object looks and never how it behaves. None gives the bare primitive.
PICK_SKIN = None

PICK_PRIMITIVE = {
    "radius": 0.033,          # 66 mm across
    "half_height": 0.0575,    # 115 mm tall
    "rgba": (0.72, 0.11, 0.13, 1.0),
    "friction": (1.0, 0.02, 0.002),   # torsional 4x the robocasa default: a can should not spin
    # Contact time constant. MuJoCo's default 0.02 is deliberately soft for solver stability, and on
    # a grasped object that softness is visible: the pads sank 7.8 mm into a 66 mm can, which looks
    # like the gripper passing through it. 0.005 with a realistic gripper force brings that to 2.2 mm,
    # 3% of the diameter, and the lift is unaffected (76 mm either way).
    "solref": (0.005, 1.0),
}

ROBOTS = {
    "rebot": {
        "mjcf": ROOT / "robots" / "seeed_rebot_devarm" / "seeed_rebot_devarm.xml",
        "arm_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        "gripper_joints": ("gripper_joint1", "gripper_joint2"),
        "eef_body": "gripper_end",
        # Measured, not chosen: scripts/measure_tcp.py, asserted by check_parity.py. The offset from
        # `gripper_end` to where the two pads meet when closed. Two wrong values preceded this one,
        # both of which produced grasps that worked often enough to look like a tuning problem.
        # (0, 0, 0.10), copied from Panda's convention, put the commanded point 113 mm out in mid-air
        # beside the gripper; the midpoint of the finger collision AABBs put it 39 mm off, because
        # these finger meshes are asymmetric. The 39 mm version closed the gripper beside the can and
        # flicked it away, which reads as the gripper passing through the object. The acceptance test
        # is symmetric pad clearance.
        "eef_offset": (-0.0109, 0.0000, 0.0050),
        "wrist_cam_pos": (-0.1009, 0.0, 0.0050),   # 90 mm back along -x, looking FRONT (+x)
        "wrist_cam_xyaxes": (0, 1, 0, 0, 0, -1),
        "base_sep": HANDOVER_SEP,
        "base_x": -0.05,
        # Per-side, searched (scripts/findhome.py <robot> <side>), not mirrored from one pose: the
        # correct mirror is robot-specific and copying left->right aimed the arm 0.43 m away.
        "home": {
            "left": (0.111, 0.966, 0.322, 0.543, 0.239, -2.922),
            "right": (0.299, 1.331, 0.463, 0.664, 0.961, -2.716),
        },
        # (closed, open) joint value, shared by both fingers. One pair rather than one value per
        # joint: Panda commands only finger_joint1 (the other follows an <equality>), so a per-joint
        # tuple silently mismatched the joint count.
        "grip_range": (0.0, 0.05),
        "grip_force": 15.0,   # N, small servo gripper. reBot fingers are slide joints, range [0, 0.05]
        # menagerie names each actuator after its joint, so the ros2_control parity check can compare
        # the two sets directly.
        "actuators_match_joints": True,
        # ros2_control / URDF side (source embodiment: needs teleop)
        "urdf": SEEED_URDF,
        "urdf_root": "base_link",
        "urdf_eef_frame": "{side}_gripper_end",
        "urdf_skip_links": (),
        "mesh_src": SEEED_MESHES,
        "mesh_glob": ("*.STL",),
        "mesh_remap": None,
        # One command interface per actuator. The reBot has two real gripper actuators plus an
        # <equality> coupling the fingers, so both are declared; Panda has one actuator driving both
        # fingers, so it declares one. Copying either pattern blindly is wrong.
        "ros2_control_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                                "gripper_joint1", "gripper_joint2"),
        "arm_ctrl_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        "grip_ctrl_joints": ("gripper_joint1", "gripper_joint2"),
    },
    # UR5e. 6-DoF like the reBot (the Panda's 7 make the IK non-square), 964 mm reach, and it wears
    # the reBot's own gripper via `graft_gripper`, so the wrist camera, which is mounted on the
    # gripper, sees a geometrically identical scene on both robots. That is what lets a policy
    # trained on the reBot drive this arm without re-recording a single episode.
    "ur5e": {
        "mjcf": Path("/home/sid/.cache/robot_descriptions/mujoco_menagerie/"
                     "universal_robots_ur5e/ur5e.xml"),
        "arm_joints": ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        "gripper_joints": ("rg_gripper_joint1", "rg_gripper_joint2"),
        "arm_ctrl_joints": ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        "grip_ctrl_joints": ("rg_gripper_joint1", "rg_gripper_joint2"),
        "ros2_control_joints": ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                                "rg_gripper_joint1", "rg_gripper_joint2"),
        "actuators_match_joints": False,
        "graft_gripper": {
            "mjcf": PKG.parent / "robots" / "shared_gripper" / "rebot_gripper.xml",
            "host_body": "wrist_3_link", "cut_body": None,
            # 110 mm of standoff past the flange, mirroring the reBot's own link6 -> gripper_end
            # offset. Without it the wrist camera (mounted 100 mm behind the gripper) sits inside the
            # UR5e's wrist and the view is 66% black. At 110 mm the near-black fraction is 51.5%
            # against the reBot's 51.4%: the arm is out of frame and what remains is the gripper.
            "pos": (0.0, 0.21, 0.0),
            "quat": (0.5, -0.5, -0.5, 0.5),
        },
        "eef_body": "rg_gripper_end",
        "eef_offset": (-0.0109, 0.0000, 0.0050),      # the reBot's: same gripper
        "grip_range": (0.0, 0.05),
        "grip_force": 15.0,
        "grip_ctrl_n": 2,
        "wrist_cam_pos": (-0.1009, 0.0, 0.0050),      # identical to the reBot, in the same frame
        "wrist_cam_xyaxes": (0, 1, 0, 0, 0, -1),
        "base_x": -0.05,
        "base_sep": 0.9,
        # Solved with damped least squares against PICK_POS / PLACE_POS with a downward approach;
        # both converge to 0.0 mm.
        "home": {"left": (0.8084, -4.5682, 1.9402, -0.5136, -0.7324, -1.5708),
                 "right": (1.0092, 3.4062, -2.2785, 2.0139, -0.1738, -1.5708)},
        "urdf": None, "urdf_root": "base", "urdf_eef_frame": "{side}_rg_gripper_end",
        "urdf_skip_links": (), "mesh_src": None, "mesh_glob": "*.obj", "mesh_remap": {},
    },

    "panda": {
        # The ros2-ready copy, not menagerie's panda.xml. menagerie drives the gripper through a
        # tendon actuator (`tendon="split"`, ctrlrange remapped to 0-255), which ros2_control cannot
        # command by joint name; check_parity caught it as "finger_joint1 has no actuator". This
        # version replaces it with a direct joint actuator on finger_joint1 and drops the tendon
        # (ntendon 1 -> 0), ctrlrange 0-0.4.
        "mjcf": ROOT / "robots" / "franka_emika_panda" / "panda_ros2.xml",
        "arm_joints": tuple(f"joint{i}" for i in range(1, 8)),
        "gripper_joints": ("finger_joint1", "finger_joint2"),
        "eef_body": "hand",
        # Measured with scripts/measure_tcp.py. The guessed 0.10 was 17.9 mm long, not enough to
        # stop Panda's 80 mm fingers closing on things, but a constant bias in every recorded grasp
        # pose. The two embodiments must agree on what "the tool point" means or the transfer number
        # measures the discrepancy instead of the policy.
        "eef_offset": (0.0049, 0.0000, 0.0602),
        "wrist_cam_pos": (-0.0851, 0.0, 0.0602),   # 90 mm back along -x, looking FRONT (+x)
        "wrist_cam_xyaxes": (0, 1, 0, 0, 0, -1),
        # Panda reaches 1.19 m against the reBot's 0.91 and its base is bulkier, so it sits wider
        # apart. Not yet put through the sensitivity sweep that set the reBot's numbers; run the
        # Panda leg of scripts/reach_gate.py before trusting these.
        "base_sep": HANDOVER_SEP,
        # Panda sits further back than the reBot: its links are much bulkier and at base_x=-0.05 the
        # two arms crowded the work area. Per-robot on purpose. What must stay identical across
        # embodiments is the task (object and plate poses), not where the arms are bolted.
        "base_x": -0.30,
        # Searched, not menagerie keyframe 0: that pose aimed the gripper off to the side and the
        # wrist cameras saw only table legs. scripts/findhome.py panda: aim 0.989, down 0.964.
        "home": {
            "left": (-0.550, 0.490, 0.624, -1.800, -0.358, 2.130, 1.018),
            "right": (-0.512, 0.501, 0.577, -1.810, -0.350, 2.163, 1.014),
        },
        "grip_range": (0.0, 0.04),
        "grip_force": 40.0,   # N. Panda finger_joint1 range [0, 0.04]
        # Panda actuators are `actuator1..8`, not joint names, and there are 8 for 9 joints (one
        # actuator drives both fingers, with an <equality> coupling them, the same topology as the
        # reBot's gripper). Any name-based joint/actuator check must skip this robot.
        "actuators_match_joints": False,
        # URDF from siddarth09/Panda_mujoco, already proven with mujoco_ros2_control. MJCF-derived,
        # so it agrees with menagerie exactly: FK checked over 400 configs at 0.0000 mm, unlike the
        # reBot, whose independently-derived URDF had every joint axis inverted.
        "urdf": ROOT / "robots" / "franka_emika_panda" / "panda_ros2.urdf",
        # `link0` is Panda's real base. Its URDF also ships its own `world` link, which must be
        # dropped: our generated file supplies a single shared `world` root for both arms, and two
        # copies of `world` would make the tree a forest.
        "urdf_root": "link0",
        "urdf_eef_frame": "{side}_hand",
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
    """Prefixed joint names for the <ros2_control> block: one per actuator, both arms."""
    return [prefixed(s, j) for s in SIDES for j in ROBOTS[key]["ros2_control_joints"]]


def start_override_path(key: str) -> Path:
    """Where the launch writes the start-position override for `key`.

    mujoco_ros2_control reads the can's start pose from a file named by the
    `override_start_position_file` hardware parameter, which lives in the URDF, so the path has
    to be fixed at generation time while its contents are written per launch from
    `can_x`/`can_y`/`can_yaw`. Under /tmp because the launch writes it on every run and an
    install tree should stay read-only.

    Why a file and not a service: the installed mujoco_ros2_control exposes only reset_world,
    set_pause and step_simulation. `SetFreeJointState` exists in the upstream source and would let
    the can be moved while running (which is what per-episode randomisation would need), but it is
    not in this build.
    """
    return Path(f"/tmp/zero_{key}_start.xml")


def ft_sensors(key: str) -> list[tuple[str, str]]:
    """(ros2_control sensor name, MJCF body it measures) for every fingertip force/torque pair.

    One per finger, not one per gripper: the useful signal during a grasp is the wrench through
    each finger's own joint, which tells you whether both pads are loaded or the object is
    sitting against one of them. That asymmetry is the failure the TCP bug produced, and a
    per-finger sensor would have shown it at once: +9.5 mm of clearance on one pad, +99 mm on
    the other.

    The naming is a contract with mujoco_ros2_control. It resolves a `<sensor mujoco_type="fts">`
    named X to MJCF sensors `X_force` and `X_torque` (suffixes are hardware params, defaults
    `_force`/`_torque`). Generate both ends from here so they cannot drift.
    """
    out = []
    for side in SIDES:
        for j in ROBOTS[key]["gripper_joints"]:
            out.append((f"{prefixed(side, j)}_ft", prefixed(side, j)))
    return out


def robot_mounts(key: str) -> dict[str, tuple[float, float, float]]:
    r = ROBOTS[key]
    return {
        "left": (r["base_x"], +r["base_sep"] / 2, TABLE_TOP_Z),
        "right": (r["base_x"], -r["base_sep"] / 2, TABLE_TOP_Z),
    }


# ======================================================================================
# THE TASK, identical for every embodiment
# ======================================================================================
# Pick a drinks can off the table, hand it across, place it in the tray. The can is built
# (PICK_PRIMITIVE) rather than taken from an asset; the tray is a robocasa mesh. Sizes are
# measured from mesh vertices, never geom_rbound, which reports a bounding-sphere radius and
# made the flat tray read as 460 mm cubed.
#     can    66 x 66 x 115 mm, 15 g   -> the pick object, a cylinder we define
#     tray   28 x 220 x 396 mm        -> the place target