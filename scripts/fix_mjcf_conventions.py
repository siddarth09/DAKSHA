"""Rewrite menagerie's reBot MJCF to match Seeed's OFFICIAL URDF conventions.

Regenerates robots/seeed_rebot_devarm/seeed_rebot_devarm.xml from the pristine
`.menagerie-orig` copy, applying two corrections. Re-runnable; safe to re-run.

Two divergences between menagerie and Seeed's URDF. Both are silent, and both break
mujoco_ros2_control, which binds the URDF and the MJCF together by joint name:

1. Gripper joint names. menagerie calls them `joint_left`/`joint_right`, the official URDF calls
   them `gripper_joint1`/`gripper_joint2`. Mismatched names are skipped, so the arm works and
   the gripper never moves.

2. Joint sign convention. menagerie inverted the rotation axis on all six arm joints. Compiling
   both models and comparing forward kinematics over 500 random configurations:
       zero pose:            identical, gripper_end (0.3017, 0, 0.2177) in both
       same joint angles:    766 mm mean / 1448 mm max disagreement
       URDF angles negated:  0.00 mm, exact agreement
   The geometry is identical and only the sign differs. Unfixed, every commanded joint rotates
   backwards in sim while RViz shows the mirror image.

The URDF wins because Seeed's Python SDK, their MoveIt config and the real RobStride motors all
use it, so it defines the hardware's positive direction.

Flipping an axis touches everything expressed in that joint's coordinates: the joint `range`
(negate and swap, since -3.14..0 becomes 0..3.14), the actuator `ctrlrange` likewise, and every
`qpos`/`ctrl` entry in every keyframe.

Run:  python scripts/fix_mjcf_conventions.py
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET

import zero_layout as L

ARM_DIR = L.MENAGERIE_ARM.parent
PRISTINE = ARM_DIR / "seeed_rebot_devarm.xml.menagerie-orig"

RENAME = {"joint_left": "gripper_joint1", "joint_right": "gripper_joint2"}
FLIP = set(L.ARM_JOINTS)  # joint1..joint6 only; the gripper slides are unaffected


def flip_pair(text: str) -> str:
    """Negate and swap a "lo hi" range so it stays ordered after an axis flip."""
    lo, hi = (float(v) for v in text.split())
    return f"{-hi:g} {-lo:g}"


def negate_vec(text: str) -> str:
    return " ".join(f"{-float(v):g}" for v in text.split())


def main() -> None:
    if not PRISTINE.exists():
        # First run: preserve the untouched menagerie file before we ever modify it.
        shutil.copy2(L.MENAGERIE_ARM, PRISTINE)
        print(f"saved pristine copy -> {PRISTINE.name}")

    root = ET.parse(PRISTINE).getroot()

    # 1. rename gripper joints (joint elements, and everything referencing them)
    # `joint1`/`joint2` are here because menagerie couples the two fingers with
    #   <equality><joint joint1="joint_left" joint2="joint_right" polycoef="0 1 0 0 0"/>
    # modelling the real 1:1 rack-and-pinion. Miss those two attributes and the model fails to
    # compile with "unknown element 'joint_left' in equality constraint". Matching is on the
    # attribute value, so there is no clash with the arm joints that happen to be named
    # joint1/joint2.
    for el in root.iter():
        for attr in ("name", "joint", "joint1", "joint2"):
            v = el.get(attr)
            if v in RENAME:
                el.set(attr, RENAME[v])

    # 2. flip the six arm joints
    flipped = []
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in FLIP:
            continue
        if (axis := joint.get("axis")) is not None:
            joint.set("axis", negate_vec(axis))
        if (rng := joint.get("range")) is not None:
            joint.set("range", flip_pair(rng))
        flipped.append(name)

    # actuator ctrlrange lives in the joint's coordinates too
    for act in root.iter():
        if act.tag in ("general", "position", "motor") and act.get("joint") in FLIP:
            if (cr := act.get("ctrlrange")) is not None:
                act.set("ctrlrange", flip_pair(cr))

    # keyframes: qpos/ctrl are ordered by joint/actuator index. Both lists are the 8 joints in
    # declaration order, so flip the first six entries of each.
    n_flip = len(L.ARM_JOINTS)
    for key in root.iter("key"):
        for attr in ("qpos", "ctrl"):
            if (v := key.get(attr)) is None:
                continue
            vals = v.split()
            vals[:n_flip] = [f"{-float(x):g}" for x in vals[:n_flip]]
            key.set(attr, " ".join(vals))

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(L.MENAGERIE_ARM, encoding="utf-8", xml_declaration=True)
    print(f"wrote {L.MENAGERIE_ARM}")
    print(f"  renamed: {RENAME}")
    print(f"  sign-flipped: {flipped}")


if __name__ == "__main__":
    main()
