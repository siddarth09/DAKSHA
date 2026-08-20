"""Assert the URDF, the MJCF and the controller config agree on joint names.

RUN THIS AFTER ANY CHANGE TO EITHER DESCRIPTION. mujoco_ros2_control resolves joints by name
across `/robot_description` (URDF) and `/mujoco_robot_description` (MJCF). A mismatch does not
reliably raise -- the joints that do match keep working, so the usual symptom is "the arm moves
but the gripper does nothing", which is easy to misread as a controller or tuning problem and
expensive to chase. The failure this catches is real and already happened once: menagerie names
the gripper joints `joint_left`/`joint_right` while Seeed's official URDF calls them
`gripper_joint1`/`gripper_joint2`.

Exit code 0 = consistent, 1 = drift (so it can gate a build).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import mujoco
import yaml

import zero_layout as L


KEY = sys.argv[1] if len(sys.argv) > 1 else "rebot"
URDF = L.PKG / "urdf" / f"zero_{KEY}.urdf"
MJCF = L.PKG / "mjcf" / f"zero_{KEY}.xml"


def urdf_joints() -> set[str]:
    root = ET.parse(URDF).getroot()
    return {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}


def urdf_ros2_control_joints() -> set[str]:
    root = ET.parse(URDF).getroot()
    return {j.get("name") for rc in root.findall("ros2_control") for j in rc.findall("joint")}


def mjcf_joints() -> set[str]:
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}


def mjcf_actuated_joints() -> set[str]:
    """Joints that some actuator actually drives -- NOT actuator names.

    Comparing actuator names only works by luck: menagerie names the reBot's actuators after
    their joints, but Panda's are `actuator1..8`. What ros2_control actually needs is that every
    commanded joint HAS an actuator behind it, so resolve each actuator's transmission target
    instead. `actuator_trnid[:, 0]` is the joint id for trntype JOINT.
    """
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    out: set[str] = set()
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            out.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]))
    return out


def controller_joints() -> set[str]:
    path = L.PKG.parent / "zero_bringup" / "config" / f"{KEY}_controllers.yaml"
    if not path.exists():
        return set()
    cfg = yaml.safe_load(path.read_text())
    out: set[str] = set()
    for key, val in cfg.items():
        if isinstance(val, dict) and "ros__parameters" in val:
            out |= set(val["ros__parameters"].get("joints", []) or [])
    return out


def mjcf_cameras() -> set[str]:
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)}


def urdf_camera_sensors() -> set[str]:
    root = ET.parse(URDF).getroot()
    return {x.get("name") for rc in root.findall("ros2_control") for x in rc.findall("sensor")}


def check_kinematics(n_samples: int = 500, tol_mm: float = 0.5) -> bool:
    """Same joint angles in the URDF and the MJCF must give the same end-effector pose.

    NAME PARITY IS NOT ENOUGH, and this is not hypothetical. menagerie inverted the rotation
    axis on all six arm joints relative to Seeed's official URDF. Names matched, geometry
    matched, the zero pose matched -- and forward kinematics disagreed by 766 mm on average
    (1448 mm worst case) because every joint turned the wrong way. mujoco_ros2_control binds the
    two descriptions by name, so it would have commanded joints happily while the sim mirrored
    every motion. `scripts/fix_mjcf_conventions.py` corrects it; this asserts it stays corrected.

    Compiles the single-arm URDF in MuJoCo (MuJoCo reads URDF natively) and compares against the
    single-arm MJCF, rather than the bimanual pair -- the prefixing is already covered above and
    a single arm isolates the convention question.
    """
    import numpy as np

    if KEY != "rebot":
        print(f"  --  kinematic check is reBot-specific (Panda's URDF is MJCF-derived and was "
              f"verified at 0.0000 mm); skipped for {KEY}")
        return True
    if not L.SEEED_URDF.exists():
        print("  --  Seeed URDF not on disk, kinematic check skipped")
        return True

    txt = L.SEEED_URDF.read_text().replace(
        "package://rebotarm_bringup/description/meshes_rs/", "")
    txt = txt.replace(
        "<robot",
        f'<robot><mujoco><compiler meshdir="{L.SEEED_MESHES}" balanceinertia="true" '
        'discardvisual="false" fusestatic="false"/></mujoco>', 1)
    mu = mujoco.MjModel.from_xml_string(txt)
    mm = mujoco.MjModel.from_xml_path(str(L.MENAGERIE_ARM))
    du, dm = mujoco.MjData(mu), mujoco.MjData(mm)

    def _id(m, obj, n):
        i = mujoco.mj_name2id(m, obj, n)
        assert i >= 0, f"{n} missing"  # -1 would silently index the last element
        return i

    bu = _id(mu, mujoco.mjtObj.mjOBJ_BODY, "gripper_end")
    bm = _id(mm, mujoco.mjtObj.mjOBJ_BODY, "gripper_end")
    rng = np.random.default_rng(0)
    lo, hi = mm.jnt_range[:6, 0], mm.jnt_range[:6, 1]
    worst = 0.0
    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        for k, j in enumerate(L.ARM_JOINTS):
            du.qpos[mu.jnt_qposadr[_id(mu, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q[k]
            dm.qpos[mm.jnt_qposadr[_id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q[k]
        mujoco.mj_forward(mu, du)
        mujoco.mj_forward(mm, dm)
        worst = max(worst, float(np.linalg.norm(du.xpos[bu] - dm.xpos[bm])))
    ok = worst * 1000 <= tol_mm
    print(f"{'OK  ' if ok else 'FAIL'}  {'URDF vs MJCF kinematics':28} "
          f"max {worst * 1000:.3f} mm over {n_samples} configs (tol {tol_mm})")
    if not ok:
        print("        run scripts/fix_mjcf_conventions.py")
    return ok


def report_superset(name: str, got: set[str], want: set[str]) -> bool:
    """MJCF may legitimately hold MORE joints than ros2_control commands.

    Panda's second finger joint is driven by an <equality> constraint, not by its own actuator,
    so it exists in the MJCF but must not be declared as a command interface. Assert containment,
    not equality.
    """
    missing = want - got
    ok = not missing
    print(f"{'OK  ' if ok else 'FAIL'}  {name:28} {len(got):>3} present, "
          f"{len(want)} required")
    if missing:
        print(f"        missing: {sorted(missing)}")
    return ok


def report(name: str, got: set[str], want: set[str]) -> bool:
    missing, extra = want - got, got - want
    ok = not missing and not extra
    print(f"{'OK  ' if ok else 'FAIL'}  {name:28} {len(got):>3} items")
    if missing:
        print(f"        missing: {sorted(missing)}")
    if extra:
        print(f"        unexpected: {sorted(extra)}")
    return ok


def main() -> int:
    want = set(L.robot_prefixed_ros2_joints(KEY))
    print(f"expected {len(want)} joints from zero_layout "
          f"for {KEY} ({len(L.SIDES)} sides x "
          f"{len(L.ROBOTS[KEY]['ros2_control_joints'])})\n")

    checks = [
        # Superset: the URDF may hold MORE movable joints than are commanded. Panda's
        # `finger_joint2` is driven by a mimic/equality, not its own command interface.
        report_superset("URDF movable joints", urdf_joints(), want),
        report("URDF <ros2_control> joints", urdf_ros2_control_joints(), want),
        report_superset("MJCF joints", mjcf_joints(), want),
        report_superset("MJCF actuated joints", mjcf_actuated_joints(), want),
    ]
    # Cameras are the same by-name binding as the joints, with the same silent failure.
    checks.append(report("MJCF cameras", mjcf_cameras(), set(L.all_cameras())))
    checks.append(report("URDF camera <sensor>s", urdf_camera_sensors(), set(L.all_cameras())))
    checks.append(check_kinematics())
    ctrl = controller_joints()
    if ctrl:
        # Controllers legitimately cover a SUBSET (arm controller + gripper controller split),
        # so this asserts containment rather than equality.
        stray = ctrl - want
        ok = not stray
        print(f"{'OK  ' if ok else 'FAIL'}  {'controller yaml joints':28} {len(ctrl):>3} joints"
              f" (subset check)")
        if stray:
            print(f"        not in either description: {sorted(stray)}")
        checks.append(ok)
    else:
        print("  --  controller yaml not present yet, skipped")

    print()
    if all(checks):
        print("all descriptions agree")
        return 0
    print("DRIFT DETECTED -- fix before launching mujoco_ros2_control")
    return 1


if __name__ == "__main__":
    sys.exit(main())
