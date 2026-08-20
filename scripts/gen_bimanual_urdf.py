"""Generate a two-arm URDF (with a ros2_control block for mujoco_ros2_control) from Seeed's
official single-arm RS URDF.

WHY GENERATE INSTEAD OF HAND-EDITING: Seeed's URDF is a 772-line SolidWorks export with no
xacro macro and no prefix support, so two copies collide on every link and joint name. Doing
it programmatically means (a) no 700 lines of hand-renamed duplication to maintain, (b) it is
re-runnable when Seeed updates the URDF, and (c) the mount poses and joint names come from
`zero_layout`, the same module the MJCF generator uses -- so the two descriptions cannot drift.

Run:  python scripts/gen_bimanual_urdf.py
Out:  zero_description/urdf/zero_bimanual.urdf
"""

from __future__ import annotations

import copy
import shutil
import xml.etree.ElementTree as ET

import zero_layout as L

# ⚠️ ABSOLUTE FILESYSTEM PATH, not a package:// URI. mujoco_ros2_control does NOT resolve
# package:// for this param -- it passes the string straight to the MuJoCo loader, and
# `package://zero_description/mjcf/zero_bimanual.xml` fails with
#   [FATAL] MuJoCo model file 'package://...' does not exist!
# Points at the SOURCE tree rather than the install share so it stays valid whether or not the
# workspace has been rebuilt. Regenerate this URDF if the project ever moves.
MJCF_PATH = str(L.OUT_MJCF)


def load_arm() -> ET.Element:
    if not L.SEEED_URDF.exists():
        raise SystemExit(
            f"missing {L.SEEED_URDF}\n"
            "clone it:  git clone --depth 1 "
            "https://github.com/Seeed-Projects/reBotArmController_ROS2.git ~/rebotarm_ros2"
        )
    return ET.parse(L.SEEED_URDF).getroot()


def prefix_side(arm: ET.Element, side: str) -> tuple[list[ET.Element], list[ET.Element]]:
    """Deep-copy every link/joint and rename it with the side prefix.

    Renames cover four places a name can hide: the link's own `name`, the joint's own `name`,
    and the joint's `<parent link=>` / `<child link=>` back-references. Missing any one of
    those produces a URDF that parses but describes a disconnected tree.

    Materials are deliberately NOT prefixed -- they are global in URDF, and duplicating them
    per side would just emit redefinition warnings.
    """
    links, joints = [], []
    for link in arm.findall("link"):
        el = copy.deepcopy(link)
        el.set("name", L.prefixed(side, link.get("name")))
        for tag in ("visual", "collision"):
            for g in el.findall(f"{tag}/geometry/mesh"):
                fn = g.get("filename").rsplit("/", 1)[-1]
                g.set("filename", f"package://zero_description/meshes/{fn}")
        links.append(el)

    for joint in arm.findall("joint"):
        el = copy.deepcopy(joint)
        el.set("name", L.prefixed(side, joint.get("name")))
        for tag in ("parent", "child"):
            ref = el.find(tag)
            if ref is not None and ref.get("link"):
                ref.set("link", L.prefixed(side, ref.get("link")))
        mimic = el.find("mimic")
        if mimic is not None and mimic.get("joint"):
            mimic.set("joint", L.prefixed(side, mimic.get("joint")))
        joints.append(el)
    return links, joints


def build() -> ET.Element:
    arm = load_arm()
    robot = ET.Element("robot", {"name": "zero_bimanual"})

    # Materials once, shared by both arms.
    for mat in arm.findall("material"):
        robot.append(copy.deepcopy(mat))

    # A world root. base_link of each arm hangs off it by a fixed joint at the mount pose,
    # which is what makes the two subtrees a single connected URDF (ros2_control and
    # robot_state_publisher both reject a forest).
    ET.SubElement(robot, "link", {"name": "world"})

    for side in L.SIDES:
        links, joints = prefix_side(arm, side)
        x, y, z = L.MOUNTS[side]
        mount = ET.SubElement(robot, "joint",
                              {"name": f"{side}_mount", "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": f"{x} {y} {z}", "rpy": "0 0 0"})
        ET.SubElement(mount, "parent", {"link": "world"})
        ET.SubElement(mount, "child", {"link": L.prefixed(side, "base_link")})
        for el in links + joints:
            robot.append(el)

    add_ros2_control(robot)
    return robot


def add_ros2_control(robot: ET.Element) -> None:
    """One <ros2_control> system covering both arms.

    `mujoco_model` is read by mujoco_ros2_control to locate the MJCF; the plugin then
    cross-checks every joint declared here against the MuJoCo model AND the URDF, which is
    exactly why zero_layout owns the name list.
    """
    rc = ET.SubElement(robot, "ros2_control", {"name": "ZeroBimanualSystem", "type": "system"})
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "mujoco_ros2_control/MujocoSystemInterface"
    p = ET.SubElement(hw, "param", {"name": "mujoco_model"})
    p.text = MJCF_PATH

    for name in L.all_prefixed_joints():
        j = ET.SubElement(rc, "joint", {"name": name})
        ET.SubElement(j, "command_interface", {"name": "position"})
        for iface in ("position", "velocity"):
            ET.SubElement(j, "state_interface", {"name": iface})

    # ── CAMERAS. Declaring a <camera> in the MJCF is NOT enough: mujoco_ros2_control only
    # publishes a camera if a <sensor> of the SAME NAME appears here. No extra plugin is needed
    # -- image/info/depth publishing is in the core system interface (the plugins package ships
    # only Heartbeat and ExternalWrench). Without these blocks the cameras exist in MuJoCo and
    # silently never reach ROS, which reads like a QoS or networking fault.
    for cam in L.all_cameras():
        sen = ET.SubElement(rc, "sensor", {"name": cam})
        for key, val in (
            ("frame_name", f"{cam}_optical_frame"),
            ("image_topic", f"{L.CAM_NS}/{cam}/image_raw"),
            ("info_topic", f"{L.CAM_NS}/{cam}/camera_info"),
            ("depth_topic", f"{L.CAM_NS}/{cam}/depth"),
        ):
            ET.SubElement(sen, "param", {"name": key}).text = val


def copy_meshes() -> int:
    dst = L.PKG / "meshes"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(L.SEEED_MESHES.glob("*.STL")):
        shutil.copy2(f, dst / f.name)
        n += 1
    return n


def main() -> None:
    robot = build()
    L.OUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(L.OUT_URDF, encoding="utf-8", xml_declaration=True)

    n_mesh = copy_meshes()
    links = [l.get("name") for l in robot.findall("link")]
    joints = [(j.get("name"), j.get("type")) for j in robot.findall("joint")]
    movable = [n for n, t in joints if t != "fixed"]
    sensors = [x.get("name") for rc in robot.findall("ros2_control") for x in rc.findall("sensor")]
    print(f"wrote {L.OUT_URDF}")
    print(f"  links {len(links)}  joints {len(joints)}  movable {len(movable)}")
    print(f"  movable: {movable}")
    print(f"  camera sensors ({len(sensors)}): {sensors}")
    print(f"  copied {n_mesh} meshes -> {L.PKG / 'meshes'}")


if __name__ == "__main__":
    main()
