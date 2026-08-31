"""Generate a dual-arm URDF (plus ros2_control block) for any registered embodiment.

    python scripts/gen_urdf.py rebot
    python scripts/gen_urdf.py panda

Neither source URDF has a xacro macro or prefix support, so two copies collide on every link and
joint name. Generating means no hand-maintained duplication, it is re-runnable when upstream
changes, and the mount poses, joint names and camera names all come from `zero_layout`, the same
module the MJCF generator uses, so the two descriptions cannot drift.
`scripts/check_parity.py <robot>` asserts they have not.

Out:  zero_description/urdf/zero_<robot>.urdf
"""

from __future__ import annotations

import copy
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import zero_layout as L


def _remap_part(part: str) -> str:
    """Converter mesh name -> the menagerie asset that actually exists.

    Three cases, all discovered by checking which references failed to resolve rather than by
    assuming a scheme:
      link0_0   -> link0_0.obj            visual, keeps its index
      link0_c   -> link0.stl              collision; menagerie ships these as .stl
      link5_c0  -> link5_collision_0.obj  link5's collision is split into 3 convex pieces
    """
    m = re.fullmatch(r"(link\d+)_c(\d+)", part)
    if m:
        return f"{m.group(1)}_collision_{m.group(2)}.obj"
    if part.endswith("_c"):
        return f"{part[:-2]}.stl"
    return f"{part}.obj"


def prefix_side(arm: ET.Element, side: str, r: dict) -> list[ET.Element]:
    """Deep-copy links+joints, prefix every name AND every back-reference.

    A name can hide in four places: the link's `name`, the joint's `name`, and the joint's
    <parent link=> / <child link=>. Miss one and the URDF still parses but describes a
    disconnected tree, which `check_urdf` reports as multiple roots.
    """
    skip = set(r["urdf_skip_links"])
    remap = re.compile(r["mesh_remap"]) if r["mesh_remap"] else None
    out: list[ET.Element] = []

    for link in arm.findall("link"):
        if link.get("name") in skip:
            continue
        el = copy.deepcopy(link)
        el.set("name", L.prefixed(side, link.get("name")))
        for tag in ("visual", "collision"):
            for g in el.findall(f"{tag}/geometry/mesh"):
                fn = g.get("filename").rsplit("/", 1)[-1]
                if remap:
                    m = remap.fullmatch(fn)
                    if m:
                        part = m.group(1)
                        # A trailing `_c` marks a collision mesh in the converter's naming, and menagerie ships those
                        # as .stl (`link0.stl`), not `_c.obj`. Visual meshes keep their index and stay .obj
                        # (`link0_0.obj`).
                        fn = _remap_part(part)
                g.set("filename", f"package://zero_description/meshes/{fn}")
        out.append(el)

    for joint in arm.findall("joint"):
        p = joint.find("parent")
        c = joint.find("child")
        # A joint touching a skipped link (Panda's world->link0 weld) is dropped; our own mount joint
        # replaces it.
        if (p is not None and p.get("link") in skip) or (c is not None and c.get("link") in skip):
            continue
        el = copy.deepcopy(joint)
        el.set("name", L.prefixed(side, joint.get("name")))
        for tag in ("parent", "child"):
            ref = el.find(tag)
            if ref is not None and ref.get("link"):
                ref.set("link", L.prefixed(side, ref.get("link")))
        mim = el.find("mimic")
        if mim is not None and mim.get("joint"):
            mim.set("joint", L.prefixed(side, mim.get("joint")))
        out.append(el)
    return out


def add_ros2_control(robot: ET.Element, key: str) -> None:
    r = L.ROBOTS[key]
    rc = ET.SubElement(robot, "ros2_control",
                       {"name": f"Zero{key.capitalize()}System", "type": "system"})
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "mujoco_ros2_control/MujocoSystemInterface"

    # `$(find ...)`, not `package://`. The plugin does not resolve package://; it hands the string
    # straight to the MuJoCo loader and dies with "MuJoCo model file 'package://...' does not exist!".
    # xacro expands $(find pkg) into an absolute path before the plugin sees it, so the launch must
    # pipe this URDF through xacro. That is how Panda_mujoco does it, and it beats baking in
    # /home/sid/... which breaks the moment the project moves.
    ET.SubElement(hw, "param", {"name": "mujoco_model"}).text = (
        f"$(find zero_description)/mjcf/zero_{key}.xml")
    # Faster-than-realtime stepping, for data collection and eval sweeps.
    ET.SubElement(hw, "param", {"name": "sim_speed_factor"}).text = "1.0"
    ET.SubElement(hw, "param", {"name": "camera_publish_rate"}).text = f"{L.CAM_RATE_HZ}"
    # Start-position override: the launch writes this file from its can_x/can_y/can_yaw arguments
    # before the sim starts. The path is fixed here because hardware params cannot be set from the
    # command line; only the file's contents vary per run.
    ET.SubElement(hw, "param", {"name": "override_start_position_file"}).text = str(
        L.start_override_path(key))
    # Without this the sim starts at qpos=0 and the arms bolt upright. gen_scene.py writes a `home`
    # keyframe holding the searched per-side poses plus the object's placed pose.
    # (mujoco_system_interface.cpp:1353 reads this param.)
    ET.SubElement(hw, "param", {"name": "initial_keyframe"}).text = "home"

    for name in L.robot_prefixed_ros2_joints(key):
        j = ET.SubElement(rc, "joint", {"name": name})
        ET.SubElement(j, "command_interface", {"name": "position"})
        for iface in ("position", "velocity"):
            ET.SubElement(j, "state_interface", {"name": iface})

    # Cameras publish only if a <sensor> of the same name as the MJCF <camera> appears here. No extra
    # plugin: image/info/depth publishing lives in the core system interface.
    #
    # Fingertip F/T sensors: mujoco_ros2_control maps a <sensor mujoco_type="fts"> named X onto MJCF
    # sensors X_force / X_torque, and exports exactly six state interfaces named force.x/y/z and
    # torque.x/y/z. That spelling is checked literally in the plugin, so a typo drops the axis rather
    # than erroring. force_torque_sensor_broadcaster reads those and republishes a WrenchStamped.
    for sensor_name, _ in L.ft_sensors(key):
        sen = ET.SubElement(rc, "sensor", {"name": sensor_name})
        ET.SubElement(sen, "param", {"name": "mujoco_type"}).text = "fts"
        for axis in ("force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z"):
            ET.SubElement(sen, "state_interface", {"name": axis})

    for cam in L.all_cameras():
        sen = ET.SubElement(rc, "sensor", {"name": cam})
        for k, v in (("frame_name", f"{cam}_optical_frame"),
                     ("image_topic", f"{L.CAM_NS}/{cam}/image_raw"),
                     ("info_topic", f"{L.CAM_NS}/{cam}/camera_info"),
                     ("depth_topic", f"{L.CAM_NS}/{cam}/depth")):
            ET.SubElement(sen, "param", {"name": k}).text = v


def build(key: str) -> ET.Element:
    r = L.ROBOTS[key]
    if not r["urdf"].exists():
        raise SystemExit(f"missing {r['urdf']}")
    arm = ET.parse(r["urdf"]).getroot()
    robot = ET.Element("robot", {"name": f"zero_{key}"})
    for mat in arm.findall("material"):
        robot.append(copy.deepcopy(mat))
    ET.SubElement(robot, "link", {"name": "world"})

    mounts = L.robot_mounts(key)
    for side in L.SIDES:
        x, y, z = mounts[side]
        m = ET.SubElement(robot, "joint", {"name": f"{side}_mount", "type": "fixed"})
        ET.SubElement(m, "origin", {"xyz": f"{x} {y} {z}", "rpy": "0 0 0"})
        ET.SubElement(m, "parent", {"link": "world"})
        ET.SubElement(m, "child", {"link": L.prefixed(side, r["urdf_root"])})
        for el in prefix_side(arm, side, r):
            robot.append(el)

    add_ros2_control(robot, key)
    return robot


def copy_meshes(key: str) -> int:
    r = L.ROBOTS[key]
    dst = L.PKG / "meshes"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in r["mesh_glob"]:
        for f in sorted(r["mesh_src"].glob(pattern)):
            shutil.copy2(f, dst / f.name)
            n += 1
    # .mtl siblings, else the .obj materials resolve to nothing in RViz
    for f in sorted(r["mesh_src"].glob("*.mtl")):
        shutil.copy2(f, dst / f.name)
    return n


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    if key not in L.ROBOTS:
        raise SystemExit(f"unknown robot {key!r}; known: {list(L.ROBOTS)}")
    robot = build(key)
    out = L.PKG / "urdf" / f"zero_{key}.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(out, encoding="utf-8", xml_declaration=True)

    n = copy_meshes(key)
    joints = [(j.get("name"), j.get("type")) for j in robot.findall("joint")]
    movable = [x for x, t in joints if t != "fixed"]
    rcj = [x.get("name") for rc in robot.findall("ros2_control") for x in rc.findall("joint")]
    sens = [x.get("name") for rc in robot.findall("ros2_control") for x in rc.findall("sensor")]
    print(f"[{key}] wrote {out}")
    print(f"  links {len(robot.findall('link'))}  joints {len(joints)}  movable {len(movable)}")
    print(f"  ros2_control joints ({len(rcj)}): {rcj}")
    print(f"  camera sensors ({len(sens)}): {sens}")
    print(f"  copied {n} meshes")


if __name__ == "__main__":
    main()
