"""Extract the reBot gripper as a standalone MJCF, for grafting onto another arm.

    python3 scripts/make_shared_gripper.py     ->  robots/shared_gripper/rebot_gripper.xml

WHY. Measured on the Panda, the trained policy reaches the can zero-shot (8.0 mm closest approach,
inside the 17 mm jaw clearance) but never commits to a grasp: the commanded gripper crosses its
0.5 threshold 15 times versus 2 on the reBot. The reach transfers; the grasp DECISION does not.
That decision is the one that depends on the close-up wrist view, and the wrist view is the one
place the two robots still differ visually -- reBot's black jaws versus Panda's white fingers.

Putting the SAME gripper on both robots removes that difference, which is UMI's premise
("hardware-agnostic across multiple robot platforms"). Doing it by moving the reBot's gripper onto
the Panda -- rather than putting a new gripper on both -- changes only the TARGET robot, so the 82
recorded episodes stay valid. Re-recording them would cost ~10 hours of teleoperation.

There is no MjSpec `detach`, so the subtree is lifted by XML surgery: the `gripper_end` body is
promoted to a worldbody child and only the assets, defaults, actuators, equality and contact
excludes it actually references are carried across.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

SRC = Path("/home/sid/projects25/src/ZERO/robots/seeed_rebot_devarm/seeed_rebot_devarm.xml")
OUT_DIR = Path("/home/sid/projects25/src/ZERO/robots/shared_gripper")
ROOT_BODY = "gripper_end"


def subtree_refs(body: ET.Element) -> tuple[set[str], set[str], set[str], set[str]]:
    """Mesh, material, default-class and body names referenced anywhere under `body`."""
    meshes, mats, classes, bodies = set(), set(), set(), set()
    for e in body.iter():
        if e.tag == "body" and e.get("name"):
            bodies.add(e.get("name"))
        for k, dst in (("mesh", meshes), ("material", mats), ("class", classes)):
            if e.get(k):
                dst.add(e.get(k))
    return meshes, mats, classes, bodies


def main() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()
    grip = next(b for b in root.iter("body") if b.get("name") == ROOT_BODY)
    meshes, mats, classes, bodies = subtree_refs(grip)
    print(f"gripper subtree: {len(bodies)} bodies, {len(meshes)} meshes, "
          f"{len(mats)} materials, classes {sorted(classes)}")

    out = ET.Element("mujoco", {"model": "rebot_gripper"})
    comp = root.find("compiler")
    meshdir = (SRC.parent / (comp.get("meshdir") or "")).resolve() if comp is not None else SRC.parent
    if comp is not None:
        comp = ET.fromstring(ET.tostring(comp))
        # ⚠️ ABSOLUTE MESH PATHS. On attach the PARENT spec's meshdir wins, so a relative path here
        # is resolved against the host arm's asset directory and the compile dies with
        # "Error opening file .../franka_emika_panda/assets/cnc7.STL".
        comp.attrib.pop("meshdir", None)
        out.append(comp)
    d = root.find("default")
    if d is not None:
        out.append(d)                             # defaults are small; carry them wholesale

    asset = ET.SubElement(out, "asset")
    kept = 0
    for a in root.find("asset"):
        n = a.get("name") or Path(a.get("file", "")).stem
        if n in meshes or n in mats or a.tag == "texture":
            a = ET.fromstring(ET.tostring(a))
            if a.get("file"):
                a.set("file", str((meshdir / a.get("file")).resolve()))
            asset.append(a)
            kept += 1
    print(f"carried {kept} assets")

    wb = ET.SubElement(out, "worldbody")
    grip = ET.fromstring(ET.tostring(grip))       # copy; drop the pose it had on link6
    for k in ("pos", "quat", "euler", "axisangle"):
        grip.attrib.pop(k, None)
    wb.append(grip)

    for tag in ("contact", "equality", "actuator"):
        src = root.find(tag)
        if src is None:
            continue
        dst = ET.SubElement(out, tag)
        for e in src:
            names = {v for k, v in e.attrib.items()
                     if k.startswith(("body", "joint", "geom", "site"))}
            # keep only entries entirely inside the subtree
            if names and all(n in bodies or n.startswith("gripper_") for n in names):
                dst.append(e)
        print(f"  <{tag}>: kept {len(list(dst))}")
        if not len(list(dst)):
            out.remove(dst)

    # ⚠️ NAMESPACE THE ASSETS. Both source models define a material called `black`, and MjSpec
    # attach with an empty prefix then fails with "repeated name 'black' in material". Bodies and
    # joints keep their names (the scene generator prefixes those per side); only assets and the
    # default classes they reference are renamed.
    ren = {}
    for a in asset:
        n = a.get("name") or Path(a.get("file", "")).stem
        if n:
            ren[n] = f"rg_{n}"
            a.set("name", ren[n])
    for e in out.iter():
        for k in ("mesh", "material"):
            if e.get(k) in ren:
                e.set(k, ren[e.get(k)])

    # Same collision for default CLASS names ("repeated default class name"): both models define
    # `collision` and `visual`. Rename the classes and every `class`/`childclass` that points at one.
    cls = {}
    dflt = out.find("default")
    if dflt is not None:
        for e in dflt.iter("default"):
            if e.get("class"):
                cls[e.get("class")] = f"rg_{e.get('class')}"
                e.set("class", cls[e.get("class")])
    for e in out.iter():
        for k in ("class", "childclass"):
            if e.get(k) in cls:
                e.set(k, cls[e.get(k)])
    print(f"namespaced {len(ren)} assets and {len(cls)} default classes with rg_")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in ("assets", "meshes"):
        s = SRC.parent / d
        if s.is_dir() and not (OUT_DIR / d).exists():
            (OUT_DIR / d).symlink_to(s)           # meshdir is relative; symlink instead of copying
    dst = OUT_DIR / "rebot_gripper.xml"
    ET.indent(out, "  ")
    dst.write_bytes(ET.tostring(out))

    m = mujoco.MjModel.from_xml_path(str(dst))
    print(f"\ncompiles: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.neq} equality")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
