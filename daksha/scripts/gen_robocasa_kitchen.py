#!/usr/bin/env python
"""Generate a real RoboCasa kitchen (NO robot) with random RoboCasa objects, and render it.

Assembles a RoboCasa KitchenArena + fixtures + N sampled RoboCasa objects into a robosuite
ManipulationTask with an empty robot list, places the objects on a clear counter region
(RoboCasa reset regions), settles, and renders overview + counter shots.

Only samples from the 'lightwheel' object registry (the pack that's downloaded).

Usage (inside bheema_rl_env):
    python gen_robocasa_kitchen.py --layout 2 --style 2 --objects 3
"""
import argparse, os, io, contextlib, math, time
import numpy as np
import mujoco
import mujoco.viewer
import PIL.Image as Image

from robocasa.models.scenes import KitchenArena
from robocasa.models.scenes.scene_registry import get_layout_path
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.models.objects.objects import MJCFObject
from robosuite.models.tasks import ManipulationTask
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.abspath(os.path.join(HERE, "..", "scenes"))
OBJ_REGISTRIES = ("lightwheel",)  


def _match(nm, names):
    return nm and any(nm == n or nm.startswith(n + "_") or nm.endswith("_" + n) for n in names)


def best_counter_region(arena):
    """(cx, cy, topz, hx, hy) of the largest CLEAR counter top strip (world frame)."""
    best = None
    for c in arena.get_fixture_cfgs():
        fx = c["model"]
        if "counter" not in c["name"].lower() or not hasattr(fx, "get_reset_regions"):
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rr = fx.get_reset_regions(env=None)
        except Exception:
            continue
        for v in rr.values():
            off, size = v["offset"], v["size"]
            cx, cy = fx.pos[0] + off[0], fx.pos[1] + off[1]
            topz = fx.pos[2] + off[2]
            hx, hy = size[0] / 2, size[1] / 2
            area = size[0] * size[1]
            if best is None or area > best[0]:
                best = (area, cx, cy, topz, hx, hy)
    return best[1:] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=int, default=2)
    ap.add_argument("--style", type=int, default=2)
    ap.add_argument("--objects", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="robocasa_kitchen")
    ap.add_argument("--view", action="store_true",
                    help="open the interactive MuJoCo viewer (collision hidden, walls transparent)")
    ap.add_argument("--no-render", action="store_true", help="skip saving PNG renders")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Building RoboCasa kitchen layout={args.layout} style={args.style} ...")
    with contextlib.redirect_stdout(io.StringIO()):
        # sample RoboCasa objects from the library
        objs, cats = [], []
        for i in range(args.objects):
            kw, info = sample_kitchen_object(
                groups="all", obj_registries=OBJ_REGISTRIES, graspable=True,
                rng=rng, max_size=(0.28, 0.28, 0.32))
            kw.pop("name", None)
            objs.append(MJCFObject(name=f"obj{i}", **kw))
            cats.append(info["cat"])

        arena = KitchenArena(layout_id=args.layout, style_id=args.style)
        arena.set_origin([0, 0, 0])
        fixtures = [cfg["model"] for cfg in arena.get_fixture_cfgs()]
        task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                                mujoco_objects=fixtures + objs)
        xml = task.get_xml()
    print(f"  fixtures: {len(fixtures)}")
    if args.objects:
        print(f"  sampled objects: {', '.join(cats)}")

    if "offwidth" not in xml:
        xml = xml.replace("<visual>", '<visual>\n    <global offwidth="1920" offheight="1080"/>', 1)

    os.makedirs(SCENES, exist_ok=True)
    xml_path = os.path.join(SCENES, f"{args.out}.xml")
    with open(xml_path, "w") as f:
        f.write(xml)
    print("saved scene ->", xml_path)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # ---- place objects on a clear counter region via their free joints ----
    region = best_counter_region(arena)
    if args.objects and region:
        cx, cy, topz, hx, hy = region
        print(f"  clear counter region: center=({cx:.2f},{cy:.2f}) topz={topz:.2f} half=({hx:.2f},{hy:.2f})")
        spacing = min(0.26, (2 * hx - 0.2) / max(args.objects, 1))
        for i in range(args.objects):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_joint0")
            if jid < 0:
                continue
            adr = model.jnt_qposadr[jid]
            px = cx + (i - (args.objects - 1) / 2) * spacing + float(rng.uniform(-0.03, 0.03))
            px = min(max(px, cx - hx + 0.1), cx + hx - 0.1)
            py = cy + float(rng.uniform(-min(hy - 0.1, 0.1), min(hy - 0.1, 0.1)))
            pz = topz + 0.15 + 0.03 * i
            yaw = float(rng.uniform(0, 2 * math.pi))
            data.qpos[adr:adr + 3] = [px, py, pz]
            data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    print(f"OK: nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh} ntex={model.ntex}")

    # settle objects onto the counter
    for _ in range(2000):
        mujoco.mj_step(model, data)

    # ---- hide enclosing walls ----
    walls = (yaml.safe_load(open(get_layout_path(args.layout))).get("room") or {}).get("walls") or []
    wall_names = [w["name"] for w in walls if w.get("enclosing_wall")]
    wb = set()
    for b in range(model.nbody):
        if _match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b), wall_names):
            wb.add(b)
    ch = True
    while ch:
        ch = False
        for b in range(1, model.nbody):
            if b not in wb and model.body_parentid[b] in wb:
                wb.add(b); ch = True
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in wb or _match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g), wall_names):
            model.geom_rgba[g, 3] = 0.0

    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    model.vis.headlight.specular[:] = 0.1

    vis = [g for g in range(model.ngeom) if model.geom_rgba[g, 3] > 0 and model.geom_bodyid[g] != 0]
    pts = data.geom_xpos[vis]
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))

    if not args.no_render:
        r = mujoco.Renderer(model, 1080, 1920)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0; opt.geomgroup[1] = 1; opt.geomgroup[2] = 1

        def shot(fn, lookat, dist, az, el):
            c = mujoco.MjvCamera()
            c.lookat[:] = lookat; c.distance = dist; c.azimuth = az; c.elevation = el
            r.update_scene(data, c, opt)
            Image.fromarray(r.render()).save(os.path.join(SCENES, fn))
            print("saved", fn)

        shot(f"{args.out}_overview.png", [center[0], center[1], 1.2], diag * 0.9, 55, -28)
        lookat = [region[0], region[1], region[2] + 0.12] if region else [center[0], center[1], 1.1]
        shot(f"{args.out}_counter.png", lookat, 1.4, 65, -20)

    if args.view:
        print("opening interactive viewer (collision hidden, walls transparent)... close window to exit")
        with mujoco.viewer.launch_passive(model, data) as v:
            v.opt.geomgroup[:] = 0
            v.opt.geomgroup[1] = 1
            v.opt.geomgroup[2] = 1
            v.sync()
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(0.002)


if __name__ == "__main__":
    main()
