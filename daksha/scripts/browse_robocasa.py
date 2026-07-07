#!/usr/bin/env python
"""Render contact sheets of RoboCasa layouts and styles so you can pick before committing.

Produces:
  scenes/browse_layouts.png  - all 60 layouts at a fixed style
  scenes/browse_styles.png   - all 60 styles at a fixed layout

Failed combos (missing asset packs) become a labeled red tile instead of crashing.

Usage:
    python browse_robocasa.py                 # both sheets, defaults
    python browse_robocasa.py --fixed-style 1 --fixed-layout 1
"""
import argparse, os, sys, contextlib, io
import numpy as np
import mujoco
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

from robocasa.models.scenes import KitchenArena
from robocasa.models.scenes.scene_registry import LayoutType, StyleType, get_layout_path
from robosuite.models.tasks import ManipulationTask
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.abspath(os.path.join(HERE, "..", "scenes"))
THUMB_W, THUMB_H = 640, 360


def _match(nm, names):
    return nm and any(nm == n or nm.startswith(n + "_") or nm.endswith("_" + n) for n in names)


def render_one(layout, style, w=THUMB_W, h=THUMB_H):
    """Build a robot-free kitchen and return a rendered RGB thumbnail (numpy)."""
    with contextlib.redirect_stdout(io.StringIO()):
        arena = KitchenArena(layout_id=layout, style_id=style)
        arena.set_origin([0, 0, 0])
        fixtures = [cfg["model"] for cfg in arena.get_fixture_cfgs()]
        task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[], mujoco_objects=fixtures)
        xml = task.get_xml()
        if "offwidth" not in xml:
            xml = xml.replace("<visual>", '<visual>\n<global offwidth="1920" offheight="1080"/>', 1)
        model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # hide enclosing walls
    walls = (yaml.safe_load(open(get_layout_path(layout))).get("room") or {}).get("walls") or []
    wall_names = [x["name"] for x in walls if x.get("enclosing_wall")]
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

    vis = [g for g in range(model.ngeom) if model.geom_rgba[g, 3] > 0 and model.geom_bodyid[g] != 0]
    pts = data.geom_xpos[vis]
    lo, hi = pts.min(0), pts.max(0)
    ctr = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))

    r = mujoco.Renderer(model, h, w)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0; opt.geomgroup[1] = 1; opt.geomgroup[2] = 1
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [ctr[0], ctr[1], 1.2]; cam.distance = diag * 0.95
    cam.azimuth = 55; cam.elevation = -25
    r.update_scene(data, cam, opt)
    img = r.render()
    r.close()
    return img


def contact_sheet(items, fixed_kind, fixed_val, out_name, cols=6):
    """items: list of (id, label). fixed_kind in {'style','layout'}."""
    n = len(items)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * THUMB_W, rows * THUMB_H), (30, 30, 34))
    draw = ImageDraw.Draw(sheet)
    ok, fail = 0, []
    for i, (idv, label) in enumerate(items):
        cx, cy = (i % cols) * THUMB_W, (i // cols) * THUMB_H
        try:
            if fixed_kind == "style":
                arr = render_one(idv, fixed_val)
            else:
                arr = render_one(fixed_val, idv)
            sheet.paste(Image.fromarray(arr), (cx, cy)); ok += 1
        except Exception as e:  # missing assets etc.
            draw.rectangle([cx, cy, cx + THUMB_W, cy + THUMB_H], fill=(90, 25, 25))
            draw.text((cx + 12, cy + THUMB_H // 2), f"FAILED\n{type(e).__name__}", fill=(255, 200, 200))
            fail.append(idv)
        # label banner
        draw.rectangle([cx, cy, cx + 150, cy + 34], fill=(0, 0, 0))
        draw.text((cx + 8, cy + 8), label, fill=(255, 255, 0))
        print(f"  [{i+1}/{n}] {label} {'ok' if idv not in fail else 'FAILED'}", flush=True)
    path = os.path.join(SCENES, out_name)
    sheet.save(path)
    print(f"saved {path}  ({ok} ok, {len(fail)} failed: {fail})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-style", type=int, default=1)
    ap.add_argument("--fixed-layout", type=int, default=1)
    ap.add_argument("--mode", choices=["layouts", "styles", "both"], default="both")
    args = ap.parse_args()

    layouts = [(m.value, f"L{m.value:02d}") for m in LayoutType if m.value >= 0]
    styles = [(m.value, f"S{m.value:02d}") for m in StyleType if m.value >= 0]

    if args.mode in ("layouts", "both"):
        print(f"== layouts (style fixed at {args.fixed_style}) ==")
        contact_sheet(layouts, "style", args.fixed_style, "browse_layouts.png")
    if args.mode in ("styles", "both"):
        print(f"== styles (layout fixed at {args.fixed_layout}) ==")
        contact_sheet(styles, "layout", args.fixed_layout, "browse_styles.png")


if __name__ == "__main__":
    main()
