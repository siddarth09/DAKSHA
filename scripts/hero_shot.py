"""Render a presentable view of a workstation, for the README.

    python scripts/hero_shot.py rebot
    python scripts/hero_shot.py ur5e

Uses the `home` keyframe after a short settle, so the image shows the state the sim boots into
rather than a hand-posed arrangement that no longer matches the model.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from PIL import Image

import zero_layout as L

W, H = 1100, 620


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    m = mujoco.MjModel.from_xml_path(str(L.PKG / "mjcf" / f"zero_{key}.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    # Settle so the can rests and the arms hold their commanded pose.
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = [0.30, 0.0, 0.88]
    cam.distance = 2.05
    cam.azimuth = 35.0
    cam.elevation = -18.0

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)          # default groups: visual shown, collision hidden

    out = L.ROOT / "docs" / f"{key}_scene.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(m, height=H, width=W) as ren:
        ren.update_scene(d, camera=cam, scene_option=opt)
        Image.fromarray(np.array(ren.render())).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
