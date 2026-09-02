"""Open any embodiment's scene in the interactive MuJoCo viewer.

    python scripts/view_scene.py rebot
    python scripts/view_scene.py panda

`python -m mujoco.viewer --mjcf=...` would also load the file, but it starts at qpos=0, so the
arms appear bolt upright and the can sits inside the table. This resets to the `home` keyframe
first, which is the state mujoco_ros2_control boots into.

Passive viewer, stepping in real time: physics runs, so a grasp can be checked by dragging the
can with the mouse. Press Tab for the control panel, and pick a camera from the drop-down to see
what the policy sees.
"""

from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer

import zero_layout as L


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    path = L.PKG / "mjcf" / f"zero_{key}.xml"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: MUJOCO_GL=egl python3 scripts/gen_scene.py {key}")

    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))

    print(f"[{key}] nq={m.nq} nu={m.nu} cameras={L.all_cameras()}")
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            mujoco.mj_step(m, d)
            v.sync()
            dt = m.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
