"""Find a home pose whose gripper points AT the task volume, for any registered embodiment.

    python scripts/findhome.py rebot
    python scripts/findhome.py panda

WHY: both menagerie models ship a "nice looking" keyframe that points the gripper at the sky
(reBot's `raised`) or off to the side (Panda's keyframe 0). Wrist cameras then see nothing
useful, and the arm starts far from the work area. Random search over joint limits, scored on
(a) approach axis aimed at LOOK_AT, (b) preference for pointing downward, (c) a standoff of
~20 cm. joint1 is clamped so the arm faces forward rather than contorting round its own base --
without that clamp the unconstrained optimum was a 133-degree base rotation reaching across.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

import zero_layout as L

J1_CLAMP = 0.45
STANDOFF = 0.20
N = 400_000

# Keep every joint this far (rad) from both of its limits. A home pose is a STARTING pose: the
# arm has to be able to move away from it in every direction. The first version of this search
# scored only aim and standoff, and returned a reBot left pose sitting 0.013 rad from joint3's
# limit -- from which reaching the lemon 12 cm below simply failed, because a joint already
# against its stop cannot contribute. Costs nothing: the search has 400k samples to spend.
LIMIT_MARGIN = 0.15


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    side = sys.argv[2] if len(sys.argv) > 2 else "left"
    r = L.ROBOTS[key]
    m = mujoco.MjModel.from_xml_path(str(r["mjcf"]))
    d = mujoco.MjData(m)
    eef = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r["eef_body"])
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in r["arm_joints"]]
    assert all(i >= 0 for i in jid) and eef >= 0
    adr = [m.jnt_qposadr[i] for i in jid]
    lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
    lo[0], hi[0] = max(lo[0], -J1_CLAMP), min(hi[0], J1_CLAMP)

    # target in THIS arm's own base frame. Searched per side rather than mirrored: the correct
    # mirror is robot-specific (measured -- reBot needs only joint1 negated, Panda needs
    # j1/j3/j5, because their joint axis conventions differ), and copying the left pose to the
    # right arm aimed it 0.43 m away from the work area.
    # Each arm aims at ITS OWN role's target: left at the pick object, right at the tray.
    # Aiming both at one shared point made the two grippers face each other -- wrong for a
    # handover, and obvious the moment it was rendered.
    tgt = np.array(L.HOME_TARGET[side]) - np.array(L.robot_mounts(key)[side])
    rng = np.random.default_rng(1)
    best = None
    for _ in range(N):
        q = rng.uniform(lo, hi)
        d.qpos[adr] = q
        mujoco.mj_forward(m, d)
        p = d.xpos[eef]
        ap = d.xmat[eef].reshape(3, 3)[:, 2]          # local +z = approach
        v = tgt - p
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        # ⚠️ REJECT SELF-COLLIDING POSES. This model is a SINGLE arm with no table and no
        # object, so any contact mj_forward finds is the arm intersecting itself. Without this
        # the search happily returned a reBot left pose with link3 driven 0.83 mm into link5:
        # the pose renders fine and IK solves fine, but the sim spends a constant -0.71 Nm of
        # constraint force fighting it, so left_joint5 can never reach its command and the arm
        # keeps a residual error no amount of servo tuning removes.
        if d.ncon:
            continue
        if (q - lo).min() < LIMIT_MARGIN or (hi - q).min() < LIMIT_MARGIN:
            continue
        aim = float(ap @ (v / n))
        down = float(ap @ np.array([0, 0, -1.0]))
        score = aim + 0.6 * down - 2.5 * abs(n - STANDOFF)
        if best is None or score > best[0]:
            best = (score, q.copy(), aim, down, n)
    score, q, aim, down, n = best
    print(f"[{key}/{side}] score {score:.3f}  aim {aim:.3f}  down {down:.3f}  standoff {n:.3f} m")
    print(f'        "{side}": (' + ", ".join(f"{v:.3f}" for v in q) + "),")


if __name__ == "__main__":
    main()
