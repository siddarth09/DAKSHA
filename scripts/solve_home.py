"""Solve a robot's `home` joint angles so its first observation matches the training data.

    MUJOCO_GL=egl python3 scripts/solve_home.py vx300s [DATASET]

`home` is not cosmetic. `observation.state` is the policy's input, so the pose the arm boots into
has to be one the policy saw at t=0 or its very first observation is out of distribution and the
whole first action chunk is drawn from nothing. Measured on the Panda, a home that merely looked
sensible put 17 of 20 state channels beyond 3 sd of the 82 episode starts and 8 outside the full
58,995-frame range.

So the target is the mean episode-start pose of the source dataset, position AND orientation, and
the search is multi-start because a single seed lands in whichever IK branch it happens to fall
into. Solutions are then filtered on contacts: a pose can hit the target exactly with the two arms
crossed, and that seeds every later solve badly.

Prints the registry block to paste into zero_layout.ROBOTS[<robot>]["home"].
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

import zero_layout as L

SEEDS = 240
DEFAULT_STATE = Path("/tmp/train_t0_mean.npy")


def rot6d_to_mat(v: np.ndarray) -> np.ndarray:
    """First two columns of a rotation matrix, re-orthonormalised. See zero_control.action."""
    a = v[:3] / np.linalg.norm(v[:3])
    b = v[3:] - a * np.dot(a, v[3:])
    b /= np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def solve(m, d, key, side, goal, seed, iters=600):
    r = L.ROBOTS[key]
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, L.prefixed(side, j))
           for j in r["arm_joints"]]
    qad = [m.jnt_qposadr[j] for j in jid]
    dof = [m.jnt_dofadr[j] for j in jid]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
    pt, Rt = goal
    for k, a in enumerate(qad):
        d.qpos[a] = seed[k]
    ang = 9.9
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        e = pt - d.site_xpos[sid]
        Rc = d.site_xmat[sid].reshape(3, 3)
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, (Rc.T @ Rt).flatten())
        nrm = np.linalg.norm(qq[1:])
        ang = 2 * np.arctan2(nrm, qq[0])
        if np.linalg.norm(e) < 1e-5 and abs(ang) < 1e-3:
            break
        er = Rc @ (qq[1:] / (nrm + 1e-12) * ang)
        Jp = np.zeros((3, m.nv))
        Jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, Jp, Jr, sid)
        J = np.vstack([Jp[:, dof], Jr[:, dof]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.04**2 * np.eye(6),
                                   np.concatenate([e, 0.6 * er]))
        for k, j in enumerate(jid):
            q = d.qpos[qad[k]] + np.clip(dq[k], -0.05, 0.05)
            if m.jnt_limited[j]:
                q = np.clip(q, *m.jnt_range[j])
            d.qpos[qad[k]] = q
    return np.linalg.norm(pt - d.site_xpos[sid]), abs(ang), [d.qpos[a] for a in qad], qad


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "vx300s"
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_STATE
    if not src.exists():
        raise SystemExit(f"missing {src}: the mean episode-start state of the source dataset")
    mu = np.load(src)
    goals = {"left": (mu[0:3], rot6d_to_mat(mu[3:9])),
             "right": (mu[10:13], rot6d_to_mat(mu[13:19]))}

    m = mujoco.MjModel.from_xml_path(str(L.PKG / "mjcf" / f"zero_{key}.xml"))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    cands, qad = {}, {}
    for side, goal in goals.items():
        seed0 = np.array(L.ROBOTS[key]["home"][side], dtype=float)
        found = []
        for i in range(SEEDS):
            mujoco.mj_resetData(m, d)
            pe, re_, q, qa = solve(m, d, key, side, goal,
                                   seed0 if i == 0 else seed0 + rng.normal(0, 1.0, len(seed0)))
            qad[side] = qa
            if pe < 2e-4 and re_ < 4e-3:
                found.append(q)
        cands[side] = found
        print(f"  {side}: {len(found)}/{SEEDS} seeds reached the target")

    # Only ARM-vs-ARM contacts disqualify a pair. `d.ncon` also counts the can and the tray
    # resting on the table, which are always present, so filtering on it can never pass.
    subtree = {}
    for side in ("left", "right"):
        ids = set()
        stack = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                   L.prefixed(side, L.ROBOTS[key]["urdf_root"]))]
        while stack:
            b_ = stack.pop()
            ids.add(b_)
            stack += [c for c in range(m.nbody) if m.body_parentid[c] == b_ and c != b_]
        subtree[side] = ids

    def arms_touch() -> bool:
        for c in range(d.ncon):
            b1 = m.geom_bodyid[d.contact[c].geom1]
            b2 = m.geom_bodyid[d.contact[c].geom2]
            if ((b1 in subtree["left"] and b2 in subtree["right"])
                    or (b1 in subtree["right"] and b2 in subtree["left"])):
                return True
        return False

    for ql in cands["left"][:80]:
        for qr in cands["right"][:80]:
            mujoco.mj_resetData(m, d)
            for k, a in enumerate(qad["left"]):
                d.qpos[a] = ql[k]
            for k, a in enumerate(qad["right"]):
                d.qpos[a] = qr[k]
            mujoco.mj_forward(m, d)
            if not arms_touch():
                print('\n        "home": {')
                for s, q in (("left", ql), ("right", qr)):
                    print(f'            "{s}": ({", ".join(f"{v:.4f}" for v in q)}),')
                print("        },")
                return
    raise SystemExit("no collision-free pair reached the training start pose")


if __name__ == "__main__":
    main()
