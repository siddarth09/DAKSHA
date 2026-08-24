"""Solve each arm's home pose with IK instead of searching for it.

    python scripts/solve_home.py rebot
    python scripts/solve_home.py panda

WHY NOT RANDOM SEARCH (what findhome.py did). A home pose has to satisfy four things at once:
the tool point a fixed standoff above the task target, the approach axis pointing down, every
joint clear of its limits, and no self-collision. Random sampling finds poses that satisfy the
soft score, not the constraints -- only ~2% of samples point within 26 deg of vertical, so with
`down` as a scoring term the optimiser traded it away and returned reBot poses tilted 55-74 deg
off vertical, which arrive sideways and knock a standing can over. Promoting `down` to a hard
constraint fixed the tilt but then the best standoff it could find was 0.416 m against a target
of 0.20, because the feasible set is too thin to hit by sampling.

The pose is not a search problem: it is exactly what IK solves. Build the desired gripper frame
(approach axis down, closing axis across the table), solve from many seeds, and keep the solution
that clears the limits and does not self-collide. Deterministic, exact, and it reports honestly
when the pose is out of reach rather than silently returning the least-bad sample.

BOTH AXES ARE MEASURED from the model, never assumed: the approach axis is the direction from the
gripper body to where the fingers close, and the closing axis is the direction between the two
fingers. Assuming "+z is forward" is what put the reBot's tool point 113 mm off in the first place.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
import pinocchio as pin

import zero_layout as L

sys.path.insert(0, str(L.ROOT / "scripts"))
from measure_tcp import grasp_centre           # noqa: E402

STANDOFF = 0.20
LIMIT_MARGIN = 0.15
N_SEEDS = 400


def gripper_axes(key: str, side: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """(approach, closing) unit axes in the eef body's LOCAL frame, both measured."""
    r = L.ROBOTS[key]
    m = mujoco.MjModel.from_xml_path(str(L.PKG / "mjcf" / f"zero_{key}.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    eb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, L.prefixed(side, r["eef_body"]))
    R = d.xmat[eb].reshape(3, 3)

    fingers = [m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, L.prefixed(side, j))]
               for j in r["gripper_joints"]]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in fingers]
    c0, c1 = (grasp_centre(m, d, n) for n in names)

    approach = np.array(r["eef_offset"], dtype=float)
    approach = R.T @ (R @ approach)                       # local, as stored
    approach /= np.linalg.norm(approach)
    closing = R.T @ (c1 - c0)
    closing /= np.linalg.norm(closing)
    # Orthogonalise: the closing axis must be perpendicular to the approach axis.
    closing -= approach * float(closing @ approach)
    closing /= np.linalg.norm(closing)
    return approach, closing


def target_rotation(approach_l, closing_l, closing_w=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Rotation putting the approach axis straight down and the closing axis across the table."""
    down = np.array([0.0, 0.0, -1.0])
    cw = np.array(closing_w, dtype=float)
    cw -= down * float(cw @ down)
    cw /= np.linalg.norm(cw)
    M_l = np.column_stack([closing_l, np.cross(approach_l, closing_l), approach_l])
    M_w = np.column_stack([cw, np.cross(down, cw), down])
    return M_w @ M_l.T


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    r = L.ROBOTS[key]
    approach_l, closing_l = gripper_axes(key)
    R_goal = target_rotation(approach_l, closing_l)
    print(f"[{key}] approach(local) {np.round(approach_l, 3)}  "
          f"closing(local) {np.round(closing_l, 3)}")

    sys.path.insert(0, str(L.PKG.parent / "zero_control"))
    from zero_control.ik import ArmIK

    m = mujoco.MjModel.from_xml_path(str(L.PKG / "mjcf" / f"zero_{key}.xml"))
    d = mujoco.MjData(m)
    urdf = str(L.PKG / "urdf" / f"zero_{key}.urdf")

    for side in L.SIDES:
        jn = [L.prefixed(side, j) for j in r["arm_joints"]]
        ik = ArmIK(urdf, r["urdf_eef_frame"].format(side=side), jn, r["eef_offset"])
        lo = ik.model.lowerPositionLimit[ik.qidx]
        hi = ik.model.upperPositionLimit[ik.qidx]
        goal = np.array(L.HOME_TARGET[side], dtype=float) + np.array([0.0, 0.0, STANDOFF])
        tgt = pin.SE3(R_goal, goal)

        adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]
        rng = np.random.default_rng(0)
        best = None
        for _ in range(N_SEEDS):
            q = np.zeros(ik.model.nq)
            q[ik.qidx] = rng.uniform(lo, hi)
            for _ in range(400):
                q = ik.step(q, tgt).q
            res = ik.step(q, tgt)
            if res.pos_err > 2e-3 or res.rot_err > 0.05:
                continue
            qa = q[ik.qidx]
            margin = float(min((qa - lo).min(), (hi - qa).min()))
            if margin < LIMIT_MARGIN:
                continue
            d.qpos[:] = 0
            for i, a in enumerate(adr):
                d.qpos[a] = qa[i]
            mujoco.mj_forward(m, d)
            selfcol = any(
                (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or "")
                .startswith(f"{side}_")
                and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or "")
                .startswith(f"{side}_")
                for c in d.contact[:d.ncon])
            if selfcol:
                continue
            # Prefer the largest limit margin among valid solutions -- most room to move.
            if best is None or margin > best[0]:
                best = (margin, qa.copy(), res.pos_err, res.rot_err)
        if best is None:
            print(f"  {side:5} NO valid pose: standoff {STANDOFF} m above "
                  f"{np.round(goal, 3)} is unreachable with the approach axis vertical")
            continue
        margin, qa, pe, re_ = best
        print(f"  {side:5} pos_err {pe*1000:6.3f} mm  rot_err {np.degrees(re_):6.3f} deg  "
              f"limit margin {margin:.3f} rad")
        print(f'        "{side}": (' + ", ".join(f"{v:.3f}" for v in qa) + "),")


if __name__ == "__main__":
    main()
