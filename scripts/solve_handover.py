"""Solve the handover geometry: how far apart must the two arms be?

A handover task is only a handover if the object CANNOT be placed by the picking arm. So:
    PICK_POS      reachable by LEFT  only
    PLACE_POS     reachable by RIGHT only
    HANDOVER_POS  reachable by BOTH
If pick and place are both reachable by one arm, the policy will skip the handover entirely and
the task silently degenerates into pick-and-place. That is the failure this script prevents --
the same class of error as the unreachable handover target that cost days on the last project,
just in the opposite direction.

Sweeps base separation and reports the first value that satisfies all three constraints, with
margin. Run:  MUJOCO_GL=egl python scripts/solve_handover.py <robot>
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

import zero_layout as L

VOXEL = 0.04
N = 60_000
MARGIN = 0.06   # a pose must be this far inside/outside an envelope to count, not borderline


def cloud(key: str) -> np.ndarray:
    """EEF positions relative to the arm's own base."""
    r = L.ROBOTS[key]
    m = mujoco.MjModel.from_xml_path(str(r["mjcf"]))
    d = mujoco.MjData(m)
    eef = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r["eef_body"])
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in r["arm_joints"]]
    adr = [m.jnt_qposadr[i] for i in jid]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    rng = np.random.default_rng(0)
    pts = np.empty((N, 3))
    for i in range(N):
        d.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        pts[i] = d.xpos[eef]
    return pts


def dist_to_cloud(pts: np.ndarray, target: np.ndarray) -> float:
    """Distance from `target` to the nearest sampled reachable point. Small = reachable."""
    return float(np.min(np.linalg.norm(pts - target, axis=1)))


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    r = L.ROBOTS[key]
    pts = cloud(key)
    z = L.TABLE_TOP_Z + 0.03
    bx = r["base_x"]
    print(f"[{key}] base_x={bx}  table y span +-{L.TABLE_HALF[1]:.2f}\n")
    print(f"{'sep':>6} {'pick|L':>8} {'pick|R':>8} {'place|L':>8} {'place|R':>8} "
          f"{'hand|L':>7} {'hand|R':>7}  verdict")
    print("-" * 78)
    good = []
    for sep in np.arange(0.40, 1.45, 0.05):
        # object in front of the LEFT arm, tray in front of the RIGHT arm, handover on centreline
        pick = np.array([0.34, +sep / 2, z])
        place = np.array([0.34, -sep / 2, z])
        hand = np.array([0.34, 0.0, z + 0.10])
        lb = np.array([bx, +sep / 2, L.TABLE_TOP_Z])
        rb = np.array([bx, -sep / 2, L.TABLE_TOP_Z])
        d = {
            "pL": dist_to_cloud(pts, pick - lb), "pR": dist_to_cloud(pts, pick - rb),
            "qL": dist_to_cloud(pts, place - lb), "qR": dist_to_cloud(pts, place - rb),
            "hL": dist_to_cloud(pts, hand - lb), "hR": dist_to_cloud(pts, hand - rb),
        }
        ok = (d["pL"] < VOXEL and d["pR"] > MARGIN          # pick: left yes, right no
              and d["qR"] < VOXEL and d["qL"] > MARGIN      # place: right yes, left no
              and d["hL"] < VOXEL and d["hR"] < VOXEL)      # handover: both
        fits_table = sep / 2 + 0.10 <= L.TABLE_HALF[1]
        v = "HANDOVER REQUIRED" if ok and fits_table else ("off table" if not fits_table else "")
        if ok and fits_table:
            good.append(sep)
        print(f"{sep:6.2f} {d['pL']:8.3f} {d['pR']:8.3f} {d['qL']:8.3f} {d['qR']:8.3f} "
              f"{d['hL']:7.3f} {d['hR']:7.3f}  {v}")
    print()
    if good:
        pick_sep = good[len(good) // 2]
        print(f"VALID separations: {good[0]:.2f} .. {good[-1]:.2f} m  -> choose {pick_sep:.2f} "
              f"(middle, most margin)")
        print(f'  "base_sep": {pick_sep:.2f},')
        print(f"  PICK_POS  = (0.34, {+pick_sep/2:+.2f}, {z:.3f})")
        print(f"  PLACE_POS = (0.34, {-pick_sep/2:+.2f}, {L.TABLE_TOP_Z+0.015:.3f})")
        print(f"  HANDOVER  = (0.34,  0.00, {z+0.10:.3f})")
    else:
        print("NO separation satisfies a true handover -- the arms' envelopes overlap too much")
        print("or the table is too narrow. Levers: widen the table, or move pick/place outward "
              "in x as well as y.")


if __name__ == "__main__":
    main()
