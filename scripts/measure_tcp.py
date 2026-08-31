"""Measure each gripper's true grasp centre and print the eef_offset to put in zero_layout.

    python scripts/measure_tcp.py

`eef_offset` moves the tool point (the pose the IK servos and the pose recorded as the action)
from the gripper body out to where the fingers actually close. Guessing it is silently
catastrophic: the reBot carried (0, 0, 0.10), 100 mm along the gripper's local +z, while its
fingers close 49 mm along local -x, putting the commanded point 113 mm from the real pinch
point. Every IK solve converged onto a point in mid-air beside the gripper, so the arm "reached"
the object and the fingers shut on nothing, which reads as a grasp-tuning or friction problem
and is neither. Panda's was 17.9 mm out, enough to bias every grasp but small enough that its
long fingers still closed on things.

The grasp centre is the midpoint of the two finger bodies' collision AABBs, which is invariant
to how far the gripper is open because the fingers move symmetrically.
"""

from __future__ import annotations

import numpy as np
import mujoco

import zero_layout as L


def pinch_point(m, d, r, side: str) -> tuple:
    """World position where the two finger pads MEET, with the gripper fully closed.

    THIS is the grasp centre. The fingers travel symmetrically, so the point where they touch when
    closed is the point an object's centre occupies when they are open around it.

    The previous version averaged the two finger bodies' collision-AABB centres, which sounds
    equivalent and is not: these finger meshes are asymmetric (one carries its knuckle), so the
    average sat 39 mm (reBot) / 22.7 mm (Panda) off the line between the pads. The jaws open
    100 mm, so an object would still land between them sometimes; grasps succeeded often enough
    to look like a tuning problem, while the failure mode was the gripper closing beside the can
    and flicking it away. The giveaway was measuring each finger's clearance separately: +9.5 mm
    on one pad, +99 mm on the other. Symmetric clearances are the acceptance test for this value.
    """
    from scipy.spatial.distance import cdist

    joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, L.prefixed(side, j))
              for j in r["gripper_joints"]]
    for j in joints:                       # fully closed: the pads touch
        d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][0]
    mujoco.mj_forward(m, d)

    clouds = []
    for j in joints:
        b = m.jnt_bodyid[j]
        pts = []
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[mid]:
                            m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].reshape(-1, 3)
            pts.append((d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g])
        assert pts, f"{side} finger has no collision mesh"
        clouds.append(np.vstack(pts))

    D = cdist(clouds[0], clouds[1])
    i, k = np.unravel_index(np.argmin(D), D.shape)
    return (clouds[0][i] + clouds[1][k]) / 2.0, float(D.min())


def measure(key: str, side: str = "left") -> tuple:
    r = L.ROBOTS[key]
    m = mujoco.MjModel.from_xml_path(str(L.PKG / "mjcf" / f"zero_{key}.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)

    eb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, L.prefixed(side, r["eef_body"]))
    R, P = d.xmat[eb].reshape(3, 3), d.xpos[eb]
    pinch, gap = pinch_point(m, d, r, side)
    measured = R.T @ (pinch - P)                       # into the eef body's local frame
    current = P + R @ np.array(r["eef_offset"], dtype=float)
    return measured, float(np.linalg.norm(current - pinch))


def main() -> None:
    for key in L.ROBOTS:
        off, err = measure(key)
        print(f"\n{key}:")
        print(f"    eef_body        {L.ROBOTS[key]['eef_body']!r}")
        print(f"    current offset  {tuple(L.ROBOTS[key]['eef_offset'])}")
        print(f"    MEASURED offset ({off[0]:.4f}, {off[1]:.4f}, {off[2]:.4f})")
        print(f"    tool point is   {err * 1000:.1f} mm from the real grasp centre")
        if err > 0.005:
            print(f"    -> paste the measured triple into ROBOTS[{key!r}]['eef_offset']")


if __name__ == "__main__":
    main()
