"""Verify the DLS IK converges on both robots, before wiring any ROS around it.

Reachable targets must converge to sub-millimetre; deliberately unreachable ones must report a
large residual rather than silently returning garbage. That reporting is the whole point.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "zero_control"))
import numpy as np, pinocchio as pin
import zero_layout as L
from zero_control.ik import ArmIK

EEF_FRAME = {"rebot": "{side}_gripper_end", "panda": "{side}_hand"}

for key in ("rebot", "panda"):
    r = L.ROBOTS[key]
    urdf = str(L.PKG / "urdf" / f"zero_{key}.urdf")
    print(f"\n===== {key} =====")
    for side in L.SIDES:
        jn = [L.prefixed(side, j) for j in r["arm_joints"]]
        ik = ArmIK(urdf, EEF_FRAME[key].format(side=side), jn, r["eef_offset"])
        q0 = np.zeros(ik.model.nq)
        for i, j in enumerate(r["arm_joints"]):
            q0[ik.qidx[i]] = r["home"][side][i]
        start = ik.fk(q0)
        print(f"  {side}: home eef {np.round(start.translation,3)}  nq={ik.model.nq} "
              f"cols={ik.cols}")

        # 1. reachable: nudge 5 cm from the home pose
        for d in ([0.05,0,0], [0,0.05,0], [0,0,-0.05], [0.03,-0.03,0.03]):
            tgt = pin.SE3(start.rotation, start.translation + np.array(d))
            res = ik.solve(q0, tgt)
            ok = "OK " if res.pos_err < 1e-3 else "BAD"
            print(f"    {ok} d={d}  pos_err {res.pos_err*1000:7.3f} mm  "
                  f"rot_err {np.degrees(res.rot_err):6.3f} deg")

        # 2. the task poses, what teleop will actually command
        for name, p in (("PICK", L.PICK_POS), ("HANDOVER", L.HANDOVER_POS), ("PLACE", L.PLACE_POS)):
            tgt = pin.SE3(start.rotation, np.array(p))
            res = ik.solve(q0, tgt, iters=200)
            verdict = "reachable" if res.pos_err < 5e-3 else "OUT OF REACH"
            print(f"    {name:9} {verdict:12} pos_err {res.pos_err*1000:8.2f} mm")

        # 3. deliberately impossible: must report a large residual, not lie
        tgt = pin.SE3(start.rotation, start.translation + np.array([3.0, 0, 0]))
        res = ik.solve(q0, tgt, iters=100)
        print(f"    unreachable-by-3m -> residual {res.residual:.3f} "
              f"(clamped={res.clamped})  {'GOOD, reported' if res.pos_err > 1.0 else 'BAD, hidden'}")
