"""The ZERO action/observation vector: the one representation that crosses embodiments.

    per hand:  pos(3) + rot6d(6) + grip(1)  = 10
    both hands:                              20

This lives in SE(3) x R. `pos` is the translation in R^3; `rot6d` encodes the SO(3) rotation as
the first two COLUMNS of the rotation matrix, re-orthonormalised on read; `grip` is a normalised
[0,1] open/close scalar and is not part of SE(3) at all.

WHY rot6d AND NOT QUATERNION OR EULER. Both minimal parameterisations of SO(3) are discontinuous:
quaternions double-cover (q and -q are the same rotation, so a regression target flips sign
arbitrarily) and Euler angles gimbal-lock. rot6d is deliberately over-parameterised -- 9 numbers
for 6 DoF -- to stay continuous everywhere, which is what a network needs to regress against.
(Zhou et al. 2019, On the Continuity of Rotation Representations in Neural Networks.)

WHY ABSOLUTE, NOT DELTAS. Poses are absolute in the world/table frame, so nothing drifts and a
recorded demo means the same thing on any robot. Gamepad input is naturally a delta; integrating
delta -> absolute happens in the teleop node, never in the dataset.

WHY THIS IS THE TRANSFER MECHANISM. reBot (6-DoF + parallel jaw), Panda (7-DoF + parallel jaw)
and G1 (7-DoF + 3-finger hand) share NO joint space. They share this. The identical 20-vector
goes into each robot's own IK and comes out as a completely different joint trajectory -- that
one swap is the whole of cross-embodiment here. It is also why joint positions must never enter
the policy's observation: the moment they do it stops transferring.
"""

from __future__ import annotations

import numpy as np

DIM_PER_HAND = 10
DIM = 20
SIDES = ("left", "right")


def hand_slice(side: str) -> slice:
    i = SIDES.index(side) * DIM_PER_HAND
    return slice(i, i + DIM_PER_HAND)


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """SO(3) -> 6D: the first two columns of R, flattened."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]])


def rot_from_6d(v: np.ndarray) -> np.ndarray:
    """6D -> SO(3) by Gram-Schmidt.

    A network's raw output is NOT orthonormal, so this projection is not optional: feeding a
    non-rigid matrix into SE(3) silently corrupts the IK error term, because `log6` of a
    non-rigid transform is meaningless. Same operation as `projectToSO3()` in Sid's Panda teleop.
    Degenerate inputs (zero, or two parallel columns) fall back to a valid rotation rather than
    producing NaN.
    """
    v = np.asarray(v, dtype=float).reshape(6)
    a, b = v[:3], v[3:]
    na = np.linalg.norm(a)
    if na < 1e-9:
        return np.eye(3)
    c1 = a / na
    b_perp = b - np.dot(c1, b) * c1
    nb = np.linalg.norm(b_perp)
    if nb < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(c1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b_perp = tmp - np.dot(c1, tmp) * c1
        nb = np.linalg.norm(b_perp)
    c2 = b_perp / nb
    return np.column_stack([c1, c2, np.cross(c1, c2)])


def pack(poses: dict[str, tuple[np.ndarray, np.ndarray]],
         grips: dict[str, float]) -> np.ndarray:
    out = np.zeros(DIM)
    for side in SIDES:
        pos, R = poses[side]
        s = hand_slice(side).start
        out[s:s + 3] = pos
        out[s + 3:s + 9] = rot_to_6d(R)
        out[s + 9] = grips[side]
    return out


def unpack(vec: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]],
                                     dict[str, float]]:
    vec = np.asarray(vec, dtype=float).reshape(DIM)
    poses, grips = {}, {}
    for side in SIDES:
        s = hand_slice(side).start
        poses[side] = (vec[s:s + 3].copy(), rot_from_6d(vec[s + 3:s + 9]))
        grips[side] = float(vec[s + 9])
    return poses, grips


def to_relative(state: "np.ndarray", action: "np.ndarray") -> "np.ndarray":
    """Express an absolute action RELATIVE to the hand's current pose -- UMI's representation.

        rel = to_relative(sample["observation.state"], sample["action"])

    Same 20-dim layout as `action`: per hand pos(3) + rot6d(6) + grip(1). The gripper value is
    already an opening fraction and is passed through unchanged; only the pose becomes relative.

    WHY IT IS DERIVED AND NOT RECORDED. It is an exact function of two things the dataset already
    stores, so recording it would be redundant -- and adding a column changes the dataset schema,
    which stops the recorder appending to episodes captured before the change. Derive at training
    time and the choice of representation stays open.

    WHY THE TOOL FRAME. The delta is rotated into the CURRENT hand frame (`R^T (p_cmd - p_meas)`)
    rather than left in the world. A world-frame delta still carries the table's orientation, so
    an arm mounted at a different angle -- or a humanoid whose base moves, which is where this is
    going -- would read the same physical motion as different numbers. Nothing in the tool-frame
    form refers to a frame the two embodiments must agree on, which is the property that makes it
    portable (UMI, arXiv 2402.10329).
    """
    import numpy as np

    poses_m, _ = unpack(np.asarray(state, dtype=float))
    poses_c, grips = unpack(np.asarray(action, dtype=float))
    out = {}
    for side in SIDES:
        p_m, R_m = poses_m[side]
        p_c, R_c = poses_c[side]
        out[side] = (R_m.T @ (p_c - p_m), R_m.T @ R_c)
    return pack(out, grips)
