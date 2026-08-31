"""Open-loop action-chunk error of a trained checkpoint, in millimetres and degrees.

    /home/sid/lerobot_env/bin/python scripts/eval_chunk_error.py

WHY. The training log reports a flow-matching loss on NORMALISED, zero-padded 32-dim vectors
(0.094 at step 50k). That number cannot be compared against anything physical and does not say
whether the policy is accurate enough to close a gripper on a 66 mm can. This unnormalises the
predicted chunk and reports error in the units the task is actually specified in.

WHAT THIS IS NOT. Every episode here was in the training set, so this measures FIT, not
generalisation. It answers "did the checkpoint learn the demonstrations" and "which can positions
fit worst", not "will it work on a new can position". A real number needs held-out episodes or a
sim rollout.

Error is broken down by position in the chunk: SmolVLA predicts 50 actions per forward pass, which
at 10 fps is 5 SECONDS of open-loop motion, so error at k=49 matters as much as at k=0.
"""

from __future__ import annotations

import numpy as np
import torch

import sys
CKPT = (sys.argv[1] if len(sys.argv) > 1 else
        "/home/sid/zero_runs/crossv2_full_c25/checkpoints/last/pretrained_model")
ROOT = "/home/sid/zero_data/crossv2_base"
# One episode per can position in cross_v2, chosen as the most central of each cluster:
#   ep  0 (0.25,0.07)   5 (0.24,0.04)  14 (0.44,0.60)  32 (0.46,0.37)  42 (0.55,0.24)
#   ep 58 (0.33,0.22)  59 (0.30,0.23)  70 (0.48,0.24)  71 (0.50,0.51)
EPISODES = [0, 5, 14, 32, 42, 58, 59, 70, 71]
FRAMES_PER_EP = 10


def rot6d_to_R(v: np.ndarray) -> np.ndarray:
    a, b = v[:3], v[3:]
    c1 = a / max(np.linalg.norm(a), 1e-9)
    bp = b - c1 * (c1 @ b)
    c2 = bp / max(np.linalg.norm(bp), 1e-9)
    return np.column_stack([c1, c2, np.cross(c1, c2)])


def geodesic_deg(p: np.ndarray, g: np.ndarray) -> float:
    R = rot6d_to_R(p).T @ rot6d_to_R(g)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main() -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(CKPT)
    cfg.pretrained_path, cfg.device = CKPT, "cuda"
    chunk = cfg.chunk_size

    probe = LeRobotDataset("zero/base", root=ROOT, episodes=[EPISODES[0]])
    fps = probe.meta.fps
    ds = LeRobotDataset("zero/base", root=ROOT, episodes=EPISODES,
                        delta_timestamps={"action": [i / fps for i in range(chunk)]})
    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT)
    policy.eval()
    print(f"loaded {CKPT}\n{len(ds)} frames over episodes {EPISODES}, chunk={chunk} "
          f"({chunk/fps:.1f} s open loop)\n")

    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    per_ep, by_k = {}, {k: [] for k in range(chunk)}
    for ep in EPISODES:
        idx = np.where(ep_col == ep)[0]
        pick = idx[np.linspace(0, len(idx) - 1, FRAMES_PER_EP).astype(int)]
        errs = []
        for i in pick:
            s = ds[int(i)]
            batch = {k: v.unsqueeze(0).to("cuda") for k, v in s.items() if torch.is_tensor(v)}
            batch["task"] = [s["task"]]
            with torch.no_grad():
                pred = post(policy.predict_action_chunk(pre(batch)))
            p = pred[0, :, :20].float().cpu().numpy()
            g = s["action"].numpy()
            for k in range(chunk):
                dl = np.linalg.norm(p[k, 0:3] - g[k, 0:3]) * 1000
                dr = np.linalg.norm(p[k, 10:13] - g[k, 10:13]) * 1000
                by_k[k].append((dl + dr) / 2)
            errs.append({
                "L_mm": np.linalg.norm(p[:, 0:3] - g[:, 0:3], axis=1).mean() * 1000,
                "R_mm": np.linalg.norm(p[:, 10:13] - g[:, 10:13], axis=1).mean() * 1000,
                "L_deg": np.mean([geodesic_deg(p[k, 3:9], g[k, 3:9]) for k in range(chunk)]),
                "R_deg": np.mean([geodesic_deg(p[k, 13:19], g[k, 13:19]) for k in range(chunk)]),
                "grip": np.abs(p[:, [9, 19]] - g[:, [9, 19]]).mean(),
            })
        per_ep[ep] = {k: float(np.mean([e[k] for e in errs])) for k in errs[0]}

    print(f"{'ep':>4}{'L pos mm':>10}{'R pos mm':>10}{'L rot deg':>11}{'R rot deg':>11}{'grip':>8}")
    print("-" * 54)
    for ep, m in per_ep.items():
        print(f"{ep:>4}{m['L_mm']:>10.1f}{m['R_mm']:>10.1f}{m['L_deg']:>11.2f}"
              f"{m['R_deg']:>11.2f}{m['grip']:>8.3f}")
    agg = {k: float(np.mean([m[k] for m in per_ep.values()])) for k in next(iter(per_ep.values()))}
    print("-" * 54)
    print(f"{'mean':>4}{agg['L_mm']:>10.1f}{agg['R_mm']:>10.1f}{agg['L_deg']:>11.2f}"
          f"{agg['R_deg']:>11.2f}{agg['grip']:>8.3f}")
    print(f"\nposition error vs how far into the chunk (mean of both arms, mm):")
    for k in (0, 4, 9, 19, 29, 39, 49):
        if k < chunk:
            print(f"   k={k:<3} t=+{k/fps:.1f}s   {np.mean(by_k[k]):6.1f} mm")
    # reBot jaw opens to 100 mm (measured in reach_gate.py); the can is OBJECT_WIDTH = 66 mm.
    # So the TCP has (100-66)/2 = 17 mm of lateral clearance per side at the grasp. Position
    # error above that means the jaw closes beside the can or knocks it over.
    print("\nreBot jaw 100 mm vs 66 mm can -> only 17 mm of lateral clearance per side at the "
          "grasp.")


if __name__ == "__main__":
    main()
