"""Rebuild a LeRobotDataset with some episodes removed.

    python3 scripts/drop_episodes.py SRC DST 32 33 34 35

WHY NOT `lerobot-edit-dataset --operation.type delete_episodes`: tried, and it fails partway on
this layout -- it looks for a video file under a `<root>_old` directory it never created, leaving
the dataset half-moved. Losing confidence in a tool mid-operation on the only copy of 30-odd
episodes of teleoperation is not worth the convenience.

WHY A FULL REBUILD. The videos are not one-file-per-episode: several episodes share an mp4 (here,
36 episodes across 4 files per camera). Removing one episode's frames therefore means re-encoding
the file that contains it, and renumbering `episode_index`, `dataset_from_index` and the video
timestamp pointers for everything after it. Rather than rewrite that index by hand, this replays
the surviving frames through lerobot's own add_frame/save_episode so lerobot recomputes all of it.

⚠️ RE-ENCODES EVERY KEPT EPISODE, so repeated deletions stack generational video loss (measured at
~1.3/255 mean absolute per pass -- negligible once, worth batching deletions rather than doing
them one at a time).

Writes to a NEW directory and never touches the source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    drop = {int(a) for a in sys.argv[3:]}
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- point at a fresh directory")

    info = json.loads((src / "meta" / "info.json").read_text())
    features = info["features"]
    for spec in features.values():                      # JSON gives lists, numpy reports tuples
        if isinstance(spec.get("shape"), list):
            spec["shape"] = tuple(spec["shape"])
    fps, robot = int(info["fps"]), info.get("robot_type") or "rebot"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(repo_id=f"zero/{robot}", root=src)
    total = source.meta.total_frames
    keep = sorted(set(range(source.meta.total_episodes)) - drop)
    print(f"source: {source.meta.total_episodes} episodes / {total} frames")
    print(f"dropping {sorted(drop)}  ->  keeping {len(keep)} episodes")

    cams = [k for k in features if k.startswith("observation.images.")]
    depth = [k for k in features if k.startswith("observation.depth.")]
    vecs = [k for k, v in features.items()
            if k not in cams and k not in depth
            and k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")]

    out = LeRobotDataset.create(repo_id=f"zero/{robot}", fps=fps, features=features,
                                root=dst, robot_type=robot,
                                image_writer_processes=0, image_writer_threads=2 * len(cams))
    prev_ep, written, kept_frames = None, 0, 0
    for i in range(total):
        row = source[i]
        ep = int(row["episode_index"])
        if ep in drop:
            continue
        if prev_ep is not None and ep != prev_ep:
            out.save_episode()
            written += 1
            print(f"  wrote episode {written - 1} (was {prev_ep})")
        frame = {}
        for k in cams:
            img = row[k].numpy()                        # (3,H,W) float 0..1
            frame[k] = (np.transpose(img, (1, 2, 0)) * 255).round().astype(np.uint8)
        for k in depth:
            frame[k] = np.asarray(row[k], dtype=np.float32)
        for k in vecs:
            frame[k] = np.asarray(row[k], dtype=np.float32)
        frame["task"] = row.get("task") or info.get("task") or "task"
        out.add_frame(frame)
        prev_ep = ep
        kept_frames += 1
        if kept_frames % 2000 == 0:
            print(f"  ... {kept_frames} frames")
    if prev_ep is not None:
        out.save_episode()
        written += 1
        print(f"  wrote episode {written - 1} (was {prev_ep})")
    out.finalize()
    print(f"\nwrote {dst}: {out.meta.total_episodes} episodes, {out.meta.total_frames} frames")


if __name__ == "__main__":
    main()
