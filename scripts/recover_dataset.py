"""Rebuild a LeRobotDataset whose episode index was lost, keeping the episodes that survived.

    python3 scripts/recover_dataset.py  ~/zero_data/rebot_pick_place.broken  ~/zero_data/rebot_recovered

WHEN THIS IS NEEDED. LeRobotDataset writes `meta/episodes/*.parquet` -- the episode index -- only
when its writers are closed. A recorder that is killed leaves `meta/info.json` claiming N episodes
with no index, so the dataset cannot be opened at all, even though the frame data and the videos
are sitting there intact. That is worth repairing rather than re-recording: the frames represent
real teleoperation time.

WHAT SURVIVES. Everything except the episode being written at the moment of death:

  * per-frame features (state, action, joints, force, ik_residual, depth) live in
    `data/chunk-*/file-*.parquet`, one file per episode. A file that was mid-write has no parquet
    footer and is unreadable; the earlier ones are fine.
  * RGB lives in one mp4 per camera with every episode concatenated, and the mp4 IS usually
    complete because it is finalised per episode. Episode boundaries are recovered from the
    per-episode frame counts, since the index that recorded them is what went missing.

HOW. Rather than hand-write the episode index (a ~200-column schema of pointers and per-feature
statistics, easy to get subtly wrong), this replays the surviving frames through
`LeRobotDataset.add_frame` / `save_episode` into a NEW dataset, so lerobot computes the metadata
and the stats itself. Slower, and correct by construction.

Videos are decoded in one sequential pass per camera, never held in memory: 5000 frames x 3
cameras of 224x224 RGB is over 2 GB.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def readable_episodes(src: Path) -> list[tuple[int, Path, int]]:
    """(episode_index, parquet path, row count) for every episode whose parquet still opens."""
    out, skipped = [], []
    for f in sorted(glob.glob(str(src / "data" / "**" / "*.parquet"), recursive=True)):
        try:
            t = pq.read_table(f)
        except Exception:
            skipped.append(Path(f).name)
            continue
        eps = sorted({int(v) for v in t.column("episode_index").to_pylist()})
        if len(eps) != 1:
            skipped.append(f"{Path(f).name} (spans {eps})")
            continue
        out.append((eps[0], Path(f), t.num_rows))
    out.sort()
    if skipped:
        print(f"  unreadable, skipping: {skipped}")
    return out


def frame_iter(path: Path):
    """Yield RGB frames from an mp4 in order, one at a time."""
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- point at a fresh directory")

    info = json.loads((src / "meta" / "info.json").read_text())
    features = info["features"]
    # ⚠️ Shapes come back from JSON as LISTS ([20]) while numpy reports TUPLES ((20,)), and
    # lerobot's validate_frame compares them with !=, so every feature fails validation with
    # "does not have the expected shape" even though the numbers match. Coerce on the way in.
    for spec in features.values():
        if isinstance(spec.get("shape"), list):
            spec["shape"] = tuple(spec["shape"])
    fps = int(info["fps"])
    robot = info.get("robot_type") or "rebot"
    print(f"source claims {info['total_episodes']} episodes / {info['total_frames']} frames "
          f"at {fps} fps")

    eps = readable_episodes(src)
    if not eps:
        raise SystemExit("no readable episode parquet found -- nothing to recover")
    total = sum(n for _, _, n in eps)
    print(f"recovering {len(eps)} episodes, {total} frames: "
          + ", ".join(f"ep{e}={n}" for e, _, n in eps))

    cams = [k.split(".")[-1] for k in features if k.startswith("observation.images.")]
    vids = {c: src / "videos" / f"observation.images.{c}" / "chunk-000" / "file-000.mp4"
            for c in cams}
    for c, v in vids.items():
        if not v.exists():
            raise SystemExit(f"missing video for {c}: {v}")

    # ⚠️ The mp4s hold EVERY episode back to back, including ones whose parquet died, so the frame
    # offset must be the cumulative count over ALL episodes in order -- not over the recovered
    # ones. Skipping a dead episode without skipping its video frames would silently pair every
    # later episode with the wrong images.
    lengths: dict[int, int] = {e: n for e, _, n in eps}
    order = sorted(lengths)
    if order != list(range(order[0], order[-1] + 1)):
        print(f"  WARNING: episode indices are not contiguous ({order}); frame offsets for "
              f"episodes after a gap cannot be derived and those episodes are dropped")
        contiguous = []
        expect = order[0]
        for e in order:
            if e != expect:
                break
            contiguous.append(e)
            expect += 1
        lengths = {e: lengths[e] for e in contiguous}
        eps = [(e, p, n) for e, p, n in eps if e in lengths]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(repo_id=f"zero/{robot}", fps=fps, features=features,
                               root=dst, robot_type=robot,
                               image_writer_processes=0, image_writer_threads=2 * len(cams))
    streams = {c: frame_iter(v) for c, v in vids.items()}
    consumed = 0
    for e, path, n in eps:
        skip = sum(lengths[k] for k in sorted(lengths) if k < e) - consumed
        for _ in range(skip):                        # advance past dead episodes' video
            for s in streams.values():
                next(s)
        consumed += skip

        table = pq.read_table(path).to_pylist()
        for row in table:
            frame = {}
            for key, spec in features.items():
                if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                    continue
                if key.startswith("observation.images."):
                    cam = key.split(".")[-1]
                    frame[key] = next(streams[cam])
                else:
                    val = row[key]
                    arr = np.asarray(val, dtype=np.float32)
                    frame[key] = arr
            frame["task"] = info.get("task") or "pick up the can and place it in the tray"
            ds.add_frame(frame)
        consumed += n
        ds.save_episode()
        print(f"  recovered episode {e}: {n} frames")

    ds.finalize()
    print(f"\\nwrote {dst}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
    print("point the recorder at this root to carry on; the next episode will be "
          f"{ds.meta.total_episodes}")


if __name__ == "__main__":
    main()
