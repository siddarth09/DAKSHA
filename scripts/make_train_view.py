"""Build a training view of a recorded dataset: same episodes, fewer columns.

    python3 scripts/make_train_view.py [SRC_DATASET] [VIEW_PREFIX]

The recorded dataset carries three 224x224 float32 depth maps per frame, 600 kB/frame, which is
20.4 GiB of parquet against 367 MiB of video. The policy never reads them. Training against the
raw dataset would pull that 20 GiB through the dataloader every epoch for nothing, and random
access across it thrashes the page cache. (It also OOM-killed a plain pandas read of the whole
set.)

A view, not a rebuild: videos are symlinked and never re-encoded, since the frames are already
correct and re-encoding costs hours and a generation of quality. Only the small vector columns
are rewritten, so a view takes seconds and ~30 MB.

Two views, because lerobot's `dataset_to_policy_features` types every `observation.*` key as
STATE (datasets/feature_utils.py) and `make_policy` only auto-derives `input_features` when they
are unset. Whatever is in the view is a policy input. The no-force view trains a stock SmolVLA
baseline through the unmodified CLI; the force view is the VLFA arm. The difference between them
is the ablation, rather than a config flag we might get wrong in one run and not the other.

Incomplete episodes are not dropped here, since that would renumber every index. Select them at
train time with `--dataset.episodes`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

import sys

SRC = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "zero_data" / "cross_v2"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else SRC.name.replace("_v1", "").replace("_v2", "")
VIEWS = {
    f"{PREFIX}_base": ["observation.state"],                       # stock SmolVLA baseline
    f"{PREFIX}_vlfa": ["observation.state", "observation.force"],  # + the F in VLFA
}
IMAGES = ["observation.images.front", "observation.images.left_wrist",
          "observation.images.right_wrist"]
BOOKKEEPING = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def build(dst: Path, obs_keys: list[str]) -> None:
    keep = [*obs_keys, "action", *BOOKKEEPING]
    keep_all = [*keep, *IMAGES]                       # images live in videos/, not the parquet
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    info = json.loads((SRC / "meta" / "info.json").read_text())
    dropped = [k for k in info["features"] if k not in keep_all]
    info["features"] = {k: v for k, v in info["features"].items() if k in keep_all}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    stats = json.loads((SRC / "meta" / "stats.json").read_text())
    (dst / "meta" / "stats.json").write_text(
        json.dumps({k: v for k, v in stats.items() if k in keep_all}, indent=4))
    shutil.copy2(SRC / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")

    # episode index: keep the layout and video pointers, drop only the per-feature stats blocks
    for src_f in sorted((SRC / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(src_f)
        cols = [c for c in t.schema.names
                if not c.startswith("stats/") or c.split("/")[1] in keep_all]
        out = dst / "meta" / "episodes" / src_f.parent.name / src_f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t.select(cols), out)

    n_rows = 0
    for src_f in sorted((SRC / "data").rglob("*.parquet")):
        t = pq.read_table(src_f, columns=[c for c in pq.read_schema(src_f).names if c in keep])
        out = dst / "data" / src_f.parent.name / src_f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
        n_rows += t.num_rows

    (dst / "videos").symlink_to(SRC / "videos")       # never re-encode
    mb = sum(f.stat().st_size for f in (dst / "data").rglob("*.parquet")) / 2**20
    print(f"{dst.name:11} {n_rows} rows  parquet {mb:7.1f} MiB  "
          f"obs={obs_keys}  dropped {len(dropped)}: {', '.join(sorted(dropped))}")


if __name__ == "__main__":
    src_mb = sum(f.stat().st_size for f in (SRC / "data").rglob("*.parquet")) / 2**20
    print(f"source {SRC.name}: parquet {src_mb:.0f} MiB")
    for name, obs in VIEWS.items():
        build(Path.home() / "zero_data" / name, obs)
