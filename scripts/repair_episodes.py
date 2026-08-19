#!/usr/bin/env python
"""Repair corrupted meta/episodes for a LeRobot v3.0 dataset.

Root cause: meta/episodes/ contains stale overlapping shards (file-000 claimed
eps 0-20, file-001 claimed eps 21-29) that overlap the correct per-data-file
shards (file-002..029). load_nested_dataset() globs ALL of them -> 88 rows for
60 real episodes -> broken global frame index mapping -> torchcodec requests
frame indices past the real video length. Also ep24's video pointer points to
file-14 instead of file-15.

Fix: rebuild a single clean 60-row meta/episodes/chunk-000/file-000.parquet from
ground truth (data parquets + actual video frame counts), preserving per-episode
stats columns. Ground-truth mapping: an episode's video file_index == its data
file_index, and within a video file episodes are concatenated in episode order.

Usage:
  python repair_episodes.py <DATASET_ROOT>            # dry-run, validate only
  python repair_episodes.py <DATASET_ROOT> --apply <BACKUP_TAG>
"""

import glob
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

R = Path(sys.argv[1])
APPLY = "--apply" in sys.argv
BACKUP_TAG = sys.argv[sys.argv.index("--apply") + 1] if APPLY else None

info = json.loads((R / "meta/info.json").read_text())
FPS = info["fps"]
print(f"=== dataset: {R}")
print(f"=== fps={FPS} info.total_episodes={info['total_episodes']} info.total_frames={info['total_frames']}")

# ---- 1. current (corrupt) episode metadata ----------------------------------
ep_files = sorted(glob.glob(str(R / "meta/episodes/chunk-000/*.parquet")))
# Stale/old-generation shards (coarse packing) that overlap the per-data-file
# shards. Rows from these are de-prioritized when carrying over stats/tasks.
STALE_SHARDS = {"file-000.parquet", "file-001.parquet"}
frames = []
for f in ep_files:
    d = pd.read_parquet(f)
    d = d.loc[:, ~d.columns.duplicated()]
    # lower priority = preferred. fine shards (1) win over stale shards (0 -> 2).
    d["__prio"] = 2 if Path(f).name in STALE_SHARDS else 1
    frames.append(d)
cur = pd.concat(frames).reset_index(drop=True)
print(f"\n[current] {len(ep_files)} shards, {len(cur)} rows, {cur['episode_index'].nunique()} unique episodes")
# deduped base table: prefer fine-shard rows for carried-over stats/tasks columns.
base = (
    cur.sort_values(["episode_index", "__prio"])
    .drop_duplicates("episode_index", keep="first")
    .drop(columns="__prio")
    .reset_index(drop=True)
)
assert len(base) == cur["episode_index"].nunique()

vkeys = [c.split("/")[1] for c in base.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
print(f"[current] video keys ({len(vkeys)}): {vkeys}")

# ---- 2. ground truth from data parquets -------------------------------------
data_files = sorted(glob.glob(str(R / "data/chunk-000/*.parquet")))
ep_data_file = {}  # episode_index -> data file_index
ep_length = {}  # episode_index -> #rows (frames)
file_eps = {}  # data file_index -> [episodes in order]
for p in data_files:
    fi = int(Path(p).stem.split("-")[1])
    s = pd.read_parquet(p, columns=["episode_index"])["episode_index"]
    for ep in s.unique().tolist():
        ep = int(ep)
        ep_data_file[ep] = fi
        ep_length[ep] = int((s == ep).sum())
        file_eps.setdefault(fi, []).append(ep)
for fi, eps in file_eps.items():
    file_eps[fi] = sorted(eps)

episodes = sorted(ep_length)
print(f"\n[data] {len(data_files)} data files, {len(episodes)} episodes, total frames={sum(ep_length.values())}")
if episodes != list(range(len(episodes))):
    print("!! WARNING: episode indices are not a contiguous 0..N range")


# ---- 3. actual video frame counts -------------------------------------------
def num_frames(vkey, fi):
    p = R / f"videos/{vkey}/chunk-000/file-{fi:03d}.mp4"
    return VideoDecoder(str(p)).metadata.num_frames


# ---- 4. recompute ground-truth fields ---------------------------------------
# global dataset_from/to_index (episode order)
ds_from, ds_to, cum = {}, {}, 0
for ep in episodes:
    ds_from[ep] = cum
    cum += ep_length[ep]
    ds_to[ep] = cum

# per video key/file: cumulative frame offsets -> from/to timestamp per episode
vid_from_ts = {vk: {} for vk in vkeys}  # vk -> ep -> from_ts
vid_to_ts = {vk: {} for vk in vkeys}
problems = []
for vk in vkeys:
    for fi, eps_in_file in file_eps.items():
        off = 0
        for ep in eps_in_file:
            vid_from_ts[vk][ep] = off / FPS
            off += ep_length[ep]
            vid_to_ts[vk][ep] = off / FPS
        actual = num_frames(vk, fi)
        if off != actual:
            problems.append(
                f"  [{vk}] file-{fi:03d}: sum(lengths)={off} != video_frames={actual} (diff {off - actual})"
            )

print("\n=== ground-truth video-length validation ===")
if problems:
    print(f"!! {len(problems)} (key,file) MISMATCHES between summed episode lengths and real video frames:")
    for pmsg in problems[:40]:
        print(pmsg)
    print("!! These indicate genuinely truncated/short videos; aborting before any write.")
    if APPLY:
        sys.exit(2)
else:
    print("OK: every video file's frame count == sum of its episodes' lengths (all keys).")

# ---- 5. build clean table by overwriting ground-truth columns ---------------
new = base.copy()
new = new.sort_values("episode_index").reset_index(drop=True)
mismatch_len = []
for i, ep in enumerate(new["episode_index"].astype(int)):
    ep = int(ep)
    if int(new.at[i, "length"]) != ep_length[ep]:
        mismatch_len.append((ep, int(new.at[i, "length"]), ep_length[ep]))
    new.at[i, "length"] = ep_length[ep]
    new.at[i, "dataset_from_index"] = ds_from[ep]
    new.at[i, "dataset_to_index"] = ds_to[ep]
    new.at[i, "data/chunk_index"] = 0
    new.at[i, "data/file_index"] = ep_data_file[ep]
    for vk in vkeys:
        new.at[i, f"videos/{vk}/chunk_index"] = 0
        new.at[i, f"videos/{vk}/file_index"] = ep_data_file[ep]
        new.at[i, f"videos/{vk}/from_timestamp"] = vid_from_ts[vk][ep]
        new.at[i, f"videos/{vk}/to_timestamp"] = vid_to_ts[vk][ep]
    if "meta/episodes/chunk_index" in new.columns:
        new.at[i, "meta/episodes/chunk_index"] = 0
    if "meta/episodes/file_index" in new.columns:
        new.at[i, "meta/episodes/file_index"] = 0

if mismatch_len:
    print(
        f"\n[note] recomputed length differs from stored length for {len(mismatch_len)} eps "
        f"(using data-derived length): {mismatch_len[:10]}"
    )

# ---- 6. final越界 check: round(to_ts*fps) <= video frames -------------------
print("\n=== final bound check (round(to_ts*fps) <= video frames) ===")
bad = []
for i, ep in enumerate(new["episode_index"].astype(int)):
    ep = int(ep)
    for vk in vkeys:
        fi = ep_data_file[ep]
        to_idx = round(new.at[i, f"videos/{vk}/to_timestamp"] * FPS)
        nf = num_frames(vk, fi)
        if to_idx > nf:
            bad.append((ep, vk, to_idx, nf))
if bad:
    print(f"!! {len(bad)} still out of bounds:")
    for b in bad[:40]:
        print("  ", b)
else:
    print(f"OK: all {len(new)} episodes x {len(vkeys)} video keys are within bounds.")

# show what changed for the previously-broken episodes
print("\n=== sample of corrected rows ===")
cols = [
    "episode_index",
    "length",
    "dataset_from_index",
    "dataset_to_index",
    "data/file_index",
    "videos/observation.images.head/file_index",
    "videos/observation.images.head/from_timestamp",
    "videos/observation.images.head/to_timestamp",
]
print(new[new["episode_index"].isin([0, 1, 2, 15, 23, 24, 25, 30])][cols].to_string(index=False))

print(f"\n[result] rebuilt table: {len(new)} rows, {new['episode_index'].nunique()} unique episodes")
total_frames_new = int(new["length"].sum())
print(
    f"[result] total frames (sum length) = {total_frames_new} "
    f"(info.json says {info['total_frames']}: "
    f"{'MATCH' if total_frames_new == info['total_frames'] else 'DIFFERS'})"
)

# ---- 7. apply ---------------------------------------------------------------
if not APPLY:
    print("\n*** DRY-RUN only. No files modified. Re-run with --apply <TAG> to write. ***")
    sys.exit(0)

if bad or problems:
    print("\n!! Refusing to apply because validation failed above.")
    sys.exit(2)

ep_dir = R / "meta/episodes"
backup = R / f"meta/episodes_backup_{BACKUP_TAG}"
print(f"\n[apply] backing up {ep_dir} -> {backup}")
shutil.copytree(ep_dir, backup)

# write single consolidated parquet using the ORIGINAL schema (preserve dtypes)
orig_schema = pq.read_schema(ep_files[0])
new = new[[c for c in orig_schema.names if c in new.columns]]
table = pa.Table.from_pandas(new, schema=orig_schema, preserve_index=False)

# remove all old shards, write the single clean file
for f in glob.glob(str(ep_dir / "chunk-000/*.parquet")):
    Path(f).unlink()
out = ep_dir / "chunk-000/file-000.parquet"
pq.write_table(table, out)
print(f"[apply] wrote {out} ({table.num_rows} rows)")

# verify via lerobot's own loader
from lerobot.datasets.utils import load_episodes  # noqa: E402  (verification step, only reached after the rewrite)

loaded = load_episodes(R)
print(
    f"[verify] lerobot load_episodes() -> {len(loaded)} rows ({'OK' if len(loaded) == len(episodes) else 'UNEXPECTED'})"
)
print("[apply] done. Backup kept at:", backup)
