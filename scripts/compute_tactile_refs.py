"""Precompute the per-episode reference tactile frames used by TactileDifference.

The gel sensor's undeformed appearance drifts between recording sessions, so the
usable signal is the difference against *this episode's* undeformed frame rather
than the absolute image. Frame 0 is that reference: every episode starts with the
grippers open and nothing in them.

Fetching frame 0 through LeRobot at training time would mean a second
``__getitem__`` per sample, and each one decodes every camera stream -- roughly
doubling data-loading cost. Instead this dumps all references once into a single
uint8 array that InjectTactileReference memory-maps, so the dataloader workers
share one copy through the page cache.

The script refuses to write if any episode's frame 0 has a closed gripper, since
a deformed reference silently poisons every difference in that episode.

Usage:
    python scripts/compute_tactile_refs.py --repo-id Xense/bottle-sorting-0810 \
        --out assets/tactile_refs/bottle-sorting-0810.npy
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
import pandas as pd

_DEFAULT_CAMERAS = (
    "left_tactile_0",
    "left_tactile_1",
    "right_tactile_0",
    "right_tactile_1",
)
# Must match the order InjectTactileReference is configured with; that transform
# indexes by position, and the policy-side names are the *_ref keys it writes.
_DEFAULT_POLICY_NAMES = (
    "left_tactile_top",
    "left_tactile_bottom",
    "right_tactile_top",
    "right_tactile_bottom",
)


def _gripper_open_at_frame0(data_dir: pathlib.Path) -> dict[int, tuple[float, float]]:
    files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    cols = ["episode_index", "frame_index", "observation.state"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    first = df.sort_values(["episode_index", "frame_index"]).groupby("episode_index").first()
    return {int(ep): (float(s[18]), float(s[19])) for ep, s in first["observation.state"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Xense/bottle-sorting-0810")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("~/.cache/huggingface/lerobot"))
    parser.add_argument("--cameras", nargs="+", default=list(_DEFAULT_CAMERAS))
    parser.add_argument("--policy-names", nargs="+", default=list(_DEFAULT_POLICY_NAMES))
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.95,
        help="gripper position above which the hand counts as open at frame 0",
    )
    parser.add_argument("--allow-closed", action="store_true", help="write anyway if some references look deformed")
    args = parser.parse_args()

    if len(args.cameras) != len(args.policy_names):
        raise SystemExit("--cameras and --policy-names must have the same length")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    grippers = _gripper_open_at_frame0((args.root / args.repo_id).expanduser())
    closed = {ep: g for ep, g in grippers.items() if min(g) <= args.open_threshold}
    print(f"{len(grippers)} episodes; frame-0 grippers closed in {len(closed)}")
    if closed and not args.allow_closed:
        sample = dict(list(closed.items())[:5])
        raise SystemExit(
            f"Refusing to write: frame 0 is not a clean reference for {len(closed)} episodes, e.g. {sample}. "
            "Re-run with --allow-closed only if you know those episodes are unusable anyway."
        )

    dataset = LeRobotDataset(args.repo_id)
    starts = dataset.meta.episodes["dataset_from_index"]
    episodes = sorted(grippers)
    if episodes != list(range(len(episodes))):
        raise SystemExit("episode indices are not contiguous from 0; the store is indexed positionally")

    store = None
    for ep in episodes:
        item = dataset[int(starts[ep])]
        frames = []
        for cam in args.cameras:
            # Kept in LeRobot's CHW layout on purpose: the reference frames are
            # injected into data["images"] alongside the real cameras, and
            # BiFlexiv's _decode_bi_flexiv transposes every entry there. Storing
            # HWC would make it transpose a reference that is already HWC.
            chw = np.asarray(item[f"observation.images.{cam}"])
            frames.append((chw * 255).round().clip(0, 255).astype(np.uint8))
        stacked = np.stack(frames)
        if store is None:
            store = np.zeros((len(episodes), *stacked.shape), dtype=np.uint8)
            print(f"reference store shape {store.shape} ({store.nbytes / 1e6:.0f} MB)")
        store[ep] = stacked
        if ep % 20 == 0:
            print(f"  episode {ep}/{len(episodes)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, store)
    meta = {
        "repo_id": args.repo_id,
        "cameras": list(args.cameras),
        "policy_names": list(args.policy_names),
        "num_episodes": len(episodes),
        "frame": 0,
        "layout": "episode, camera, C, H, W (LeRobot CHW, uint8)",
        "shape": list(store.shape),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
