"""Can the water/no-water label be read off a channel other than tactile?

The bottle-sorting experiment only tests the tactile encoder if tactile is the
*only* place the label lives. This script labels every episode (far bin = heavy =
water, near bin = light) from the release-point TCP x, then fits a small linear
probe per channel and reports held-out accuracy against the majority-class
baseline. A channel that beats baseline is a shortcut the policy can take instead
of using touch.

Two splits are reported and they answer different questions:

* random  -- episodes shuffled. Optimistic: this dataset was recorded in blocks
             (long runs of one class), so neighbouring episodes share lighting,
             gel state and setup drift, and the probe can exploit that.
* grouped -- whole recording blocks held out. This is the number to trust. If a
             channel only separates under `random`, what it learned is session
             drift, not the bottle.

Probe: standardise -> PCA -> ridge regression onto +/-1 -> sign. Deliberately
weak. A weak probe finding signal means the signal is blatant; it not finding
any is weaker evidence, so treat the numbers as a lower bound on leakage.

Usage:
    python scripts/audit_bottle_sorting_leakage.py --repo-id Xense/bottle-sorting-0810
"""

from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np
import pandas as pd

_STATE_NAMES = (
    *[f"ltcp_{a}" for a in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")],
    *[f"rtcp_{a}" for a in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")],
    "l_grip",
    "r_grip",
)


def label_episodes(data_dir: pathlib.Path, *, far_x: float = 0.7) -> dict[int, dict]:
    """Per episode: class label, grasp frame, release frame, full state array.

    Label comes from where the right arm opens its gripper after the grasp: the
    far bin sits at x ~= 0.93, the near bin at x ~= 0.45.
    """
    files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    cols = ["episode_index", "frame_index", "observation.state"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_values(["episode_index", "frame_index"])

    out: dict[int, dict] = {}
    for ep, sub in df.groupby("episode_index"):
        state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
        closed = state[:, 19] < 0.6
        if not closed.any():
            continue
        grasp = int(np.argmax(closed))
        reopen = np.where(~closed[grasp:])[0]
        release = grasp + int(reopen[0]) if len(reopen) else len(state) - 1
        out[int(ep)] = {
            "state": state,
            "grasp": grasp,
            "release": release,
            "label": int(state[release, 9] > far_x),
        }
    return out


def blocks_of(labels: np.ndarray) -> np.ndarray:
    """Group id per episode: consecutive same-label episodes form one recording block."""
    group, gid = np.zeros(len(labels), dtype=int), 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            gid += 1
        group[i] = gid
    return group


def _pool(img: np.ndarray, n: int) -> np.ndarray:
    """Block-mean an (H, W) image down to (n, n)."""
    h, w = img.shape
    hh, ww = h // n * n, w // n * n
    return img[:hh, :ww].reshape(n, hh // n, n, ww // n).mean(axis=(1, 3))


def _probe(x_train, y_train, x_test, n_comp: int, ridge: float) -> np.ndarray:
    """Standardise -> PCA -> ridge onto +/-1. Returns predicted labels for x_test."""
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-8
    a, b = (x_train - mu) / sd, (x_test - mu) / sd
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    basis = vt[: min(n_comp, min(a.shape) - 1)].T
    a, b = a @ basis, b @ basis
    a = np.hstack([a, np.ones((len(a), 1))])
    b = np.hstack([b, np.ones((len(b), 1))])
    t = np.where(y_train == 1, 1.0, -1.0)
    w = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ t)
    return (b @ w > 0).astype(int)


def cross_val(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, n_comp=16, ridge=10.0, folds=5) -> float:
    """Accuracy with whole groups held out (groups=arange => plain random split)."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    order = rng.permutation(uniq)
    correct = total = 0
    for k in range(folds):
        held = set(order[k::folds].tolist())
        test = np.array([g in held for g in groups])
        if test.all() or not test.any() or len(np.unique(y[~test])) < 2:
            continue
        pred = _probe(x[~test], y[~test], x[test], n_comp, ridge)
        correct += int((pred == y[test]).sum())
        total += int(test.sum())
    return correct / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Xense/bottle-sorting-0810")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("~/.cache/huggingface/lerobot"))
    parser.add_argument("--offset", type=int, default=15, help="frames after grasp to sample images at")
    parser.add_argument("--downsample", type=int, default=24, help="probe image resolution")
    args = parser.parse_args()

    data_dir = (args.root / args.repo_id).expanduser()
    eps = label_episodes(data_dir)
    keys = sorted(eps)
    y = np.array([eps[e]["label"] for e in keys])
    groups = blocks_of(y)
    baseline = max(y.mean(), 1 - y.mean())
    print(f"{len(keys)} episodes | far/water={int(y.sum())} near/no-water={int((1 - y).sum())}")
    print(f"recording blocks: {len(np.unique(groups))} (max run {np.bincount(groups).max()})")
    print(f"majority-class baseline: {baseline:.3f}\n")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id)
    cams = [
        "observation.images.head",
        "observation.images.right_wrist",
        "observation.images.right_tactile_0",
        "observation.images.right_tactile_1",
    ]
    feats: dict[str, list] = {c: [] for c in cams}
    state_feats = []
    n = args.downsample
    for e in keys:
        rec = eps[e]
        frame = min(rec["grasp"] + args.offset, rec["release"], len(rec["state"]) - 1)
        item = ds[int(ds.meta.episodes["dataset_from_index"][e] + frame)]
        state_feats.append(rec["state"][frame])
        for cam in cams:
            img = np.asarray(item[cam], dtype=np.float32).mean(0)  # CHW -> grayscale HW
            feats[cam].append(_pool(img, n).ravel())

    channels: dict[str, np.ndarray] = {"state (20-D proprioception)": np.stack(state_feats)}
    for cam in cams:
        channels[cam.replace("observation.images.", "")] = np.stack(feats[cam])
    channels["tactile_0 + tactile_1"] = np.hstack([channels["right_tactile_0"], channels["right_tactile_1"]])

    print(f"linear probe at grasp+{args.offset} frames ({args.offset / 30:.2f}s)")
    print(f"  {'channel':30s} {'random split':>13s} {'grouped split':>14s}")
    for name, x in channels.items():
        rnd = cross_val(x, y, np.arange(len(y)))
        grp = cross_val(x, y, groups)
        flag = "  <-- leaks" if grp > baseline + 0.08 else ""
        print(f"  {name:30s} {rnd:13.3f} {grp:14.3f}{flag}")
    print(f"  {'(baseline)':30s} {baseline:13.3f} {baseline:14.3f}")


if __name__ == "__main__":
    main()
