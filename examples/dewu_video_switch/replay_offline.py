#!/usr/bin/env python
"""Develop the detector on a recorded LeRobot episode — OFFLINE, no transport.

This is the fast inner loop for the detection algorithm. It feeds each frame of a
recorded episode straight through the production detector + SceneController and
prints the resulting scene-switch timeline. No websocket, no browser, no
xense_client, no inference server — only `lerobot` + `numpy`. Edit detector.py,
rerun, read the timeline.

The live/browser path is the *integration* test (replay_lerobot.py → app.py →
browser). Use this one to decide *what* the detector should do; use that one to
confirm the whole switch path renders.

Usage
-----
    # default repo (override with --repo-id); state-only is plenty for the
    # gripper detector, so we skip video decode unless --with-images is given.
    python -m examples.dewu_video_switch.replay_offline --episode 0
    python -m examples.dewu_video_switch.replay_offline --repo-id Xense/pack_6_cosmetic_bottles_into_carton --episode 0
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from examples.dewu_video_switch import detector as _detector
    from examples.dewu_video_switch.controller import SceneController
except ImportError:  # standalone copy on the dev machine
    from controller import SceneController  # type: ignore
    import detector as _detector  # type: ignore

DEFAULT_REPO = "Xense/pack_6_cosmetic_bottles_into_carton"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "OFFLINE detector dev loop: run detector + SceneController over one recorded LeRobot "
            "episode and print the resulting scene-switch timeline. No websocket, no browser, no "
            "robot — needs only lerobot + numpy. Use this to design the detector; use replay_lerobot "
            "for the live integration test."
        ),
        epilog=(
            "Prints one line per committed switch (frame, time, scene) plus a summary: switch count, "
            "dwell-time stats, and the raw per-frame proposal mix — so you can judge how lively/stable "
            "the signal is before wiring it to the live player."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help="LeRobot dataset repo id to replay (must expose observation.state; head image only if --with-images).",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Local dataset root. Default: the HuggingFace cache (~/.cache/huggingface/lerobot).",
    )
    ap.add_argument("--episode", type=int, default=0, help="Episode index within the dataset to replay.")
    ap.add_argument(
        "--detector",
        default="gripper",
        help="Detector to evaluate: 'gripper' (real grasp-state) or 'stub' (brightness). See detector.make_detector.",
    )
    ap.add_argument(
        "--camera",
        default="head",
        help="Camera key suffix, i.e. observation.images.<camera>. Only used with --with-images.",
    )
    ap.add_argument(
        "--with-images",
        action="store_true",
        help="Also decode and feed the head image to the detector (slower). Off by default since the "
        "gripper detector only needs the state.",
    )
    ap.add_argument(
        "--confirm-frames",
        type=int,
        default=5,
        help="SceneController: consecutive frames a proposal must repeat before it commits (mirrors app.py).",
    )
    ap.add_argument(
        "--min-dwell-s",
        type=float,
        default=1.0,
        help="SceneController: minimum hold time (s) before another switch is allowed (mirrors app.py).",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="Cap the number of frames replayed; 0 = the whole episode."
    )
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading {args.repo_id} ...")
    ds = LeRobotDataset(args.repo_id, root=args.root)
    fps = float(ds.fps)
    img_key = f"observation.images.{args.camera}"
    if not (0 <= args.episode < ds.num_episodes):
        raise SystemExit(f"--episode {args.episode} out of range [0,{ds.num_episodes})")
    ep_meta = ds.meta.episodes[args.episode]
    start = int(ep_meta["dataset_from_index"])
    end = int(ep_meta["dataset_to_index"])
    if args.max_frames:
        end = min(end, start + args.max_frames)
    n = end - start

    detector = _detector.make_detector(args.detector)
    controller = SceneController(confirm_frames=args.confirm_frames, min_dwell_s=args.min_dwell_s)

    print(
        f"Episode {args.episode}: {n} frames @ {fps:.0f} Hz ({n / fps:.1f}s) — "
        f"detector={args.detector}, confirm_frames={args.confirm_frames}, min_dwell_s={args.min_dwell_s}\n"
        f"{'frame':>7} {'time':>8}  scene"
    )

    switches = 0
    dwell_frames: list[int] = []
    last_switch_frame = start
    proposed_counts: dict[str, int] = {}

    for i in range(start, end):
        item = ds[i]
        frame: dict = {"step": i - start}
        if "observation.state" in item:
            frame["state"] = item["observation.state"].numpy().astype(np.float32)
        if args.with_images and img_key in ds.features:
            t = item[img_key]  # (3,H,W) float in [0,1]
            frame["images"] = {
                args.camera: t.permute(1, 2, 0).clamp(0, 1).mul(255).round().to("cpu").numpy().astype(np.uint8)
            }

        proposed = detector.detect(frame)
        if proposed is not None:
            proposed_counts[proposed] = proposed_counts.get(proposed, 0) + 1
        committed = controller.update(proposed)
        if committed is not None:
            f = i - start
            print(f"{f:>7} {f / fps:>7.2f}s  → {committed}")
            switches += 1
            dwell_frames.append(i - last_switch_frame)
            last_switch_frame = i

    # Summary: how lively / stable is this signal?
    print("\n--- summary ---")
    print(f"committed switches: {switches}")
    if switches:
        dw = np.array(dwell_frames[1:] or dwell_frames) / fps
        print(f"dwell between switches: min={dw.min():.1f}s mean={dw.mean():.1f}s max={dw.max():.1f}s")
    if proposed_counts:
        total = sum(proposed_counts.values())
        dist = ", ".join(f"{k}={v / total:.0%}" for k, v in sorted(proposed_counts.items()))
        print(f"raw per-frame proposal mix: {dist}")
    print("final scene:", controller.current)


if __name__ == "__main__":
    main()
