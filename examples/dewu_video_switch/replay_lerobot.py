#!/usr/bin/env python
"""Stream a LeRobot dataset episode into the detection pipeline.

Purpose: develop and test the detection algorithm on REAL recorded data before
the robot client exists. This replays one episode's {state, head image} frames
through the exact production path — the real ObsSubscriber → the detection
app (:9100) → detector → SceneController → browser switch — at the dataset's
native frame rate.

It is the dataset-backed sibling of sim_laptop.py: same wire path, but the
frames come from `Xense/newbalance_shoe_insole_retrieval_and_packing_0611`
instead of synthetic brightness ramps.

Usage
-----
1) Start the detection app (picks up your detector.py):
       python -m examples.dewu_video_switch.app
   and open  http://localhost:8080
2) Replay an episode into it:
       python -m examples.dewu_video_switch.replay_lerobot --episode 0
       python -m examples.dewu_video_switch.replay_lerobot --episode 3 --loop --fps 30

Note: until a real detector is wired into detector.make_detector(), the shipped
StubDetector keys off image brightness, so switches on real footage will look
arbitrary — that is expected. The point here is to exercise the algorithm you
are developing against real frames.
"""

import argparse
import time

import numpy as np
import torch

# Auto: real ObsSubscriber on the robot laptop, vendored ws sender on a dev /
# display machine that has no xense_client (see ws_sender.py).
from examples.dewu_video_switch.ws_sender import make_obs_subscriber_auto

DEFAULT_REPO = "Xense/newbalance_shoe_insole_retrieval_and_packing_0611"


def _chw_float_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) float[0,1] tensor -> (H,W,3) uint8 ndarray (raw-camera format)."""
    return t.permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8).contiguous().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "LIVE integration test: stream one recorded LeRobot episode's {state, head image} into a "
            "running detection app (machine ③) over the real obs-stream path, at the dataset frame rate. "
            "On the robot laptop this uses the production ObsSubscriber; on a box without "
            "xense_client it transparently falls back to the vendored ws sender (ws_sender.py)."
        ),
        epilog="Start `python -m examples.dewu_video_switch.app` first, then run this against its --uri.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help="LeRobot dataset repo id to replay. The default shoe-insole set may not be public — pass one you have.",
    )
    ap.add_argument("--root", default=None, help="Local dataset root. Default: the HuggingFace cache.")
    ap.add_argument("--episode", type=int, default=0, help="Episode index within the dataset to stream.")
    ap.add_argument(
        "--uri",
        default="ws://127.0.0.1:9100",
        help="obs WebSocket URI of the video-playback laptop's app (its --obs-port, default 9100).",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Replay/send rate in Hz. 0 = use the dataset's native fps (mimics the real 30 Hz control loop).",
    )
    ap.add_argument(
        "--camera",
        default="head",
        help="Camera to stream, i.e. observation.images.<camera>. The detector uses the head image.",
    )
    ap.add_argument(
        "--no-state",
        action="store_true",
        help="Do NOT stream observation.state. The gripper detector needs the state, so leave this off for it.",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="Cap the number of frames streamed; 0 = the whole episode."
    )
    ap.add_argument(
        "--loop", action="store_true", help="Replay the episode repeatedly until Ctrl+C (good for a kiosk demo)."
    )
    args = ap.parse_args()

    # Imported lazily so `app.py` on a machine without lerobot is unaffected.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading {args.repo_id} ...")
    ds = LeRobotDataset(args.repo_id, root=args.root)
    fps = args.fps or ds.fps
    img_key = f"observation.images.{args.camera}"
    if img_key not in ds.features:
        raise SystemExit(f"{img_key!r} not in dataset features: {list(ds.features)}")

    if not (0 <= args.episode < ds.num_episodes):
        raise SystemExit(f"--episode {args.episode} out of range [0,{ds.num_episodes})")
    ep_meta = ds.meta.episodes[args.episode]
    start = int(ep_meta["dataset_from_index"])
    end = int(ep_meta["dataset_to_index"])
    if args.max_frames:
        end = min(end, start + args.max_frames)
    n = end - start
    print(f"Episode {args.episode}: frames [{start},{end}) = {n} frames, streaming to {args.uri} at {fps:.0f} Hz")

    sub = make_obs_subscriber_auto(args.uri, cameras=(args.camera,), send_state=not args.no_state)
    sub.on_episode_start()

    period = 1.0 / fps
    try:
        while True:
            t_next = time.monotonic()
            for i in range(start, end):
                item = ds[i]
                head = _chw_float_to_hwc_uint8(item[img_key])
                obs = {"images_raw": {args.camera: head}}
                if not args.no_state:
                    obs["state"] = item["observation.state"].numpy().astype(np.float32)
                # action isn't needed by the detector; send zeros for payload parity.
                sub.on_step(obs, {"actions": np.zeros(20, dtype=np.float32)})

                t_next += period
                sleep = t_next - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
            if not args.loop:
                break
            print("loop: restarting episode")
    except KeyboardInterrupt:
        pass
    finally:
        sub.on_episode_end()
        sub.close()
        print("done")


if __name__ == "__main__":
    main()
