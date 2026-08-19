#!/usr/bin/env python
"""Link test: drive the REAL ObsSubscriber against a running detection app,
WITHOUT the inference server or the robot.

This exercises the exact production class (examples.bi_flexiv_rizon4_rt.subscribe
.ObsSubscriber) by feeding it synthetic observations shaped like
env.get_observation() output: {"state": (20,), "images_raw": {"head": HWC}}.
It only needs xense_client importable (already present on the robot laptop) —
it does NOT import the robot SDK or connect to the policy server.

Usage
-----
1) On the video-playback laptop, start the app:
       python -m examples.dewu_video_switch.app
   and open  http://<play-ip>:8080  in a browser.

2) On the robot laptop (or any machine with xense_client installed):
       python -m examples.dewu_video_switch.sim_laptop --uri ws://<play-ip>:9100

You should see the browser flip scene_a ↔ scene_b (~2 s each) and the app console
print "[scene] → ...". That confirms ports, firewall, msgpack codec, and the
whole switch path end-to-end.
"""

import argparse
import time

import numpy as np

# Auto: real ObsSubscriber on the robot laptop, vendored ws sender on a dev /
# display machine that has no xense_client (see ws_sender.py).
from examples.dewu_video_switch.ws_sender import make_obs_subscriber_auto


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Synthetic link test for the obs-stream path — no inference server, no robot, no dataset. "
            "Sends frames that alternate between two synthetic states (low/high gripper + dark/bright "
            "head image), so the app flips scene_a <-> scene_b whichever detector it runs (gripper or "
            "stub). Confirms ports, firewall, msgpack codec, and the whole switch path end-to-end."
        ),
        epilog="Uses the production ObsSubscriber if xense_client is present, else the vendored ws sender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--uri",
        default="ws://127.0.0.1:9100",
        help="obs WebSocket URI of the video-playback laptop's app (its --obs-port, default 9100).",
    )
    ap.add_argument("--hz", type=float, default=30.0, help="Send rate in Hz (mimics the robot's runtime_hz).")
    ap.add_argument(
        "--hold-s", type=float, default=2.0, help="Seconds to hold each synthetic scene before alternating."
    )
    ap.add_argument("--seconds", type=float, default=0.0, help="Total run time in seconds; 0 = run until Ctrl+C.")
    args = ap.parse_args()

    sub = make_obs_subscriber_auto(args.uri)
    sub.on_episode_start()

    period = 1.0 / args.hz
    frames_per_scene = max(1, int(args.hold_s * args.hz))
    # Two synthetic states, alternating. The gripper value drives the default
    # GripperSceneDetector; the brightness drives the StubDetector — so this link
    # test flips scene_a <-> scene_b whichever detector app.py is running.
    #   index 0: low gripper  + dark  -> scene_a
    #   index 1: high gripper + bright -> scene_b
    grips = [0.2, 0.9]
    brights = [0.15, 0.85]

    print(
        f"Sending synthetic frames to {args.uri} at {args.hz:.0f} Hz (scene every {args.hold_s:.0f}s). Ctrl+C to stop."
    )
    t0 = time.time()
    i = 0
    try:
        while True:
            k = (i // frames_per_scene) % 2
            head = np.full((480, 640, 3), int(brights[k] * 255), dtype=np.uint8)
            state = np.zeros(20, dtype=np.float32)
            state[18] = state[19] = grips[k]  # left/right gripper.pos
            obs = {"state": state, "images_raw": {"head": head}}
            action = {"actions": np.zeros(20, dtype=np.float32)}
            sub.on_step(obs, action)
            i += 1
            if args.seconds and (time.time() - t0) > args.seconds:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        sub.on_episode_end()
        sub.close()
        print("done")


if __name__ == "__main__":
    main()
