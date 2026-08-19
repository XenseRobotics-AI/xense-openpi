#!/usr/bin/env python
"""Visual + event-trace DEBUG for the shoe_sm detector on a recorded episode.

This is the "what fires when, and why" tool. It replays one LeRobot episode
through the production `ShoeStateMachineDetector` (the same code path the live app
runs) and gives you two synchronized views of every detection point:

  1. EVENT TRACE (stdout) — one line each time a detection point fires, with the
     signal that caused it:
        [pick ] f=  287 t=  9.57s  state 0->1  right_tcp=(0.62,-0.04,-0.11) in_box=Y grasp=closed
        [blue ] f=  512 t= 17.07s  state 1     area=4.2% >= 2.0%  -> next
        [reset] f= 1403 t= 46.77s  state 4->0  (both TCPs home)
     plus the committed scene switches the SceneController emits to the browser.

  2. ANNOTATED VIDEO (--out debug_ep<N>.mp4, and/or live --show window) — the head
     camera with the blue HSV mask tinted in, the ROI box, and a HUD showing the
     current state / scene / pick phase / grasp / blue area%, with a banner that
     flashes on each pick / blue / reset.

Unlike replay_offline.py this DOES decode images (needed for the blue point and
the overlay) and DOES accept --detector-config, so it exercises the full shoe_sm
exactly as configured. No websocket, no browser, no robot — needs lerobot + numpy
+ opencv.

Usage
-----
    python -m examples.dewu_video_switch.replay_debug --episode 0
    python -m examples.dewu_video_switch.replay_debug --episode 0 \
        --detector-config examples/dewu_video_switch/shoe_sm.json --out debug_ep0.mp4
    python -m examples.dewu_video_switch.replay_debug --episode 0 --show   # live window
"""

from __future__ import annotations

import argparse
import contextlib

import numpy as np

try:
    from examples.dewu_video_switch.controller import SceneController
    from examples.dewu_video_switch.shoe_state_machine import ShoeStateMachineDetector
except ImportError:  # standalone copy on the dev machine
    from controller import SceneController  # type: ignore
    from shoe_state_machine import ShoeStateMachineDetector  # type: ignore

DEFAULT_REPO = "Xense/newbalance_shoe_insole_retrieval_and_packing_0611"

# HUD colors (BGR).
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_EVENT_COLOR = {"pick": (0, 200, 0), "blue": (255, 120, 0), "reset": (0, 0, 220)}


def _to_hwc_rgb_uint8(t) -> np.ndarray:
    """LeRobot image item -> HWC uint8 RGB. Handles CHW float[0,1] and HWC uint8."""
    arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        m = float(arr.max()) if arr.size else 1.0
        arr = arr * 255.0 if m <= 1.0 + 1e-6 else arr
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(arr)


def _text(cv2, bgr, s, org, scale=0.6, color=_WHITE, thick=1):
    """White text with a black outline so it stays readable over any background."""
    cv2.putText(bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_hud(cv2, bgr, *, frame_i, t, last, committed, banner, blue_area=None, events=None) -> None:
    """Overlay live status (top-left), a scrolling event log (bottom-left), and an
    event banner. Drawn in place on bgr.

    blue_area is the LIVE mask coverage from BlueInsoleDetector.annotate() this frame
    (computed every frame), so the HUD always reflects what the vision sees — unlike
    last['last_blue_area'], which the state machine stops updating once blue has fired
    for the current shoe.
    """
    h, w = bgr.shape[:2]
    state = last.get("state", "?")
    n = last.get("n_shoes", "?")
    scene = last.get("scene", "?")
    pick = last.get("pick") or {}
    area = blue_area if blue_area is not None else last.get("last_blue_area")
    area_s = f"{area * 100:.1f}%" if isinstance(area, (int, float)) else "--"

    # --- top-left: live status ---
    _text(cv2, bgr, f"f={frame_i:>5}  t={t:6.1f}s", (8, 22))
    _text(cv2, bgr, f"state {state}/{n}   scene={scene}", (8, 46))
    if pick:  # a pick is still pending (state < n_shoes): show the arm we're waiting on
        arm = pick.get("arm", "?")
        grasp = "closed" if pick.get("grasp_closed") else "open"
        in_box = "Y" if pick.get("in_box") else "n"
        col = (0, 220, 0) if pick.get("in_box") else _WHITE
        _text(cv2, bgr, f"next pick: arm={arm} in_box={in_box} grasp={grasp}", (8, 70), color=col)
    else:
        _text(cv2, bgr, "next pick: (all shoes picked)", (8, 70))
    blue_col = (0, 200, 255) if isinstance(area, (int, float)) and area >= 0.05 else _WHITE
    _text(cv2, bgr, f"blue area: {area_s}", (8, 94), color=blue_col)

    flip = last.get("flip") or {}
    fd = flip.get("delta")
    fd_s = f"{fd:.2f}" if isinstance(fd, (int, float)) else "--"
    fready = bool(flip.get("ready"))
    _text(
        cv2,
        bgr,
        f"flip gate: d={fd_s} {'READY' if fready else 'wait'}",
        (8, 118),
        color=(0, 220, 0) if fready else _WHITE,
    )

    # --- bottom-left: accumulated event log (most recent last) ---
    if events:
        shown = events[-12:]
        y0 = h - 14 - 18 * (len(shown) - 1)
        _text(cv2, bgr, "EVENTS", (8, y0 - 22), scale=0.5, color=(0, 255, 255))
        for i, ln in enumerate(shown):
            _text(cv2, bgr, ln, (8, y0 + 18 * i), scale=0.5)

    # --- center banner: flashes on each fired event ---
    if banner is not None:
        text, color, _ = banner
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        x0 = (w - tw) // 2
        cv2.rectangle(bgr, (x0 - 12, 8), (x0 + tw + 12, 8 + th + 18), color, -1)
        cv2.putText(bgr, text, (x0, 8 + th + 6), cv2.FONT_HERSHEY_DUPLEX, 0.9, _WHITE, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Visual + event-trace debug for shoe_sm over one recorded LeRobot episode: prints when "
            "each detection point (pick/blue/reset) fires and why, and renders an annotated video "
            "(head image + blue mask + ROI + state/scene HUD). Needs lerobot + numpy + opencv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--repo-id", default=DEFAULT_REPO, help="LeRobot dataset repo id to replay.")
    ap.add_argument("--root", default=None, help="Local dataset root. Default: the HuggingFace cache.")
    ap.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    ap.add_argument("--camera", default="head", help="Camera key, i.e. observation.images.<camera>.")
    ap.add_argument(
        "--detector-config",
        default=None,
        help="Path to a shoe_sm JSON config (bbox, blue HSV, etc.). Omit to use placeholder defaults.",
    )
    ap.add_argument("--confirm-frames", type=int, default=5, help="SceneController confirm frames.")
    ap.add_argument("--min-dwell-s", type=float, default=1.0, help="SceneController min dwell seconds.")
    ap.add_argument("--max-frames", type=int, default=0, help="Cap frames replayed; 0 = whole episode.")
    ap.add_argument("--out", default=None, help="Annotated mp4 path. Default debug_ep<N>.mp4. '' disables.")
    ap.add_argument("--show", action="store_true", help="Also show a live cv2 window (needs a display).")
    ap.add_argument("--fps", type=float, default=0.0, help="Output/window fps. 0 = dataset fps.")
    ap.add_argument("--banner-frames", type=int, default=12, help="How many frames an event banner holds.")
    args = ap.parse_args()

    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # episodes=[ep] makes LeRobotDataset check/sync ONLY this episode's files instead
    # of the whole dataset (which can be many GB and still downloading). The dataset
    # is then indexed 0..len-1 over just this episode's frames, in order.
    print(f"Loading {args.repo_id} episode {args.episode} ...")
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    fps = args.fps or ds.fps
    # Tolerate minor per-frame timestamp sync drift in the recorded videos: without
    # this, one frame a fraction of a frame-period off raises FrameTimestampError from
    # torchcodec and aborts the whole run (lerobot's default tolerance is 1e-4 s).
    # Allow up to one frame period; genuinely undecodable frames are still skipped below.
    with contextlib.suppress(Exception):
        ds.tolerance_s = max(ds.tolerance_s, 1.0 / fps)
    img_key = f"observation.images.{args.camera}"
    if img_key not in ds.features:
        raise SystemExit(f"camera key {img_key!r} not in dataset features: {sorted(ds.features)}")

    # Decode ONLY the head stream. ds[f] otherwise decodes every camera video (both
    # wrists + all tactiles) on every frame — none of which we use — which dominates
    # the runtime. Patch _query_videos to drop all but the head key before decode.
    try:
        _orig_query_videos = ds._query_videos

        def _query_head_only(query_ts, ep_idx, _orig=_orig_query_videos, _keep=img_key):
            return _orig({k: v for k, v in query_ts.items() if k == _keep}, ep_idx)

        ds._query_videos = _query_head_only
    except Exception as e:
        print(f"[warn] could not restrict decode to head ({type(e).__name__}); decoding all cameras")

    n = len(ds)
    if args.max_frames:
        n = min(n, args.max_frames)

    detector = ShoeStateMachineDetector(config_path=args.detector_config)
    blue = detector._ensure_blue()  # the BlueInsoleDetector (for the mask overlay)
    controller = SceneController(confirm_frames=args.confirm_frames, min_dwell_s=args.min_dwell_s)

    out_path = args.out
    if out_path is None:
        out_path = f"debug_ep{args.episode}.mp4"
    writer = None  # opened lazily once we know the frame size

    cfg_s = args.detector_config or "(placeholder defaults)"
    print(
        f"Episode {args.episode}: {n} frames @ {fps:.0f} Hz ({n / fps:.1f}s) — config={cfg_s}\n"
        f"{'EVENT':>6}  {'frame':>6} {'time':>8}  detail"
    )

    switches = 0
    n_events = {"pick": 0, "blue": 0, "reset": 0}
    banner = None  # (text, color, frames_left)
    show = args.show  # disabled on the fly if the cv2 build has no GUI (headless wheel)
    event_log: list[str] = []  # accumulated pick/blue/switch/reset lines, drawn on the frame
    skipped = 0  # frames whose video failed to decode (bad timestamps) — skipped, not fatal

    # Fast path for text-only runs (no video written/shown): pick + presentation-pose
    # detection use only the state (read straight from the parquet, every frame, no video
    # decode); the head image is needed only for the ~vision_stride blue check, so decode
    # it just on the detector's vision-check frames. ~vision_stride x fewer decodes.
    text_only = out_path == "" and not show
    vstride = max(1, detector.cfg.vision_stride)
    states_all = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float32) if text_only else None

    for f in range(n):
        t = f / fps
        head = None
        if text_only:
            state = states_all[f]
            if (f + 1) % vstride == 0:  # aligns with the detector's frame_i % vision_stride == 0
                try:
                    head = _to_hwc_rgb_uint8(ds[f][img_key])
                except Exception as ex:
                    skipped += 1
                    if skipped <= 5:
                        print(f"[warn] frame {f}: image decode failed ({type(ex).__name__}); skipping image")
        else:
            try:
                item = ds[f]
            except Exception as ex:
                skipped += 1
                if skipped <= 5:
                    print(f"[warn] frame {f}: video decode failed ({type(ex).__name__}); skipping")
                continue
            head = _to_hwc_rgb_uint8(item[img_key])
            state = item["observation.state"].numpy().astype(np.float32) if "observation.state" in item else None

        frame = {"step": f, "state": state, "images": ({args.camera: head} if head is not None else {})}
        scene = detector.detect(frame)
        last = detector.last
        committed = controller.update(scene)

        # ---- event trace ----
        ev = last.get("event")
        if ev:
            n_events[ev] = n_events.get(ev, 0) + 1
            if ev == "pick":
                pk = last.get("pick") or {}
                xyz = pk.get("xyz") or (float("nan"),) * 3
                arm = pk.get("arm", "?")
                detail = (
                    f"state {last['from_state']}->{last['to_state']}  arm={arm}  "
                    f"tcp=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) "
                    f"in_box={'Y' if pk.get('in_box') else 'n'} "
                    f"grasp={'closed' if pk.get('grasp_closed') else 'open'}"
                )
                log_line = f"{t:6.1f}s PICK {last['from_state']}->{last['to_state']} (arm {arm})"
            elif ev == "blue":
                bd = last.get("blue") or {}
                area = bd.get("area_frac") or 0.0
                detail = f"state {last['state']}  area={area * 100:.1f}%  -> next"
                log_line = f"{t:6.1f}s BLUE state{last['state']} ({area * 100:.0f}%)"
            else:  # reset
                detail = f"state {last['from_state']}->{last['to_state']}  (both TCPs home)"
                log_line = f"{t:6.1f}s RESET {last['from_state']}->0"
            print(f"[{ev:>5}]  {f:>6} {t:7.2f}s  {detail}")
            event_log.append(log_line)
            banner = (ev.upper(), _EVENT_COLOR[ev], args.banner_frames)

        if committed is not None:
            switches += 1
            print(f"[switch] {f:>6} {t:7.2f}s  -> {committed}")
            event_log.append(f"{t:6.1f}s switch #{switches} -> {committed}")

        # ---- annotated frame ----
        if out_path != "" or show:
            _present, _area, bgr = blue.annotate(head)
            _draw_hud(
                cv2,
                bgr,
                frame_i=f,
                t=t,
                last=last,
                committed=committed,
                banner=banner,
                blue_area=_area,
                events=event_log,
            )
            if banner is not None:
                banner = (banner[0], banner[1], banner[2] - 1)
                if banner[2] <= 0:
                    banner = None
            if out_path != "":
                if writer is None:
                    h, w = bgr.shape[:2]
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise SystemExit(f"could not open VideoWriter for {out_path!r}")
                writer.write(bgr)
            if show:
                try:
                    cv2.imshow("shoe_sm debug", bgr)
                    if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False
                    note = "" if out_path != "" else " Re-run with --out to still get a video."
                    print(
                        "[show] this OpenCV build has no GUI support (opencv-python-headless, as "
                        f"lerobot requires) — live --show window disabled; writing the mp4 instead.{note}"
                    )

    if writer is not None:
        writer.release()
    if show:
        with contextlib.suppress(cv2.error):
            cv2.destroyAllWindows()

    print("\n--- summary ---")
    print(f"events: pick={n_events['pick']} blue={n_events['blue']} reset={n_events['reset']}")
    print(f"committed switches: {switches}   final scene: {controller.current}")
    if skipped:
        print(f"skipped frames (undecodable video): {skipped}/{n}")
    if out_path != "" and writer is not None:
        print(f"annotated video: {out_path}")


if __name__ == "__main__":
    main()
