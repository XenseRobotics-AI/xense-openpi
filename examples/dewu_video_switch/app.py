#!/usr/bin/env python
"""Video-playback laptop app for the dewu seamless video-switch demo (Plan B).

Topology
--------
    robot laptop (examples.bi_flexiv_rizon4_rt.main --forward)
        │  ws push  {state, head image}  (msgpack)
        ▼
    THIS PROCESS  (the video-playback laptop)
        ├─ obs ws server   :9100   ← laptop connects here, pushes frames
        ├─ detector (thread pool)  → SceneController (debounce)
        ├─ switch ws server :9101  → browsers connect, receive {"scene": id}
        └─ static http      :8080  → serves web/ (the seamless player)

Why three listeners: the laptop is a ws *client* (so it needs no inbound port),
the browser is a ws *client*, and the player HTML is plain static files. Detection
runs in a thread-pool executor on the *latest* frame only (older frames are
dropped while busy) so a heavy keypoint model never stalls the switch broadcast.

Run:
    python -m examples.dewu_video_switch.app          # from the openpi repo root
    # or, copied standalone onto the video-playback laptop:
    python app.py
Then open  http://<this-machine-ip>:8080  on the display.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import http.server
import json
import pathlib
import sys
import threading
import time

import numpy as np

# Support both `python -m examples.dewu_video_switch.app` and standalone `python app.py`.
try:
    from examples.dewu_video_switch import detector as _detector
    from examples.dewu_video_switch import msgpack_numpy
    from examples.dewu_video_switch.controller import SceneController
except ImportError:  # standalone copy on the video-playback laptop
    from controller import SceneController  # type: ignore
    import detector as _detector  # type: ignore
    import msgpack_numpy  # type: ignore

import websockets

WEB_DIR = pathlib.Path(__file__).parent / "web"


class LatestFrame:
    """Single-slot frame holder: producer overwrites, consumer drops staleness."""

    def __init__(self) -> None:
        self._frame: dict | None = None
        self._event = asyncio.Event()

    def put(self, frame: dict) -> None:
        self._frame = frame
        self._event.set()

    async def get(self) -> dict:
        await self._event.wait()
        self._event.clear()
        assert self._frame is not None
        return self._frame


class App:
    def __init__(
        self,
        detector: _detector.Detector,
        controller: SceneController,
        debug_log: str | None = None,
        gate_print_every: int = 15,
        blue_frames_dir: str | None = None,
    ) -> None:
        self._detector = detector
        self._controller = controller
        self._latest = LatestFrame()
        self._frontends: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

        # ---- blue-frame dump (SEE what the vision counts as insole blue) ----
        # While a shoe is active, periodically save the head image with the blue mask
        # tinted in + the ROI box, so a blue FALSE positive (e.g. the arm's side during
        # the insole grab) is visible as a picture, not just an area number.
        self._blue_frames_dir = blue_frames_dir
        if blue_frames_dir:
            pathlib.Path(blue_frames_dir).mkdir(parents=True, exist_ok=True)

        # ---- debug logging (why did / didn't a scene switch fire) ----
        # Every frame's detector snapshot (state, gate distances, blue area) is
        # appended to `debug_log` as JSONL for offline analysis, and a throttled
        # one-liner is printed live while a shoe is active but its blue switch has
        # not yet fired — so a stuck gate is visible both live and after the fact.
        self._debug_fh = open(debug_log, "a", buffering=1) if debug_log else None  # noqa: SIM115
        self._debug_path = debug_log
        self._gate_print_every = max(1, gate_print_every)
        self._min_pose_dist: float | None = None   # closest pose-gate approach this shoe
        self._max_blue_area: float | None = None    # best blue area seen this shoe
        self._gate_state: int = 0                    # state we are tracking min/max for
        cfg = getattr(detector, "cfg", None)
        self._pose_tol = getattr(cfg, "present_pose_tol", None)
        self._flip_min = getattr(cfg, "flip_min_delta", None)
        self._blue_min = getattr(getattr(cfg, "blue", None), "min_area_frac", None)
        self._vision_confirm = getattr(cfg, "vision_confirm", None)
        self._save_stride = int(getattr(cfg, "vision_stride", 15) or 15)

    def _save_blue_frame(self, head, state: int, frame_i: int, event: str | None) -> None:
        """Save the head image with the blue mask + ROI drawn on it. The tinted pixels
        are exactly what BlueInsoleDetector counts as insole -> makes a false positive
        (arm side, box) obvious. Named so it sorts by shoe/frame and flags the fire."""
        if not self._blue_frames_dir:
            return
        try:
            import cv2  # noqa: PLC0415 — heavy, only when dumping frames

            bd = self._detector._ensure_blue()  # BlueInsoleDetector
            present, area, bgr = bd.annotate(head)
            tag = "FIRE_" if event == "blue" else ""
            a = int(round((area or 0) * 100))
            fn = f"{tag}s{state}_f{frame_i:05d}_a{a:02d}_{'Y' if present else 'n'}.jpg"
            cv2.imwrite(f"{self._blue_frames_dir}/{fn}", bgr)
        except Exception:  # noqa: BLE001 — diagnostics must never break the loop
            pass

    # ----- obs ingress (from robot laptop) -----

    async def _obs_handler(self, websocket) -> None:
        peer = getattr(websocket, "remote_address", "?")
        print(f"[obs] laptop connected: {peer}")
        self._debug("obs_connect", peer=str(peer))
        # Handshake: greet the just-connected robot client so its startup connection
        # check can confirm the detection app (not just any listener on this port) is
        # really up before it starts inference — mirrors the VLA policy server sending
        # metadata on connect. One-way from here on; the client only pushes obs.
        with contextlib.suppress(Exception):
            await websocket.send(
                msgpack_numpy.packb(
                    {"type": "dewu_obs_hello", "app": "dewu_video_switch", "detector": type(self._detector).__name__}
                )
            )
        try:
            async for message in websocket:
                if isinstance(message, str):
                    continue
                try:
                    frame = msgpack_numpy.unpackb(message)
                except Exception as e:
                    print(f"[obs] decode error: {e}")
                    self._debug("obs_decode_error", error=str(e))
                    continue
                self._latest.put(frame)
        except websockets.ConnectionClosed:
            pass
        finally:
            print(f"[obs] laptop disconnected: {peer}")
            self._debug("obs_disconnect", peer=str(peer))

    # ----- switch egress (to browsers) -----

    async def _switch_handler(self, websocket) -> None:
        self._frontends.add(websocket)
        print(f"[switch] browser connected ({len(self._frontends)} total)")
        # Send the current scene immediately so a late-joining display syncs up.
        if self._controller.current is not None:
            await self._safe_send(websocket, self._controller.current)
        try:
            await websocket.wait_closed()
        finally:
            self._frontends.discard(websocket)
            print(f"[switch] browser disconnected ({len(self._frontends)} total)")

    async def _broadcast(self, scene: str) -> None:
        if not self._frontends:
            return
        await asyncio.gather(*(self._safe_send(ws, scene) for ws in list(self._frontends)))

    @staticmethod
    async def _safe_send(ws, scene: str) -> None:
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"scene": scene}))

    # ----- debug logging -----

    def _debug(self, kind: str, **fields) -> None:
        """Append one JSONL record to the debug log (no-op if logging is off)."""
        if self._debug_fh is None:
            return
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        with contextlib.suppress(Exception):
            self._debug_fh.write(json.dumps(rec, default=float) + "\n")

    def _log_detection(self, snap: dict | None, committed: str | None, frame: dict | None = None) -> None:
        """Persist every detector snapshot and print a throttled live gate readout
        showing exactly which gate is holding the blue switch back."""
        if not snap:
            return
        # Raw LEFT-arm 9-D pose (state[0:9] = xyz + 6-D rot) — logged so the insole
        # presentation pose can be re-calibrated on THIS robot: read left9 at the frames
        # where the insole is actually shown to the head cam, average, -> present_poses.
        left9 = None
        st = (frame or {}).get("state")
        if isinstance(st, np.ndarray) and st.shape[-1] >= 9:
            left9 = [round(float(v), 4) for v in st[:9]]
        state = snap.get("state") or 0
        # Reset the per-shoe closest-approach trackers whenever the shoe changes.
        if state != self._gate_state:
            self._gate_state = state
            self._min_pose_dist = None
            self._max_blue_area = None
        pose = snap.get("pose") or {}
        blue = snap.get("blue") or {}
        flip = snap.get("flip") or {}
        dist = pose.get("dist")
        if isinstance(dist, (int, float)):
            self._min_pose_dist = dist if self._min_pose_dist is None else min(self._min_pose_dist, dist)
        area = blue.get("area_frac")
        if isinstance(area, (int, float)):
            self._max_blue_area = area if self._max_blue_area is None else max(self._max_blue_area, area)

        # File: log events always, plus every active-shoe frame (skip idle standby noise).
        event = snap.get("event")
        if self._debug_fh is not None and (event or state != 0 or blue.get("checked")):
            self._debug(
                "frame",
                frame=snap.get("frame"),
                state=state,
                scene=snap.get("scene"),
                event=event,
                committed=committed,
                blue_fired=snap.get("blue_fired"),
                pose=pose,
                blue=blue,
                flip=flip,
                left9=left9,
                min_pose_dist=self._min_pose_dist,
                max_blue_area=self._max_blue_area,
            )

        # Dump the annotated head periodically while a shoe is active (and always on the
        # blue fire) so a blue false positive can be eyeballed, not just read as a number.
        # NB: uses `frame` (the raw dict) here, before it is rebound to an int below.
        if self._blue_frames_dir and state != 0:
            fr = snap.get("frame") or 0
            if event == "blue" or fr % self._save_stride == 0:
                head = ((frame or {}).get("images") or {}).get("head")
                if isinstance(head, np.ndarray) and head.ndim == 3:
                    self._save_blue_frame(head, state, fr, event)

        # Console: a throttled gate line while a shoe is active but blue hasn't fired.
        if state and not snap.get("blue_fired"):
            frame = snap.get("frame") or 0
            if frame % self._gate_print_every == 0:
                parts = [f"[gate] s{state} f{frame}"]
                if isinstance(dist, (int, float)):
                    parts.append(
                        f"pose.dist={dist:.2f}/{self._pose_tol} "
                        f"min={self._min_pose_dist:.2f} ready={'T' if pose.get('ready') else 'F'}"
                    )
                elif isinstance(flip.get("delta"), (int, float)):
                    parts.append(
                        f"flip.delta={flip['delta']:.2f}/{self._flip_min} "
                        f"ready={'T' if flip.get('ready') else 'F'}"
                    )
                if isinstance(area, (int, float)):
                    parts.append(
                        f"blue.area={area:.3f}/{self._blue_min} "
                        f"max={self._max_blue_area:.3f} present={'T' if blue.get('present') else 'F'}"
                    )
                print(" ".join(parts))

    # ----- detection loop -----

    async def _detect_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            frame = await self._latest.get()
            # Run (possibly heavy) detection off the event loop.
            proposed = await loop.run_in_executor(None, self._detector.detect, frame)
            committed = self._controller.update(proposed)
            if committed is not None:
                print(f"[scene] → {committed}")
                await self._broadcast(committed)
            self._log_detection(getattr(self._detector, "last", None), committed, frame)

    # ----- orchestration -----

    async def run(self, obs_port: int, switch_port: int) -> None:
        self._loop = asyncio.get_running_loop()
        async with (
            websockets.serve(self._obs_handler, "0.0.0.0", obs_port, max_size=None, compression=None),
            websockets.serve(self._switch_handler, "0.0.0.0", switch_port, max_size=None),
        ):
            print(f"[ws] obs server   on :{obs_port}  (robot laptop connects here)")
            print(f"[ws] switch server on :{switch_port} (browsers connect here)")
            await self._detect_loop()


class _QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler that stays quiet about the disconnects a <video> tag causes.

    The player preloads several clips and routinely aborts / resets range requests
    as it buffers, which surfaces as BrokenPipeError / ConnectionResetError out of
    copyfile(). Those are normal and unrelated to scene switching, so we silence the
    per-request 2xx logging here and swallow the reset itself in _QuietHTTPServer.
    404s and other real errors still print via log_error()."""

    def log_message(self, fmt, *args):
        pass

    def log_error(self, fmt, *args):
        sys.stderr.write("[http] " + (fmt % args) + "\n")


class _QuietHTTPServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer that drops the traceback for browser-side disconnects."""

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return  # browser cancelled/reset a video range request — expected, ignore
        super().handle_error(request, client_address)


def _serve_static(http_port: int) -> None:
    handler = functools.partial(_QuietHTTPRequestHandler, directory=str(WEB_DIR))
    httpd = _QuietHTTPServer(("0.0.0.0", http_port), handler)
    print(f"[http] player at http://0.0.0.0:{http_port}  (serving {WEB_DIR})")
    httpd.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Detection + seamless video-switch app for the video-playback laptop (machine ③). "
            "Receives forwarded robot obs (head image + state) from the robot laptop, runs the "
            "detector + scene debounce, and pushes scene switches to the in-browser player."
        ),
        epilog=(
            "Three listeners start together: obs ws (robot laptop connects in), switch ws "
            "(browsers connect in), and static http (the player). Open http://<this-host>:<http-port> "
            "on the display. Feed it with examples.dewu_video_switch.replay_lerobot (real dataset) "
            "or .sim_laptop (synthetic link test)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--obs-port", type=int, default=9100,
        help="TCP port of the obs WebSocket server the robot laptop's ForwardSubscriber pushes frames into.",
    )
    p.add_argument(
        "--switch-port", type=int, default=9101,
        help="TCP port of the switch WebSocket server browsers connect to; it broadcasts {\"scene\": id} on each committed switch.",
    )
    p.add_argument(
        "--http-port", type=int, default=8080,
        help="TCP port of the static HTTP server for web/ (the seamless player HTML and the video clips).",
    )
    p.add_argument(
        "--detector", default="gripper",
        help="Which detector to run: 'shoe_sm' = per-shoe state machine (pose+gripper pick event + "
        "OpenCV blue insole); 'gripper' = simple grasp-state detector; 'stub' = brightness (link test "
        "only). See detector.make_detector.",
    )
    p.add_argument(
        "--detector-config", default=None,
        help="Path to a JSON config for the detector (shoe_sm only: bounding box, blue HSV, etc.). "
        "See shoe_sm.example.json. Omit to use placeholder defaults.",
    )
    p.add_argument(
        "--confirm-frames", type=int, default=5,
        help="Debounce: a proposed scene must repeat for this many consecutive frames before it may commit. "
        "Higher = steadier but slower to react.",
    )
    p.add_argument(
        "--min-dwell-s", type=float, default=1.0,
        help="Debounce: minimum seconds to hold the current scene before another switch is allowed. "
        "Raise to ~2.5 to suppress brief flicker on a marketing display.",
    )
    p.add_argument(
        "--debug-log", nargs="?", const="", default="",
        help="Path to a JSONL debug log capturing every detector snapshot (state, pose/flip gate "
        "distances, blue area) so a stuck scene switch can be diagnosed. Default: an auto-named "
        "dewu_debug_<timestamp>.jsonl in the cwd. Pass a path to override, or '' / 'none' to disable.",
    )
    p.add_argument(
        "--save-blue-frames", nargs="?", const="", default="",
        help="Directory to dump the head image (with the blue mask + ROI drawn on it) at every "
        "vision-check frame while a shoe is active, so blue FALSE positives can be inspected as "
        "pictures. Default: an auto-named dewu_blue_<timestamp>/ dir. '' / 'none' to disable.",
    )
    args = p.parse_args()

    if args.debug_log in ("none", "off", "-"):
        debug_log = None
    elif args.debug_log:
        debug_log = args.debug_log
    else:
        debug_log = time.strftime("dewu_debug_%Y%m%d_%H%M%S.jsonl")

    if args.save_blue_frames in ("none", "off", "-"):
        blue_frames_dir = None
    elif args.save_blue_frames:
        blue_frames_dir = args.save_blue_frames
    else:
        blue_frames_dir = time.strftime("dewu_blue_%Y%m%d_%H%M%S")

    detector = _detector.make_detector(args.detector, config_path=args.detector_config)
    controller = SceneController(confirm_frames=args.confirm_frames, min_dwell_s=args.min_dwell_s)
    app = App(detector, controller, debug_log=debug_log, blue_frames_dir=blue_frames_dir)
    if debug_log:
        print(f"[debug] logging detector snapshots to {debug_log}  (disable with --debug-log none)")
    if blue_frames_dir:
        print(f"[debug] saving annotated blue-check frames to {blue_frames_dir}/  (disable with --save-blue-frames none)")

    threading.Thread(target=_serve_static, args=(args.http_port,), daemon=True).start()
    try:
        asyncio.run(app.run(args.obs_port, args.switch_port))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
