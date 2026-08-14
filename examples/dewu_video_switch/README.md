# dewu video switch (Plan B: off-laptop detection + seamless video switching)

Runs the detection + the seamless video player on the **video-playback laptop**
(Windows / macOS / Linux), separate from the robot host laptop. The laptop only
streams a slim obs payload and spends zero extra cycles on detection or
rendering, and the **switch decision is computed on the playback laptop** and
pushed to its own browser over localhost.

## Quick start — the shoe-insole demo

The current demo drives four pre-recorded shoe clips off the **real detection
events** so it looks like live robot inspection. Scenes alternate by shoe group
(white shoe = group 1, red shoe = group 2): after each pick a `<group>-1` clip
plays, and once the blue insole is seen it switches to `<group>-2`. `standby`
holds clip `1-1` frozen on its first frame, and switching is an **instant
seamless cut** (no cross-dissolve):

```
standby → 1-1 → 1-2 → 2-1 → 2-2 → 1-1 → 1-2 → 2-1 → 2-2 → standby
         └ shoe1 white ┘ └ shoe2 red ┘ └ shoe3 white ┘ └ shoe4 red ┘
   (pick → x-1,  blue insole → x-2;  reset → standby)
```

### One-time setup

Use an env with `websockets msgpack numpy opencv-python lerobot` (the
`lerobot-xense` conda env has them). Symlink the four product clips into
`web/videos/` under their **scene names** (swap the target to change a clip —
`web/index.html` references `videos/<scene>.mp4`, never the product filename):

```bash
cd examples/dewu_video_switch/web/videos
ln -sf <path>/dewu-whiteshoes-1-1.mp4 1-1.mp4   # shoe 1/3 detecting
ln -sf <path>/dewu-whiteshoes-1-2.mp4 1-2.mp4   # shoe 1/3 after blue insole
ln -sf <path>/dewu-red-shoes-2-1.mp4  2-1.mp4   # shoe 2/4 detecting
ln -sf <path>/dewu-red-shoes-2-2.mp4  2-2.mp4   # shoe 2/4 after blue insole
```

### Run (from the repo root)

```bash
conda activate lerobot-xense

# 1) detection + web player + switch ws  (serves web/ on :8080, switch ws :9101, obs ws :9100)
python -m examples.dewu_video_switch.app \
    --detector shoe_sm \
    --detector-config examples/dewu_video_switch/shoe_sm.json

# 2) open the display in a browser:  http://<this-host>:8080

# 3a) DEMO from a recorded episode (mimics the robot; run on any host with lerobot):
python -m examples.dewu_video_switch.replay_lerobot \
    --repo-id Xense/newbalance_shoe_insole_retrieval_and_packing_0611 --episode 0

# 3b) LIVE on the robot laptop: add the obs subscriber to the usual launch instead of 3a
python -m examples.bi_flexiv_rizon4_rt.main --host <5090-ip> --port 8000 \
    --subscribe --subscribe_url ws://<play-ip>:9100
```

### Tune / debug the detector

`shoe_sm.json` holds the tuned config: per-arm pick boxes (shoes 1-2 = right arm,
3-4 = left arm), blue HSV/ROI, the **insole-flip pose gate**, `min_area_frac`
(blue area threshold), and `scene_groups`. To see exactly when each pick / blue /
reset fires with an annotated video + HUD (decodes only the head stream), use the
debug tool:

```bash
python -m examples.dewu_video_switch.replay_debug --episode 0 \
    --detector-config examples/dewu_video_switch/shoe_sm.json --out debug_ep0.mp4
# add --show for a live window; --out "" for a fast text-only event trace
```

The sections below describe the general three-machine framework and the earlier
2-scene gripper example; the demo above is the current shoe_sm configuration of
that same pipeline.

## System architecture — three machines

Three processes collaborate: ① the openpi inference server — a dedicated **5090
GPU server** running the VLA (the laptop does **not** infer; it is a thin policy
client), ② the robot host laptop (runs the control loop **and** streams the
detection-necessary data), and ③ the video-playback laptop (detection +
switching + display).

```
══ ① xense-openpi INFERENCE SERVER ───────────────────── RTX 5090 server · :8000 ══
   scripts/serve_policy.py policy:checkpoint --policy.config=<cfg> --policy.dir=<ckpt>
   π₀ / π₀.₅ VLA   ·   WebSocket policy server (msgpack · request → respond)
────────────────────────────────────────────────────────────────────────────────
        ▲ observation:  images 224² · state(20) · prompt
        │ WebSocket / msgpack
        │                                     ▼ actions: 20-D × horizon (chunk)
══ ② ROBOT HOST LAPTOP ── examples.bi_flexiv_rizon4_rt.main --subscribe ────
   Flexiv Rizon4 ×2  +  cameras(head, wrists)  ──▶  Environment.get_observation()
        ▲ apply_action @ 1000 Hz (inner)                  │ runtime loop @ 30 Hz
        │                                                 ▼
   WebsocketClientPolicy ──▶ (RTC) ActionChunkBroker ──▶ Runtime ──▶ subscribers[]
                                                                      │
   ObsSubscriber ◀───────────────────────────────────────────────┘
   (daemon thread · single-slot drop-oldest · swallows all errors ·
    NEVER blocks the 30 Hz control loop or the home-on-exit motion)
────────────────────────────────────────────────────────────────────────────────
        │ ws push:  { step, state(20), images:{ head: HWC uint8 }, action(20) }
        │ WebSocket / msgpack  ·  one-way  ·  non-blocking   ──────────▶  :9100
        ▼
══ ③ VIDEO-PLAYBACK LAPTOP ── examples.dewu_video_switch.app ──────────────────
   deps:  websockets + msgpack + numpy        (no openpi · no xense_client)

   :9100 obs ws  ──▶  LatestFrame  ──▶  detector (thread pool)
                                        └─ GripperSceneDetector — scene id / frame
                                                 │
                                        SceneController (--confirm-frames, --min-dwell-s)
                                                 │  emits only on a committed switch
                       ┌─────────────────────────┴─────────────────────────┐
                 :9101 switch ws                                      :8080 http (web/)
                 {"scene":"scene_a"|"scene_b"}                        player + video clips
                       └───────────────────▶  BROWSER  ◀───────────────────┘  (localhost)
                 all clips preloaded · muted · looping · switch = opacity cross-dissolve
                 scene_a ⇄ scene_b  →  web/videos/scene_{a,b}.mp4 → white_shoe_{0,1}.mp4
────────────────────────────────────────────────────────────────────────────────
   ▲ The switch decision (detector + SceneController) runs ON ③, the playback
     laptop — which also serves the player and the browser. Detection +
     switching + display are colocated there; that is the deployment target.
```

### Communication framework

| Link | Transport | Endpoint | Direction | Payload | Notes |
|------|-----------|----------|-----------|---------|-------|
| **A** ②→① | WebSocket / msgpack | `<5090-ip>:8000` | request → response | obs `{images 224², state(20), prompt}` → action chunk `(20-D × horizon)` | inference runs on the 5090 server, not the laptop; RTC overlaps inference with execution |
| **B** ②→③ | WebSocket / msgpack | `ws://<play-ip>:9100` | one-way push | `{step, state(20), images:{head HWC u8}}` (+`action` opt-in) | rate = `subscribe_hz` (≤ runtime_hz); zero-copy, single-slot drop-oldest, non-blocking |
| **C** ③→browser | WebSocket / JSON | `ws://<play-ip>:9101` | one-way push | `{"scene":"scene_a"\|"scene_b"}` | only on committed switch; current scene sent on connect |
| **D** ③→browser | HTTP | `http://<play-ip>:8080` | request → response | `web/` player + `scene_{a,b}.mp4` | static; clips fetched once, then play locally |

Inside ③ (in-process): obs ws → `LatestFrame` → `detector` (thread pool) →
`SceneController` → switch broadcast. The robot ↔ laptop link inside ② is the
robot SDK (1000 Hz inner control) + cameras, not a network channel.

The obs subscriber (`ObsSubscriber`) lives in the laptop's **control** process as
a `Subscriber` (`runtime.Runtime` calls it after each step) — the laptop is a
thin policy client; inference itself is on the 5090 server. Per step it does
~nothing: cheap rate gates, then stash *references* to the head frame + state
(no copy, no serialization) into a single-slot queue. A daemon thread owns the
socket and does all msgpack + I/O off-loop, swallowing every error — so a dead
playback laptop or a slow link can **never** disturb the robot's 30 Hz control
loop or its home-on-exit motion.

**Startup handshake.** The one exception to "never blocks" is *before* the control
loop starts: with `--subscribe`, `ObsSubscriber` waits at construction until ③'s
obs server is up and greets it (③ sends a `dewu_obs_hello` on connect), then proceeds
— exactly like the VLA policy client (`WebsocketClientPolicy._wait_for_server`) waits
for the inference server before running. So a launch with `--subscribe` will **not**
run inference while the screen PC is unreachable; it retries forever (or aborts after
`--subscribe_handshake_timeout` seconds, if set). Once running, the swallow-all-errors,
never-block contract above still holds.

> **Local testing (where this repo is now):** all three roles collapse onto one
> box. `replay_offline.py` runs the detector over a recorded episode with no ws /
> browser; `replay_lerobot.py` and `sim_laptop.py` feed the live app and — with no
> `xense_client` present — transparently fall back to the vendored ws sender
> (`ws_sender.py`), so ② needs only ③'s deps. See the sections below.

## Where each machine's code lives

### ① RTX 5090 inference server — runs the openpi VLA policy server

The only machine that touches the GPU; ② and ③ never run the model.

**Run:**

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=<cfg> --policy.dir=<ckpt>     # WebSocket policy server on :8000
```

| Path | Role |
|------|------|
| `scripts/serve_policy.py` | Entry point. Serves a checkpoint over WebSocket: `policy:checkpoint --policy.config=<cfg> --policy.dir=<ckpt>` |
| `src/openpi/serving/websocket_policy_server.py` | WebSocket server (`:8000`): receives an observation, returns an action chunk (msgpack) |
| `src/openpi/models/` (+ `models_pytorch/`) | π₀ / π₀.₅ model definitions (the VLA itself) |
| `src/openpi/policies/bi_flexiv_policy.py` | Input/output transforms for this bi-arm robot (robot obs ↔ model format) |
| `src/openpi/training/config.py` + `configs/` | The named `TrainConfig` that defines the model + norm stats the server loads |

### ② robot host laptop — real-time control loop + streams detection data

A thin policy client to ①, and a one-way producer to ③. Runs the openpi
`examples/bi_flexiv_rizon4_rt/*` against the installed `xense_client` SDK.

**Run** (minimal — full flag set in the **Run** section below):

```bash
python -m examples.bi_flexiv_rizon4_rt.main \
    --host <5090-ip> --port 8000 \
    --subscribe --subscribe_url ws://<play-ip>:9100
```

| Path | Role |
|------|------|
| `examples/bi_flexiv_rizon4_rt/main.py` | Entry. Wires env + policy client + runtime + subscribers; `--subscribe` enables obs streaming |
| `examples/bi_flexiv_rizon4_rt/env.py` (`BiFlexivRizon4RTEnvironment`) | Talks to the dual Flexiv Rizon4 + cameras; `get_observation()` / `apply_action()` |
| **`examples/bi_flexiv_rizon4_rt/subscribe.py` (`ObsSubscriber`)** | **The detection-data sender**: head stream + state → ③, configurable `subscribe_hz`, zero-copy, never blocks control |
| `examples/bi_flexiv_rizon4_rt/recorder.py` | Optional LeRobot dataset recorder (subscriber) |
| `examples/bi_flexiv_rizon4_rt/intervention.py` | Optional Pico4 human-in-the-loop teleop |
| `xense_client` (installed SDK, not in this repo) | `runtime.Runtime` (loop that calls subscribers), `WebsocketClientPolicy` (→ ①), `ActionChunkBroker` / `rtc_action_chunk_broker` (chunking + RTC), `msgpack_numpy` |

### ③ video-playback laptop — detection + switching + the player

Everything here is in `examples/dewu_video_switch/`. Needs only
`websockets + msgpack + numpy` (no openpi, no xense_client).

**Run:**

```bash
python -m examples.dewu_video_switch.app     # then open http://<play-ip>:8080
```

**Production — must run on ③:**

| Path | Role |
|------|------|
| `app.py` | The app: obs ws `:9100` (← ②) → detector loop → switch ws `:9101` (→ browser) + http `:8080` (player) |
| `detector.py` (`GripperSceneDetector`) | Maps a frame → scene id from the robot grasp signal (self-calibrating) |
| `controller.py` (`SceneController`) | Debounce / hysteresis; emits only a *committed* switch |
| `msgpack_numpy.py` | Vendored codec to decode ②'s frames (no xense_client) |
| `web/index.html` | The browser player (clips preloaded, switch = opacity cross-dissolve) |
| `web/videos/scene_{a,b}.mp4` | The clips (local symlinks → `white_shoe_{0,1}.mp4`) |
| `requirements.txt` | The complete dependency list for ③ |

**Dev / test only — run on any box, not needed in production:**

| Path | Role |
|------|------|
| `replay_offline.py` | Run the detector over a recorded episode, print the switch timeline (needs `lerobot`+`numpy`) |
| `replay_lerobot.py` | Stream a dataset episode into the live app (integration test) |
| `sim_laptop.py` | Synthetic link test (no robot, no dataset) |
| `ws_sender.py` | Vendored ws sender the two replay scripts fall back to when `xense_client` is absent |

The browser runs on ③ too (localhost), so detection + switching + display are all
colocated on the playback laptop.

## Run

**Video-playback laptop** (needs only `pip install -r requirements.txt` — no openpi):

```bash
python -m examples.dewu_video_switch.app        # from the openpi repo root
# or copy this folder onto the machine and run:  python app.py
```

Open `http://<play-ip>:8080` on the display. All clips play muted &
looping underneath; a switch is a pure opacity crossfade — no load, no seek, so
it's imperceptible. Cross-platform because it's just a browser (Chrome/Safari).

**Robot laptop** — add the subscribe flags to the usual launch. `--subscribe`
blocks at startup until this obs server is up (handshake), so start ③ first:

```bash
python -m examples.bi_flexiv_rizon4_rt.main \
    --robot_recipe forward-05 \
    --host 192.168.142.158 --port 8000 \
    --inner_control_hz 1000 \
    --interpolate_cmds --runtime_hz 30 --rtc_enabled \
    --subscribe \
    --subscribe_url ws://<play-ip>:9100
```

The obs subscriber ships only the **two channels the detector needs — head camera
stream + robot state** — taken *by reference* from the laptop's observation
buffer (zero-copy; all msgpack + socket work runs on a daemon thread, so the
30 Hz control loop pays ~zero). Useful flags:

- `--subscribe_hz 10` — cap the stream to 10 frames/s (raw head 640×480 @30 Hz
  ≈ 27 MB/s, so throttle for bandwidth). `0` (default) = stream every step.
- `--subscribe_cameras head` — which raw cameras (default: head only).
- `--subscribe_state` — stream the 20-D robot state (on by default; the gripper detector needs it).
- `--subscribe_action` — also send the model action (debug overlay; off by default).
- `--subscribe_stride N` — integer every-Nth-step subsample (alternative to `--subscribe_hz`).
- `--subscribe_handshake_timeout S` — startup handshake gives up and aborts after `S`
  seconds if ③ is unreachable. `0` (default) = wait forever, retrying (like the VLA client).

## Develop the detector on a LeRobot dataset (before the robot exists)

You don't need the robot client, the inference server, or even `xense_client` to
develop the detection algorithm. Two replay paths:

**Offline (fast inner loop)** — run the detector + `SceneController` straight over
an episode and print the scene-switch timeline. No websocket, no browser; needs
only `lerobot` + `numpy`:

```bash
python -m examples.dewu_video_switch.replay_offline --episode 0
python -m examples.dewu_video_switch.replay_offline --episode 0 --min-dwell-s 2.5
```

Example (8-episode `Xense/pack_6_cosmetic_bottles_into_carton`, episode 0):

```
  frame     time  scene
    287    9.57s  → scene_a
    679   22.63s  → scene_b
    ...
committed switches: 6   dwell min=11.1s mean=12.2s max=13.1s
```

**Live (integration)** — stream an episode through the actual switch path and
watch the player flip:

```bash
# 1) start the app (loads detector.py; default detector = gripper)
python -m examples.dewu_video_switch.app
# open http://localhost:8080

# 2) stream a dataset episode into it
python -m examples.dewu_video_switch.replay_lerobot --repo-id Xense/pack_6_cosmetic_bottles_into_carton --episode 0
```

`replay_lerobot.py` pulls each frame's `observation.state` (+ `observation.images.head`)
and streams it at the dataset fps. On the robot laptop it uses the production
`ObsSubscriber`; on a machine without `xense_client` it transparently falls
back to the vendored ws sender (`ws_sender.py`), so both replay scripts and
`sim_laptop.py` run locally with just the detection-app deps. Its default repo is
the (not-yet-public) shoe-insole dataset — pass `--repo-id` to use one you have.

## The shipped detector

`detector.py` ships `GripperSceneDetector` as the default (`--detector gripper`).
It switches scene on the robot's **grasp state**: it reads one gripper position
from the 20-D state (`[18]` left, `[19]` right) and emits `scene_a` while it sits
in its open plateau, `scene_b` while closed — so the screen flips exactly when the
robot grabs / releases. The open/closed split self-calibrates (running min/max +
hysteresis), so it works across robots/datasets without hand-tuned thresholds.

- Which physical state shows which clip is a product choice: flip `scene_low` /
  `scene_high` (or pick the other gripper via `grip_index`) in `make_detector()`.
- `SceneController` (`--confirm-frames`, `--min-dwell-s`) debounces on top; raise
  `--min-dwell-s` (e.g. 2.5) to suppress brief flicker on a marketing display.
- A heavier model (e.g. a keypoint net over the head image) plugs in the same
  way: subclass `Detector`, implement `detect(frame) -> scene_id | None`, register
  in `make_detector()`. `StubDetector` (brightness) is kept only for the synthetic
  `sim_laptop` link test.

## Videos

`web/videos/scene_a.mp4` / `scene_b.mp4` — for local testing these are symlinked
to the repo's `white_shoe_0.mp4` / `white_shoe_1.mp4`. On the real playback
machine, drop product clips under the same `scene_a.mp4` / `scene_b.mp4` names.
Binaries and symlinks here are git-ignored.
