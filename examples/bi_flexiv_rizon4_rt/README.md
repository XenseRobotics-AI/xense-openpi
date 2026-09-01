# BiFlexiv Rizon4 RT — Real Robot Inference

Dual-arm Flexiv Rizon4 inference client using the OpenPI policy server.

## Prerequisites

- `lerobot-xense` conda environment with `lerobot2mcap` installed
- Flexiv Rizon4 RT arms reachable over Ethernet
- OpenPI policy server running (see below)

## Network Setup

Robot FastDDS communication and policy server inference must run on **separate physical links** to avoid network contention (FastDDS 1 kHz commands are latency-sensitive and will stall if competing with large inference payloads).

| Interface | IP | Connects to |
|---|---|---|
| enp8s0 (motherboard Ethernet) | 192.168.142.216 (DHCP) | Router → Robot arms (FastDDS) |
| enx6c1ff7618da5 (USB-C adapter) | **10.142.1.2**/24 (static) | Direct cable → Policy server |

**Policy server** (GPU machine) sets its corresponding port to **10.142.1.1**/24.

### Configure static IPs (one-time per boot)

On this machine (robot client):
```bash
sudo ip addr add 10.142.1.2/24 dev enx6c1ff7618da5
```

On the policy server:
```bash
sudo ip addr add 10.142.1.1/24 dev <interface_name>
```

Verify connectivity:
```bash
ping 10.142.1.1
```

Then use `--args.host 10.142.1.1` when launching the robot client.

## Usage

### Terminal 1: Start OpenPI Policy Server

```bash
cd ~/openpi
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora \
    --policy.dir=<checkpoint_dir>
```

### Terminal 2: Run Robot Client

`--args.robot-recipe` is required — it names the physical bench.

```bash
cd ~/openpi
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-05 \
    --args.host 10.142.1.1 \
    --args.port 8000
```

#### Picking a bench

The bench's arm SNs, start/home poses, head camera and gripper block come from a
recipe under [`recipes/`](recipes/): `forward-01`, `forward-04`, `forward-05`,
`forward-06`, `diagonal-02`. A path works too, so a recipe from the lerobot-xense
tree can be used directly:

```bash
--args.robot-recipe ~/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-04.yaml
```

This replaced `--bi_mount_type <name>`, which indexed a `stations/` table inside
lerobot. That table was removed upstream (`3b964bc6`) — the config dataclass no
longer carries bench hardware at all, so the caller supplies it, and recipes are
the mechanism lerobot's own CLIs use. See [`recipes/README.md`](recipes/README.md)
for what belongs in a recipe versus on the command line, and for the warning
about keeping the two repos' copies of a bench in sync.

#### Common options

The recipe carries bench hardware; the CLI owns run tuning outright. Every
tuning flag has a concrete default, so it is **always** applied on top of the
decoded recipe — a tuning key written into a recipe (or already present in an
upstream lerobot teleop/record recipe you point at by path) loses to the flag.
The loader logs each key it overrides, so the swap is visible rather than
silent.

| Flag | Default | Description |
|---|---|---|
| `--args.robot-recipe` | — (required) | Bench recipe name under `recipes/`, or a path |
| `--args.host` | `localhost` | Policy server host |
| `--args.port` | `8000` | Policy server port |
| `--args.stiffness-ratio` | `0.2` | Cartesian stiffness (0–1) |
| `--args.inner-control-hz` | `1000` | How often each 1 kHz RT loop consumes a new Python command |
| `--args.interpolate-cmds` | `True` | Enable linear interpolation between consumed commands |
| `--args.runtime-hz` | `20.0` | Policy inference Hz |
| `--args.action-horizon` | `50` | Steps per action chunk |
| `--args.num-episodes` | `1` | Episodes to run |
| `--args.max-episode-steps` | `100000` | Max steps per episode |
| `--args.dry-run` | `False` | Print actions, do not execute |
| `--args.pico4-intervention` | `False` | Enable Pico4 VR human-in-the-loop intervention (see below) |
| `--args.pico4-pos-sensitivity` | `1.0` | Position sensitivity passed to `BiPico4Config` when intervention is on |
| `--args.pico4-ori-sensitivity` | `1.0` | Orientation sensitivity passed to `BiPico4Config` when intervention is on |

#### RTC (real-time correction) mode

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-05 \
    --args.host 10.142.1.1 --args.port 8000 \
    --args.rtc-enabled \
    --args.action-queue-size-to-get-new-actions 40 \
    --args.execution-horizon 50 \
    --args.blend-steps 3
```

#### Dry run (no robot motion)

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-05 \
    --args.host 10.142.1.1 --args.port 8000 \
    --args.dry-run
```

#### Human intervention via Pico4 VR controllers

Hold **both** Pico4 side (grip) buttons together to take over the robot from the
policy mid-episode; release either grip to hand control back. While intervention
is active, policy inference is paused (no WebSocket round-trip to the server),
and on release the `ActionChunkBroker` cache is cleared so the next step
re-infers fresh from the current observation.

Prerequisites:

- `BiPico4` teleop from `lerobot-xense` is importable (`lerobot.teleoperators.bi_pico4`)
- XenseVR PC Service running and both Pico4 controllers detected
- Not compatible with `--args.rtc-enabled` in this release (the RTC broker has its own
  execution queue + blending; startup is refused if both flags are set)

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-05 \
    --args.host 10.142.1.1 --args.port 8000 \
    --args.pico4-intervention
```

Recommended first-run flow:

1. `--args.dry-run --args.pico4-intervention` — hold either grip and confirm the printed 20D
   action tracks the controller pose; release and confirm the next log shows the
   `Clearing ActionChunkBroker cache (intervention released).` line.
2. Real run with `--args.stiffness-ratio 0.1` — verify the handoff does not snap the
   arm. The wrapper resyncs the teleop's internal target to the live TCP pose
   every non-intervention frame to keep the first override frame continuous.

Control scheme (inherited from `BiPico4`):

| Input | Effect |
|---|---|
| Either grip held | Both arms follow controller pose (intervention ON) |
| Both grips released | Intervention OFF; policy resumes from the next observation |
| Left / right trigger | Respective gripper position while intervention is on |

While intervention is active, every step's action dict carries `is_intervention: True`
(and `False` otherwise). Enabling `--args.record` alongside `--args.pico4-intervention`
automatically adds a frame-level `observation.is_intervention` column to the dataset
(1 = human takeover frame, 0 = policy), so recorded episodes identify which frames
were teleoperated. Override the auto behavior with `--args.record-intervention-flag
true|false` (e.g. force it off, or on without Pico4).

---

## Synchronized Recording

Record a new LeRobot-format dataset while running inference (raw 640×480 images,
absolute actions — same format as the training data).

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-05 \
    --args.host 10.142.1.1 --args.port 8000 \
    --args.record \
    --args.record-repo-id Xense/my_new_dataset \
    --args.task "pack 6 cosmetic bottles into the carton"
```

The dataset is saved locally to `~/.cache/huggingface/lerobot/<repo_id>` by default.
Use `--args.record-root /path/to/dir` to override the save location.

### Resuming an existing dataset

Append new episodes to an already-recorded dataset with `--args.resume` (episode
numbering continues, frames append to the same parquet/video files):

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-05 \
    --args.host 10.142.1.1 --args.port 8000 \
    --args.record \
    --args.record-repo-id Xense/my_new_dataset \
    --args.task "pack 6 cosmetic bottles into the carton" \
    --args.resume
```

`fps`/`features`/`robot_type` must match the existing dataset — use the same
`--args.record-repo-id`/`--args.record-root` and `--args.runtime-hz` as the
original run. The dataset is opened in offline mode, so a missing or incomplete
local dataset fails loudly instead of silently pulling from the Hub.

Video encoding knobs: `--args.record-vcodec auto` (default) picks a hardware
encoder (e.g. `h264_nvenc`) when available, else `libsvtav1`;
`--args.record-streaming-encoding` (default on) encodes frames in background
threads during recording so episode saves are near-instant.

On shutdown (Ctrl+C, keyboard exit, or a runtime error) the recorder's
`finalize()` saves any partial in-memory episode before the robot disconnects.

### Keyboard-delimited episodes (lerobot style)

Add `--args.keyboard-control` to delimit episodes with the keyboard instead of
a fixed count (requires the synchronous runtime — incompatible with
`--args.action-hz > 0`):

| Key | Effect |
|---|---|
| Right arrow | Start a new episode (idle) / end + save it (running) |
| Left arrow | Discard the current episode and re-record it |
| Enter | End + save (`is_success=True` with `--args.confirm-success`) |
| Backspace | End + save (`is_success=False` with `--args.confirm-success`) |
| ESC | End + save the current episode, then exit cleanly |

With `--args.confirm-success`, each frame gets an `observation.is_success`
column backfilled from the end key (right arrow leaves it NaN = unconfirmed).

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-05 \
    --args.host 10.142.1.1 --args.port 8000 \
    --args.record \
    --args.record-repo-id Xense/my_new_dataset \
    --args.task "pack 6 cosmetic bottles into the carton" \
    --args.keyboard-control --args.confirm-success
```

---

## Converting Recorded Dataset to MCAP

[MCAP](https://mcap.dev/) files can be opened in [Foxglove Studio](https://foxglove.dev/)
for visual inspection of observations, actions, and camera streams.

> Run in the `lerobot-xense` environment where `lerobot2mcap` is installed.

### Convert all episodes

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset
```

### Convert specific episodes

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset \
    --episodes 0 1 2
```

### Parallel conversion (faster for large datasets)

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset \
    --jobs 4
```

Each episode produces a separate `.mcap` file under the output directory.
Open any `.mcap` file directly in Foxglove Studio to inspect it.

---

## Action / State Space

20-dimensional Cartesian space:

| Index | Key |
|---|---|
| 0–2 | `left_tcp.x/y/z` |
| 3–8 | `left_tcp.r1–r6` (rotation) |
| 9–11 | `right_tcp.x/y/z` |
| 12–17 | `right_tcp.r1–r6` (rotation) |
| 18 | `left_gripper.pos` |
| 19 | `right_gripper.pos` |

TCP positions are **delta** actions; gripper positions are **absolute**.

---

## File Structure

```
examples/bi_flexiv_rizon4_rt/
├── main.py         # Entry point and CLI args
├── recipe.py       # Recipe YAML loader (bench hardware) binding
├── env.py          # OpenPI Environment adapter (image resize, obs format, keyboard wrapper)
├── real_env.py     # BiFlexivRizon4RT robot control wrapper
├── recorder.py     # LeRobot-format episode recorder subscriber
├── keyboard_control.py # Lerobot-style keyboard episode delimiting
├── intervention.py # Pico4 VR human-in-the-loop intervention wrappers
├── subscribe.py    # Obs streamer to the video-playback laptop
├── recipes/        # Bench recipes (arm SNs, poses, cameras, grippers)
└── runs/           # Run YAMLs (preset CLI flags per launch)
```
