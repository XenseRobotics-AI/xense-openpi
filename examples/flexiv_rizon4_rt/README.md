# Flexiv Rizon4 RT Robot Inference

This example runs inference on a real Flexiv Rizon4 robot using the **RT driver** (`flexiv_rt`), which spawns a C++ RT thread at **1 kHz** (SCHED_FIFO) for deterministic Cartesian streaming control.

This is the only Flexiv path left. lerobot removed the NRT driver (`flexivrdk`)
and its `flexiv_rizon4` package at `d6d02f88`, and the `flexiv_rizon4_real`
example that wrapped it was deleted with it. What the RT driver gives you:
RT Cartesian control only (no joint impedance), a 10D action space always, a
non-blocking RT reset trajectory, and <1 ms command latency through shared
memory instead of ~10 ms per NRT call.

## Hardware Requirements

- Flexiv Rizon4 7-DOF collaborative robot
- A gripper — `serial` (USB-serial parallel jaw) or `taccap_follower` (centric
  TacCap). Required: this example's state and action vectors are 10D ending in
  `gripper.pos`, and lerobot omits that key when no gripper is configured. The
  old `flare_gripper` backend no longer exists.
- Any camera the policy consumes, pinned in the recipe. **Nothing comes through
  the gripper any more** — neither backend carries a camera, and this driver
  (unlike the bimanual one) does no USB-hub auto-discovery.
- Network connection to the robot

## Setup

```bash
# Install lerobot-xense with RT support
pip install -e /path/to/lerobot-xense

# Install flexiv_rt (libpyflexiv)
# Follow instructions at your internal repo
```

## Start Policy Server

```bash
python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_base_xense_flare_pick_and_place_cube \
    --policy.dir=checkpoints/your_checkpoint
```

## Pick a bench

`--args.robot-recipe` is required. It names a recipe under
[`recipes/`](recipes/) — or a path to any recipe YAML — that carries the arm SN,
start pose, cameras and the typed `gripper:` block. The lerobot config dataclass
no longer holds bench hardware, so something has to supply it, and recipes are
what lerobot's own CLIs use. [`recipes/README.md`](recipes/README.md) covers what
belongs in one; [`recipes/default.yaml`](recipes/default.yaml) is a template to
copy per bench, not a real bench.

## Usage

### Basic Inference (non-RTC)

```bash
python -m examples.flexiv_rizon4_rt.main \
    --args.robot-recipe default \
    --args.host 192.168.2.215 \
    --args.port 8000
```

### With RTC Enabled

```bash
python -m examples.flexiv_rizon4_rt.main \
    --args.robot-recipe default \
    --args.host 192.168.2.215 \
    --args.port 8000 \
    --args.rtc-enabled \
    --args.execution-horizon 20 \
    --args.runtime-hz 25
```

### Dry Run (print actions without executing)

```bash
python -m examples.flexiv_rizon4_rt.main \
    --args.robot-recipe default \
    --args.host 192.168.2.215 \
    --args.port 8000 \
    --args.dry-run
```

## Configuration Options

The recipe carries bench hardware; the CLI owns run tuning outright. Every
tuning flag has a concrete default, so it is **always** applied on top of the
decoded recipe — a tuning key written into a recipe (or already present in an
upstream lerobot teleop/record recipe you point at by path) loses to the flag.
The loader logs each key it overrides, so the swap is visible rather than
silent.

### Robot

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--args.robot-recipe` | — (required) | Bench recipe name under `recipes/`, or a path |
| `--args.host` | localhost | Policy server IP |
| `--args.port` | 8000 | Policy server port |
| `--args.use-force` | False | Enable force control axes |
| `--args.go-to-start` | False | Move to start position on connect |
| `--args.runtime-hz` | 20.0 | Control loop frequency (Hz) |
| `--args.dry-run` | False | Print actions without executing |

### RT-specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--args.stiffness-ratio` | 0.2 | Cartesian stiffness multiplier (×K_x_nom) |
| `--args.inner-control-hz` | 1000 | How often the 1 kHz RT loop consumes a new Python command |
| `--args.interpolate-cmds` | True | Enable linear interpolation between consumed commands |
| `--args.start-position-degree` | None | Start joint angles (deg); None = use the recipe's |
| `--args.zero-ft-sensor-on-connect` | True | Zero FT sensor on startup |

### Gripper — recipe only

The flat `gripper_type` / `gripper_mac_addr` / `gripper_cam_size` /
`gripper_rectify_size` / `gripper_max_pos` flags are gone: lerobot promoted
grippers to a typed device family at `3b964bc6`, so the gripper is one `gripper:`
block in the recipe, decoded through `lerobot.grippers.GripperConfig`. A knob
belonging to the other backend, or a typo, now fails the parse instead of being
silently ignored.

| Backend | Hardware | Key knobs |
|---|---|---|
| `serial` | parallel jaw, USB serial | `gripper_v_max`, `gripper_f_max`, `gripper_min_pos`, `gripper_max_pos`, `side`/`port`/`sn` |
| `taccap_follower` | centric TacCap, FDCAN | `kp`, `kd`, `feedforward_torque`, `control_hz`, `side` |

A single arm has no side to infer, so the `serial` backend needs `side` (or an
explicit `port`/`sn`) to find its board. See
[`recipes/README.md`](recipes/README.md) and lerobot's
`src/lerobot/grippers/README.md`.

### RTC

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--args.rtc-enabled` | False | Enable RTC mode |
| `--args.execution-horizon` | 50 | Execution window size (< action_horizon) |
| `--args.action-queue-size-to-get-new-actions` | 20 | Queue threshold for new inference |
| `--args.blend-steps` | 3 | Steps for blending old/new actions |
| `--args.default-delay` | 2 | Default inference delay (steps) |

## Control Architecture

```
Python main.py (runtime_hz, default 20 Hz)
  └─ ActionChunkBroker / RTCActionChunkBroker
      └─ WebsocketClientPolicy  →  Policy Server (GPU)
          └─ FlexivRizon4RTEnvironment
              └─ FlexivRizon4RT.send_action()
                  └─ cc.set_target_pose()  [shared memory write]
                      └─ C++ RT thread (1 kHz, SCHED_FIFO)
                          └─ StreamCartesianMotionForce  →  Robot
```

## Safety Notes

1. Always ensure the robot workspace is clear before running
2. The robot moves to the recipe's start position on connect only with `--args.go-to-start` (default: off)
3. Use `--args.dry-run` to verify action values without robot movement
4. Press Ctrl+C for graceful shutdown — RT thread stops, robot returns to home
5. `stiffness_ratio=0.2` (20% nominal) provides compliant behaviour; increase for stiffer tracking
