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
  TacCap). Optional; drop the `gripper:` block to run without one. The old
  `flare_gripper` backend no longer exists.
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

`--robot_recipe` is required. It names a recipe under
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
    --robot_recipe default \
    --host 192.168.2.215 \
    --port 8000
```

### With RTC Enabled

```bash
python -m examples.flexiv_rizon4_rt.main \
    --robot_recipe default \
    --host 192.168.2.215 \
    --port 8000 \
    --rtc_enabled \
    --execution_horizon 20 \
    --runtime_hz 25
```

### Dry Run (print actions without executing)

```bash
python -m examples.flexiv_rizon4_rt.main \
    --robot_recipe default \
    --host 192.168.2.215 \
    --port 8000 \
    --dry_run
```

## Configuration Options

Precedence is `dataclass default < recipe YAML < --args.* flag`, so any flag
below overrides what the recipe says.

### Robot

| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_recipe` | — (required) | Bench recipe name under `recipes/`, or a path |
| `host` | localhost | Policy server IP |
| `port` | 8000 | Policy server port |
| `use_force` | False | Enable force control axes |
| `go_to_start` | False | Move to start position on connect |
| `runtime_hz` | 20.0 | Control loop frequency (Hz) |
| `dry_run` | False | Print actions without executing |

### RT-specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stiffness_ratio` | 0.2 | Cartesian stiffness multiplier (×K_x_nom) |
| `inner_control_hz` | 1000 | How often the 1 kHz RT loop consumes a new Python command |
| `interpolate_cmds` | True | Enable linear interpolation between consumed commands |
| `start_position_degree` | None | Start joint angles (deg); None = use the recipe's |
| `zero_ft_sensor_on_connect` | True | Zero FT sensor on startup |

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
| `rtc_enabled` | False | Enable RTC mode |
| `execution_horizon` | 30 | Execution window size (< action_horizon) |
| `action_queue_size_to_get_new_actions` | 20 | Queue threshold for new inference |
| `blend_steps` | 5 | Steps for blending old/new actions |
| `default_delay` | 2 | Default inference delay (steps) |

## Control Architecture

```
Python main.py (25 Hz)
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
2. The robot moves to the recipe's start position on connect only with `--go_to_start` (default: off)
3. Use `--dry_run` to verify action values without robot movement
4. Press Ctrl+C for graceful shutdown — RT thread stops, robot returns to home
5. `stiffness_ratio=0.2` (20% nominal) provides compliant behaviour; increase for stiffer tracking
