# BiElite CS66 RT — Real Robot Inference

Dual-arm Elite CS66 inference client using the OpenPI policy server. Trained with
`LeRobotBiEliteDataConfig` (see `configs/_examples/pi05_base_bi_elite_*.yaml`).

State/action are 20D and **per-side-grouped**, the driver's own feature order:

```
[left_tcp.x/y/z/r1-r6 (0-8), left_gripper.pos (9),
 right_tcp.x/y/z/r1-r6 (10-18), right_gripper.pos (19)]
```

This differs from the BiFlexiv example (both grippers last). TCP poses are in the
driver's world frame — see [`recipes/README.md`](recipes/README.md) for why the
recipe's mount rotations must match the training data.

## Usage

### Terminal 1: policy server (GPU machine)

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_bi_elite_egg_carton_to_tray_1012_h100 \
    --policy.dir=<checkpoint_dir>
```

### Terminal 2: robot client (`lerobot-xense` conda environment)

Always start with a dry run — the arms connect and go to the start pose, but
policy actions are only printed:

```bash
python -m examples.bi_elite_cs66_rt.main --args.run dry-run --args.host <server>
```

Then for real:

```bash
python -m examples.bi_elite_cs66_rt.main \
    --args.robot-recipe diagonal-07-taccap --args.host <server> --args.port 8000
```

Useful flags (see `--help` for all):

| Flag | Effect |
|---|---|
| `--args.rtc-enabled` | Real-time chunking; the prefix is re-based on the server |
| `--args.action-hz 60` | Decoupled runtime: actions at 60 Hz, obs at `runtime_hz` |
| `--args.max-lin-speed 0.15` / `--args.max-ang-speed 0.8` | Tighter servo-loop speed caps than the recipe's |
| `--args.no-go-to-start` | Start from wherever the arms are |

Ctrl+C stops the runtime and disconnects the robot; a second Ctrl+C forces exit.

## Not included (yet)

Pico4 intervention, LeRobot recording and the obs subscriber stream from the
BiFlexiv example are not ported.
