# XTac-UMI checkpoint on BiFlexiv Rizon4 RT

Serves a checkpoint trained on handheld XTac-UMI data (`bi_taccap_gripper`, see
the `xense-taccap-lerobot` repo) on a dual Flexiv Rizon4 RT with taccap grippers.
Only the two wrist cameras are connected; the head camera is never sent.

## Dim order

The rig and the robot use the **same 20D order** — the recording rig's, grouped
per side:

```
[left_tcp(0-8), left_gripper(9), right_tcp(10-18), right_gripper(19)]
```

`real_env.py` builds that vector directly from the driver's named keys
(`left_tcp.x`, …, `right_gripper.pos`), so no dim regrouping happens anywhere in
this example. Note this is **not** the order `examples/bi_flexiv_rizon4_rt` uses
(both TCPs first, both grippers last) — the two examples are not interchangeable.

Each 9D TCP pose is `[x, y, z, r1..r6]`, where r1-r3 / r4-r6 are the first two
columns of the rotation matrix.

## Coordinate frames

There is **no world/base frame conversion** — no extrinsic calibration between the
Pico4 rig and the arms, and no per-episode re-basing into a robot frame. Which
cancels the frame depends on how the checkpoint was trained:

- **Rebased checkpoints** (`use_delta_cartesian_actions`): the dataset was first
  re-based onto the arms' start TCPs (`rebase_umi_dataset.py`), and the TCP dims
  are delta-encoded against the current state. Absolute poses reach the model.
- **Frame-relative checkpoints** (`use_frame_relative_poses`, below): the model
  only ever sees SE(3) motion, so neither the Pico world origin nor a robot base
  enters the numbers at all.

The one conversion that does happen for **rebased** checkpoints is the **gripper
end-frame change of basis** (`gripper_frame.py`): the Flexiv driver reports TCP
orientations with z along the fingertips (y right, x up) while that training data
uses x along the fingertips (y left, z up). The change of basis is a constant,
self-inverse rotation applied at the websocket boundary; translation is untouched.

That axis claim is a hardware/CAD fact this repo cannot verify. If a bench's
driver already reports UMI-convention poses, turn it off:

```bash
--args.no-align-gripper-frames
```

Frame-relative checkpoints must run with it off: the server chains the predicted
motion straight onto the arm's current TCP, so the robot TCP frames have to
already match the XTac-UMI gripper frame (`--args.frame-relative-poses` refuses to
run together with the change of basis).

## Frame-relative checkpoints

`use_frame_relative_poses` trains on the raw recording: the model's state is the
pose since the episode's first frame, `inv(T_0) @ T_t`, and each action is the
motion since the previous one. Gripper openings stay absolute in both.

The client therefore sends the state as **two frames**, `[T_0, T_t]`; the server
computes the relative pose for the model and uses `T_t` as the point the
predicted motions chain onto. The two frames never reach the model.

`runs/frame-relative.yaml` and the training config line up as follows:

| Run file setting | Value | Why |
| --- | --- | --- |
| `frame_relative_poses` | `true` | Sends the two-frame state `[T_0, T_t]` |
| `state_anchor` | `episode_start` | Same anchor as training |
| `align_gripper_frames` | `false` | TCP frames already match; no change of basis on top |
| `go_to_start` | `true` | Every episode starts at the same pose, which keeps the anchor consistent with training |
| `action_horizon` | `20` | Matches the training `action_horizon: 20` |
| `dry_run` | `true` | Sends no action until asked |
| `runtime_hz` | `30` | Matches the recording rate |

Start the server with the config the checkpoint was trained under (the norm stats
in `assets/` are keyed by it):

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_xtac_umi_g1_pick_place_frame_relative_h100 \
    --policy.dir=<checkpoint_dir> \
    --port=8000
```

Then, on the robot host:

```bash
# 1. Dry run: prints the actions, moves nothing.
python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main \
    --args.run frame-relative \
    --args.host <policy-server-ip> --args.port 8000

# 2. Once the printed actions look sane, execute.
python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main \
    --args.run frame-relative \
    --args.host <policy-server-ip> --args.port 8000 \
    --args.no-dry-run

# Optional: pace-decoupled runtime, actions faster than 30 Hz
#   --args.action-hz 45        # 1.5x the recorded speed
```

Keep the synchronous runtime (`action_hz` left at 0) until the dry run looks
right. `action_hz` above 30 plays the same chunks faster rather than finer: N
steps take N/action_hz seconds, so 60 Hz runs the demonstrations at 2x.

In the dry run, check two things first: the first action's left TCP xyz should be
within a few millimetres of the robot's current TCP, and dims 9 and 19 should look
like gripper openings (0-1). Either being far off means the state/action order is
not what the checkpoint expects.

## Start the policy server (rebased checkpoints)

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_xtac_umi_g1_clean_plates_fwd04_rebased_1014_h100 \
    --policy.dir=<checkpoint_dir> \
    --port=8000
```

Frame-relative checkpoints use their own config name and run file; see
[Frame-relative checkpoints](#frame-relative-checkpoints).

The checkpoint must have been trained with the current XTac-UMI conventions:
`base_0_rgb` black and masked out, left wrist in the left slot, right wrist in
the right slot, 20D per-side-grouped state/action.

## Run the client

Activate the `lerobot-xense` conda environment. The generic command line below
works for any checkpoint; frame-relative ones should use
`runs/frame-relative.yaml` instead (see above).

```bash
# A run file presets the flags (see runs/README.md)
python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main --args.run dry-run --args.host 192.168.142.220

# ...or spell it out
python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-04 \
    --args.host 192.168.142.220 \
    --args.port 8000 \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run
```

`--args.robot-recipe` reuses `examples/bi_flexiv_rizon4_rt/recipes/` so benches
are defined once for the repo; `forward-04` is the taccap-gripper bench. A path
to any recipe YAML works too. The head camera the recipe pins is dropped at
construction, and `use_force` / `gripper.enable_tactile` are forced off — the
checkpoint consumes the 20D pose/gripper space only.

Dry-run suppresses policy actions, but the robot still connects and the episode
reset can move it to the configured start pose. Add `--args.no-go-to-start` when
the connection itself must not perform that move.

`--args.action-hz` switches the decoupled runtime on (`0` keeps the synchronous
one). It is available for every checkpoint here; raising it above 30 plays the
same action chunks faster.
