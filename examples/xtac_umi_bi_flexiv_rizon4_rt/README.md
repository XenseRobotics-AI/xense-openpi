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

There is **no world/base frame conversion**. XTac-UMI poses are recorded in the
Pico4 gravity-aligned world frame and the Flexiv reports base-frame poses; the
policy bridges that gap through `use_delta_cartesian_actions` (TCP dims are
delta-encoded against the current state at train and inference time), not through
an extrinsic calibration or a per-episode re-basing.

The one conversion that does happen is the **gripper end-frame change of basis**
(`gripper_frame.py`): the Flexiv driver reports TCP orientations with z along the
fingertips (y right, x up) while the training data uses x along the fingertips
(y left, z up). The change of basis is a constant, self-inverse rotation applied
at the websocket boundary; translation is untouched.

That axis claim is a hardware/CAD fact this repo cannot verify. If a bench's
driver already reports UMI-convention poses, turn it off:

```bash
--args.no-align-gripper-frames
```

## Start the policy server

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_xtac_umi_pick_up_cube_0807_h200 \
    --policy.dir=<checkpoint_dir> \
    --port=8000
```

The checkpoint must have been trained with the current XTac-UMI conventions:
`base_0_rgb` black and masked out, left wrist in the left slot, right wrist in
the right slot, 20D per-side-grouped state/action.

## Run the client

Activate the `lerobot-xense` conda environment, then:

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
