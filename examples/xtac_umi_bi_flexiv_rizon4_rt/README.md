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

Checkpoints are trained on **re-based** data. `rebase_umi_dataset.py` composes
each handheld episode onto the arms' start TCPs, the offline equivalent of
teleoperating the arms with the hands. Its output state and actions are Flexiv
base-frame poses in the Flexiv TCP convention, so **the client converts nothing**:
the 20D state the driver reports goes to the server as is, and the actions come
back executable.

```bash
python -m examples.xtac_umi_bi_flexiv_rizon4_rt.rebase_umi_dataset \
    --src-repo-id Xense/<recording> --dst-repo-id Xense/<recording>-fwd04 \
    --left-home  x y z qw qx qy qz \
    --right-home x y z qw qx qy qz
```

The homes are the Flexiv `tcp_pose` of each arm at the recipe's start joints. Keep
`go_to_start` on so an episode begins where the training data does.

A raw recording cannot be trained directly: its poses are relative to the headset
at record start, not to the robot. `LeRobotXtacUmiDataConfig` accepts only the 20D
re-based layout and rejects the raw 29D one.

## Start the policy server

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_xtac_umi_g1_clean_plates_fwd04_rebased_1014_h100 \
    --policy.dir=<checkpoint_dir> \
    --port=8000
```

The checkpoint must have been trained on re-based data with the current XTac-UMI
conventions: `base_0_rgb` black and masked out, left wrist in the left slot, right
wrist in the right slot, 20D per-side-grouped state/action.

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
