# Inference recipes — single-arm Flexiv Rizon4 RT

One file per physical bench. `--args.robot-recipe <name>` picks one; the loader
([`../recipe.py`](../recipe.py)) decodes the `robot:` block through draccus into
a `FlexivRizon4RTConfig`, the same way `lerobot-record` decodes its own.

```bash
python -m examples.flexiv_rizon4_rt.main \
    --args.host 192.168.2.215 --args.port 8000 \
    --args.robot-recipe default
```

A bare name resolves to `<name>.yaml` here; anything with a `/` or a `.yaml`
suffix is taken as a path.

[`default.yaml`](default.yaml) is a **template, not a bench** — lerobot-xense
ships no committed single-arm recipe to transcribe from, so its arm SN, camera
paths and gripper backend are unverified placeholders. Copy it, fill in your
hardware, commit that.

## Why this exists

The gripper stopped being a pile of flat fields on the arm config. Upstream
`3b964bc6` promoted grippers to their own device family with two backends —
`serial` (USB-serial parallel jaw) and `taccap_follower` (centric TacCap, MIT
impedance) — configured by one typed `gripper:` block. `gripper_type`,
`gripper_mac_addr`, `gripper_cam_size`, `gripper_rectify_size` and
`gripper_max_pos` no longer exist, and neither does the `flare_gripper` backend
they described.

Two consequences worth knowing before you write a recipe:

- **Nothing comes through the gripper.** Neither backend carries a camera. The
  wrist camera that used to arrive with the Flare gripper is now an ordinary
  `cameras:` entry, and the tactile sensors are ordinary `xense` cameras.
- **This driver does no auto-discovery.** The bimanual driver sniffs each
  gripper's wrist + tactile cameras off its USB hub at connect; the single-arm
  one does not. Pin every camera the policy needs.

The key `wrist_cam` is the one `env.py` maps to the policy's
`observation/wrist_image_left`; other keys pass through under their own names.

## What goes in one, and what doesn't

**Recipe = bench hardware.** Arm SN, start pose (degrees, J1..J7), cameras, the
`gripper:` block.

**CLI = run tuning.** `--args.use-force`, `--args.go-to-start`,
`--args.stiffness-ratio`, `--args.inner-control-hz`,
`--args.interpolate-cmds`, `--args.zero-ft-sensor-on-connect`,
`--args.log-level`. They are applied on top of the decoded config:

```
dataclass default  <  recipe YAML  <  --args.* flag
```
