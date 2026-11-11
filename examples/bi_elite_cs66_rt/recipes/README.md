# Inference recipes — bimanual Elite CS66 RT

One file per physical bench and gripper. `--args.robot-recipe <name>` picks one;
the loader ([`../recipe.py`](../recipe.py), shared with the Flexiv examples via
[`../../flexiv_recipe.py`](../../flexiv_recipe.py)) decodes the `robot:` block
through draccus into a `BiEliteCS66RTConfig`, the way `lerobot-record` and
`lerobot-teleoperate` decode theirs. A path works too, so a recipe from the
lerobot-xense tree can be used directly:

```bash
--args.robot-recipe ~/lerobot-xense/recipes/teleop/bi_elite_cs66_rt/diagonal-08-taccap.yaml
```

| Recipe | Bench | Gripper | Datasets recorded on it |
|---|---|---|---|
| `diagonal-07-taccap` | diagonal-07 | taccap_follower | — |
| `diagonal-08-taccap` | diagonal-08 | taccap_follower | `Xense/bi_elite_cs66_rt_egg_carton_to_tray` |

## The world frame must match the training data

The driver lifts each arm's TCP pose from its tilted base frame into one shared,
gravity-aligned world frame and maps actions back the same way. With
`native_assembly` set, the frame is libpyelite's packaged world<-base transform,
rotation **and translation**; without it, the driver uses the rotation-only
`*_world_rotation` (or the `*_mount_*_deg` angles). The policy reads and outputs
**absolute world-frame poses**, so a checkpoint is only valid on a recipe whose
world frame matches the one the data was recorded with. Dropping
`native_assembly` keeps the orientations identical but shifts every position by
the assembly translation (~1.2 m in z on bench 08): the state lands outside the
training range and the policy drives the arms the wrong way. Check the first
logged action against a dataset episode's first frame before letting the arms move. When the upstream
lerobot-xense recipe changes these, copy the change here *and* re-check which
datasets were recorded before vs. after it.

## Recipe vs CLI

**Recipe = bench hardware**: controller IPs, start/home poses, mount rotations,
head camera, `gripper:` block, payload and the singularity/joint-velocity guards.

**CLI = run tuning**: `--args.go-to-start`, `--args.enable-tactile`,
`--args.log-manipulability`, `--args.max-lin-speed`, `--args.max-ang-speed`. The
loader logs each recipe value a flag overrides. The speed caps default to `None`,
which keeps the recipe's value.
