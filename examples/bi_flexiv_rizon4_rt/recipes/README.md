# Inference recipes — bimanual Flexiv Rizon4 RT

One file per physical bench. `--args.robot-recipe <name>` picks one; the loader
([`../recipe.py`](../recipe.py)) decodes the `robot:` block through draccus into
a `BiFlexivRizon4RTConfig`, exactly the way `lerobot-record` and
`lerobot-teleoperate` decode theirs.

```bash
python -m examples.bi_flexiv_rizon4_rt.main \
    --args.host 192.168.5.87 --args.port 8000 \
    --args.robot-recipe forward-05
```

A bare name resolves to `<name>.yaml` in this directory; anything containing a
`/` or ending in `.yaml` is taken as a path, so a recipe living in the
lerobot-xense tree works too:

```bash
--args.robot-recipe ~/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-05.yaml
```

## Why this exists

Bench hardware used to be selected by `--args.bi-mount-type <name>`, which
indexed a `stations/` table inside lerobot. That layer was removed upstream
(`3b964bc6`): the config dataclass no longer carries arm SNs, start/home poses or
camera SNs at all, and whoever builds the config has to supply them. Recipes are
how lerobot supplies them, so this is the same mechanism, pointed at files we
own.

## What goes in one, and what doesn't

**Recipe = bench hardware.** Arm SNs, start/home joint poses (degrees, J1..J7),
the head camera, and the `gripper:` block — the things that are true of the
bench regardless of which policy is running.

**CLI = run tuning.** `--args.inner-control-hz`, `--args.interpolate-cmds`,
`--args.stiffness-ratio`, `--args.use-force`, `--args.go-to-start`,
`--args.enable-tactile-sensors`, `--args.log-level`. These stay flags because
they change between runs on the same bench.

The CLI owns them outright: each has a concrete default, so it is **always**
applied on top of the decoded config. A tuning key written into a recipe — or
already present in an upstream lerobot teleop/record recipe you point at by
path — loses to the flag. The loader logs every key it overrides, e.g.

```
[warning] CLI flags override forward-04.yaml: enable_tactile_sensors: True -> False, log_level: 'INFO' -> 'DEBUG'
```

The recipes here leave tuning keys out entirely, so nothing is shadowed when
you use one of them.

Tuning that you want to *keep* across runs belongs in a run YAML rather than in
a recipe: [`../runs/README.md`](../runs/README.md). A run file names its bench
with `robot_recipe:` and presets the flags, so the bench file stays purely about
hardware and one bench can serve several tasks.

Note `enable_tactile_sensors` defaults to **off** for inference: the policy
consumes `head`, `left_wrist` and `right_wrist` only, so the four tactile
cameras would cost USB bandwidth and loop time for frames nothing reads. Pass
`--args.enable-tactile-sensors` when recording a dataset that wants them.

## Cameras

Only `head` is pinned. Both gripper backends carry their wrist camera and two
tactile sensors on the gripper's own USB hub, and sniff them off it at connect
(`gripper.auto_discover_cameras`, on by default for both). The injected keys are
`{side}_wrist` and `{side}_tactile_{0,1}` — the same names the pinned wiring
used, so datasets stay compatible.

## Keeping these in sync with lerobot-xense

Each bench's hardware exists in two places: here, and in
`lerobot-xense/recipes/{teleop,record}/bi_flexiv_rizon4_rt/`. That is the
accepted cost of self-contained recipes — upstream's `recipes/README.md` carries
the same warning. **When a bench changes — a swapped camera, a re-mounted arm, a
new arm SN — grep every recipe naming that bench, in both repos, and update all
of them.** A missed copy does not fail loudly; it connects to the wrong serial or
drives to a stale start pose.

| recipe | grippers | ported from |
|---|---|---|
| `forward-01.yaml` | taccap_follower | `lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-01.yaml` |
| `forward-04.yaml` | taccap_follower | `…/teleop/bi_flexiv_rizon4_rt/forward-04.yaml` |
| `forward-05.yaml` | serial (unverified upstream) | `…/teleop/bi_flexiv_rizon4_rt/forward-05.yaml` |
| `forward-06.yaml` | serial | `…/record/bi_flexiv_rizon4_rt/assemble_box.yaml` |
| `diagonal-02.yaml` | taccap_follower | `…/teleop/bi_flexiv_rizon4_rt/diagonal-02.yaml` |

`forward-01` and the retired `forward-dewu` station are the **same two arms**
(`Rizon4s-063458` / `Rizon4s-063670`) and the same head camera
(`337322070722`); `forward-01` is that bench after its re-mount, and its poses
are the current ones. The shoe-insole demo in the root README runs `forward-05`.
