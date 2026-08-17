# Run configs — bimanual Flexiv Rizon4 RT

One file per *launch*. `--args.run <name>` presets the CLI flags for a run, so a
demo that used to be ten lines of flags is one line:

```bash
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole
```

A bare name resolves to `<name>.yaml` in this directory; anything containing a
`/` or ending in `.yaml` is taken as a path.

## Run vs recipe

| | Describes | Lives in | Changes when |
|---|---|---|---|
| **Recipe** | The bench: arm SNs, start/home poses, head camera, `gripper:` block | [`../recipes/`](../recipes/) | You re-mount the arms or swap a gripper |
| **Run** | The launch: which policy server, which task, RTC, recording, obs streaming, control-loop tuning | here | You start a different demo on the same bench |

A run file names its bench with `robot_recipe:`, so one bench serves as many
runs as you have tasks, and one run file can be pointed at another bench by
overriding `--args.robot-recipe` on the command line.

## What goes in one

Keys are **exactly the CLI field names** — a run file is the command you would
have typed, so `--args.subscribe-hz 10` becomes `subscribe_hz: 10.0`. Every field
of `Args` in [`../main.py`](../main.py) is allowed; see `--help` for the full list
with its defaults. A key that isn't a field fails the load rather than being
silently ignored, which is the whole point of writing it down.

Do not add a `run:` key — the file *is* the run.

## Precedence

```
dataclass defaults  <  run YAML  <  CLI flags
```

So a run file is a starting point, never a cage:

```bash
# The dewu demo, but don't move the arms
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole --args.dry-run

# ...and against a different policy server
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole --args.host 192.168.2.100
```

`main` logs the run file it loaded and which keys the CLI overrode, so the swap
is visible in the log rather than inferred from behaviour. `--help` shows the
effective defaults *after* the run file is applied.

## Shipped runs

| File | What it is |
|---|---|
| [`dewu-shoe-insole.yaml`](dewu-shoe-insole.yaml) | Robot host of the three-machine dewu demo: forward-05, RTC on, head camera streamed to the screen PC at 10 Hz |
| [`dry-run.yaml`](dry-run.yaml) | Bench check on forward-05 — connects and runs the policy, prints actions instead of executing them |

The mechanism lives in [`../../run_config.py`](../../run_config.py) and is shared
by the other robot examples. Tests: `examples/run_config_test.py` also parses
every file in this directory, so a run file can't drift away from `Args`.
