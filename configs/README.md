# `configs/` — Per-task YAML training configs

This directory holds **one YAML per training/inference task**. Each file is a
self-contained representation of a `TrainConfig` and can be loaded by name:

```bash
# YAML file: configs/_examples/pi05_base_bi_flexiv_shoe_insole_..._h100.yaml
python scripts/train.py pi05_base_bi_flexiv_shoe_insole_..._h100 --exp-name=0519
```

## What's where

```
configs/
├── README.md                  ← this file
├── _examples/                 ← shared, in git, treat as templates
│   ├── debug_pi05.yaml
│   └── pi05_base_bi_flexiv_*.yaml
└── <your_task>.yaml           ← per-user (gitignored), put your in-flight configs here
```

The repo's `.gitignore` keeps **only** `configs/_examples/` (and this README)
under version control. Anything you drop into `configs/` directly is private
to your machine — perfect for in-flight experiments that don't need to be
shared yet.

## Lookup order (`get_config(name)`)

1. `configs/<name>.yaml`             — your local copy
2. `configs/_examples/<name>.yaml`   — shared example
3. Generated configs (`config._generated_configs`) — the RoboArena baselines,
   which are built as a family in Python because they pass tokenizer *classes*
   and lambdas that can't be serialized. Nothing else lives in Python.

## Writing a new config

Copy the closest example and edit. The bare minimum:

```yaml
# configs/_examples/<task_name>.yaml  (or configs/<task_name>.yaml for personal)

model:
  type: Pi0Config              # see openpi.training.registry.MODELS
  pi05: true
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m

data:
  type: LeRobotBiFlexivDataConfig   # see openpi.training.registry.DATA_CONFIGS
  repo_id: Xense/<your_dataset>
  default_prompt: "..."
  base_config:
    prompt_from_task: true

weight_loader:
  type: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi05_base/params

batch_size: 256
num_train_steps: 40000
num_workers: 64
fsdp_devices: 8
```

The **filename** is the config name — there is no `name:` field inside the
YAML. `exp_name` is supplied on the CLI (`--exp-name=...`).

## Full reference: `_FULL_REFERENCE.yaml`

[`configs/_examples/_FULL_REFERENCE.yaml`](_examples/_FULL_REFERENCE.yaml) lists
**every** `TrainConfig` field, every registered class for each polymorphic
slot, and the current default for each value. Open it alongside the example
you're editing — it answers "what does field X do" and "what other classes
can go in `data.type`" without forcing you to read `config.py`.

That file's name starts with `_` on purpose: leading-underscore files are docs,
not configs, so they're left out of the config listing and the `train.py`
subcommand choices. A test still parses it on every run, so the docs can't
silently rot.

## ⚠️ Before committing a YAML to `_examples/`

Read it once. **Do not check in machine-local absolute paths** such as
`/home/<you>/.../checkpoints/...`. Use the upstream URL
(`gs://openpi-assets/...`) or a path that works for every contributor.

## Registering a new class

Adding a new model/data-config/weight-loader Python class? Register its
string name in `src/openpi/training/registry.py` so YAML files can
reference it via `type: <YourClass>`.

## LoRA configs

Set `model.paligemma_variant` and/or `model.action_expert_variant` to a `*_lora`
value. **Do not write a `freeze_filter`** — the loader sees the `lora` substring
and derives the `flax.nnx` filter tree via `Pi0Config.get_freeze_filter()`, which
is the same thing the Python configs used to spell out by hand.

## Dumping a Python config to YAML

```bash
python scripts/dump_config_to_yaml.py <name>                       # to stdout
python scripts/dump_config_to_yaml.py <name> --output-dir configs  # to a file
```

Useful for the generated RoboArena baselines, or for turning a config you built
in a REPL into a file. The script re-parses what it wrote and refuses to save
anything that doesn't reload to an identical config.

## Why are some configs NOT here?

The RoboArena baselines (`paligemma_*_droid`) pass tokenizer **classes** and
lambdas as config values, and neither survives serialization. They're generated
as a family by `config._generated_configs()` and resolved by `get_config()` after
the YAML lookup misses. Everything else is a file in this directory.

## Checking your work

```bash
pytest src/openpi/training/config_yaml_test.py   # every example parses, no local paths
python scripts/train.py --help                   # your config should be listed
```
