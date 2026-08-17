# YAML Config System — Feature Notes & Changelog

> **Everything below the "Phase 1" heading describes the original dual-track
> design and is kept for context only.** `_CONFIGS` no longer exists; see
> Phase 2 first, then [`configs/README.md`](../configs/README.md) for how the
> system works today.

---

## Phase 2 — `_CONFIGS` removed

Phase 1 shipped YAML *alongside* the Python `_CONFIGS` list. Two sources of
truth for the same thing drifted almost immediately: by the time this phase
started, `configs/_examples/` held a `..._newbalacne_..._0515_h100.yaml` whose
`_CONFIGS` twin had been edited in place into `_0616_h100`, and the equivalence
test that was supposed to catch exactly that had been red on `main`. So YAML is
now the only place configs live.

**What made the last 7 configs migratable.** Each blocker got a mechanism rather
than an exception:

| Blocker | Fix |
|---|---|
| `SimpleDataConfig(data_transforms=lambda model: ...)` in the three DROID inference configs | `DroidInferenceDataConfig` — the lambda's only dynamic input was the model type, and `create()` already receives the model config, so no closure is needed. Verified to produce a byte-identical `DataConfig` for all three. |
| `repack_transforms=Group(inputs=[RepackTransform({...})])` in the ALOHA configs | `registry.TRANSFORMS` + a `_build_group` in the loader. `RepackTransform` carries nothing but a nested `str -> str` mapping, so it serializes cleanly. |
| `assets=AssetsConfig(asset_id="droid")` | The loader builds any nested block whose field annotation is a dataclass, instead of hard-coding `base_config`. No `type:` needed — the annotation already names the class. |
| `freeze_filter=Pi0Config(...).get_freeze_filter()` on LoRA configs | The loader already re-derived this from the `*_lora` variants on load; `dump` now skips it for the same reason, making the round-trip symmetric. |

**What stayed in Python.** Only the five RoboArena baselines, via
`config._generated_configs()`. They pass tokenizer *classes* as config values,
and they're generated as a family rather than authored one file at a time — a
YAML file per baseline would be the wrong shape for them.

**Order of operations.** All 13 configs were dumped and the (then still
existing) equivalence test was run to prove every YAML was `==` to the Python
config it replaced. Only then was `_CONFIGS` deleted. `config.py` went from 985
lines to 679 — schema and lookup, no config data.

**Interface changes:**

- `cli()` now builds its choices from `all_configs()` (every YAML on disk plus
  the generated ones). Same UX: `train.py <name> --exp-name=... --batch-size=64`,
  and `--help` still lists every config.
- `get_config()`'s third lookup tier is `_generated_configs()` instead of
  `_CONFIGS_DICT`.
- `yaml_examples_equivalence_test.py` is gone (nothing left to compare against);
  `config_yaml_test.py` replaces it — every example parses, names don't collide
  with generated ones, `all_configs()` loads clean, and no example may carry a
  machine-local `/home/...` path.
- `scripts/migrate_configs_to_yaml.py` → `scripts/dump_config_to_yaml.py`, now a
  general "dump any config by name" tool rather than a one-shot migration.
- Two configs that pointed `weight_loader.params_path` at another contributor's
  laptop (`/home/li/hubo/...`, `/home/ubuntu/...`) now point at
  `gs://openpi-assets/checkpoints/pi05_base/params`, with the original path
  recorded in a comment. Continue from a local checkpoint with
  `--weight_loader.params_path=...` on the CLI.
- `..._newbalacne_..._0515_h100.yaml` was deleted. It was the orphan half of the
  drift described above: its `_CONFIGS` twin had been edited in place into
  `_0616_h100` (different `repo_id`, prompt and step count), leaving a YAML for a
  run nobody was doing.

---

## Phase 1 — YAML alongside `_CONFIGS` (historical)

**Branch:** `feature/yaml-config-per-task`

This section describes the original design: training configs as **one YAML file
per task**, alongside the existing Python `_CONFIGS` list. Old configs kept
working; new tasks were to prefer YAML to avoid merge conflicts in the
919-line central `config.py`.

---

## Why

The `_CONFIGS` list in `src/openpi/training/config.py` has become a coordination
hotspot:

- 17 entries, ~30–50 lines each, all edited by everyone who adds a task.
- Recent merge incident: someone pushed an earbuds config to `main` with a
  hard-coded local path (`/home/li/hubo/.../checkpoints/20000/params`) directly
  into the shared file.
- Adding a new task requires modifying a file that 8+ contributors are also
  modifying — git conflicts on every PR.
- 90% of new configs are slight variations on existing ones (same model, same
  recipe, different `repo_id`/`prompt`) — pure copy-paste.

The YAML system targets these problems specifically: **new task = new YAML
file = no shared-file edits = no conflicts.**

---

## How it works

### Lookup order (`get_config(name)`)

```
1. configs/<name>.yaml             ← per-user, gitignored
2. configs/_examples/<name>.yaml   ← shared, in git
3. _CONFIGS_DICT[name]             ← legacy Python registry
```

First match wins. Existing call sites (`scripts/train.py`,
`scripts/compute_norm_stats.py`, `scripts/serve_policy.py`) don't change —
`get_config(name)` transparently checks both YAML and the legacy dict.

### One file per task

```yaml
# configs/_examples/pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100.yaml

model:
  type: Pi0Config              # ← string discriminator, resolved via registry.py
  pi05: true
  paligemma_variant: gemma_2b
  enable_training_time_rtc: true

data:
  type: LeRobotBiFlexivDataConfig
  repo_id: Xense/shoe_insole_retrieval_and_packing0515
  default_prompt: "Open the shoe tongue..."
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

The **filename stem is the config name**. There is no `name:` field inside
the YAML; if you put one, it's ignored. `exp_name` is supplied on the CLI
(`--exp-name=...`), not in the YAML.

### Type discriminator + registry

Polymorphic fields (`model`, `data`, `weight_loader`, `lr_schedule`,
`optimizer`) carry a `type:` string that maps to a class via
`src/openpi/training/registry.py`. Adding a new model/data-config class?
Register it there once, and all YAMLs can reference it.

### `.gitignore` rule

```
configs/*.yaml          # default: per-user yamls are gitignored
!configs/_examples/     # but shared templates are tracked
!configs/README.md      # and so is the docs file
```

So `configs/my_experiment.yaml` is private to your machine, but
`configs/_examples/some_shared_template.yaml` is in git.

---

## What changed in the repo

### New files

| File | Lines | Purpose |
|---|---|---|
| `src/openpi/training/registry.py` | ~110 | Maps YAML `type:` strings to Python classes (models, data configs, weight loaders, schedules, optimizers). Single source of truth for what YAML can reference. |
| `src/openpi/training/yaml_loader.py` | ~220 | `load(path)` / `loads(text, name)` / `dump(config)`. Built on OmegaConf. Skips `tyro.MISSING` sentinels and dataclass-default fields to keep YAML compact. Raises clearly when a value can't round-trip (lambdas, `flax.nnx` filters, etc.). |
| `src/openpi/training/yaml_loader_test.py` | ~165 | 10 unit tests: load, dump, round-trip, error handling, file-vs-string parsing. |
| `src/openpi/training/yaml_examples_equivalence_test.py` | ~50 | Parametrized test: every YAML in `configs/_examples/` must load to a `TrainConfig` that is `==` to the corresponding entry in `_CONFIGS_DICT`. Catches drift. |
| `scripts/migrate_configs_to_yaml.py` | ~75 | One-shot tool: dumps every YAML-serializable entry of `_CONFIGS` to `configs/_examples/`. Includes round-trip equality check before writing. Reports skipped configs (those that can't be YAML-ified) with reasons. |
| `configs/README.md` | ~100 | User-facing docs for the YAML system: schema, lookup order, how to add a new task. |
| `configs/_examples/debug_pi05.yaml` | — | Smallest example, used by CI. |
| `configs/_examples/pi05_base_bi_flexiv_*.yaml` × 5 | — | Migrated production configs. |

### Modified files

| File | Change |
|---|---|
| `pyproject.toml` | `+ "omegaconf>=2.3.0"` |
| `.gitignore` | `+5 lines` for the `configs/*.yaml` rule above |
| `src/openpi/training/config.py` | `get_config()` gains YAML fallback (~40 line addition); `_CONFIGS` list is **untouched** — all 17 legacy entries still work via the dict fallback |
| `CLAUDE.md` | Updated "Adding New Robot Support" and "Fine-tuning Workflow" sections to describe YAML workflow |

### Unmodified (intentionally)

- All `DataConfigFactory` subclasses (`LeRobotBiFlexivDataConfig`, etc.)
- The `TrainConfig` dataclass itself
- All policy modules (`bi_flexiv_policy.py`, etc.)
- All model modules (`pi0_config.py`, etc.)
- Downstream scripts (`train.py`, `serve_policy.py`, etc.) — they call
  `get_config(name)` the same way as before

---

## Migration status

The migration script (`scripts/migrate_configs_to_yaml.py`) was run once and
produced 6 YAMLs in `configs/_examples/`. The remaining 11 legacy `_CONFIGS`
entries **cannot** be expressed as YAML and stay in Python:

### Migrated to YAML (6)

- `debug_pi05`
- `pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100`
- `pi05_base_bi_flexiv_newbalacne_shoe_insole_retrieval_and_packing_0515_h100`
- `pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0520_a100`
- `pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100`
- `pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora` — **true LoRA config**,
  uses the auto-derived freeze_filter described below

The `_0422_merged_fixed_h100` config is still in `_CONFIGS` but is not
mirrored to `_examples/`: the name has `_lora` in it but the actual config
is full fine-tuning, which is misleading. New users should not copy it.

### Stay in `_CONFIGS` (11)

| Config | Reason it can't be YAML |
|---|---|
| `pi0_droid` / `pi0_fast_droid` / `pi05_droid` | Use `SimpleDataConfig` with a `data_transforms` **lambda** that closes over `model_config` |
| `pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora` | LoRA: `freeze_filter = nnx.All(PathRegex(...), Not(PathRegex(...)), ...)` — flax filter trees don't serialize |
| `tie_shoes_50_episodes_lora_no_adjust_training_time_rtc_0202` | Same — LoRA freeze_filter |
| `tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100` | Custom `RepackTransform(structure=...)` for ALOHA-format cameras (different from BiFlexiv default) |
| `paligemma_binning_droid` / `paligemma_fast_specialist_droid` / `paligemma_vq_droid` | Carry tokenizer **classes** (not instances) — `<class BinningTokenizer>` etc. — not values |
| `paligemma_fast_droid` / `paligemma_diffusion_droid` | RoboArena-generated; contain lambdas |

These will continue to work via the `_CONFIGS_DICT` fallback indefinitely.
Removing them is **not** a goal of this change.

---

## Testing

```bash
# Unit tests for the loader itself
pytest src/openpi/training/yaml_loader_test.py

# Equivalence: every shipped YAML must == its _CONFIGS counterpart
pytest src/openpi/training/yaml_examples_equivalence_test.py

# Run both at once
pytest src/openpi/training/yaml_loader_test.py \
       src/openpi/training/yaml_examples_equivalence_test.py
```

Both files combined: **17 tests pass.**

The equivalence test is the most important one — it pins YAML output to the
original Python dataclass via `==`. If anyone edits `config.py` and forgets
to regenerate the YAMLs (or vice versa), CI fails with a pointer to
`scripts/migrate_configs_to_yaml.py --overwrite`.

---

## Verifying the change end-to-end

The branch was developed and tested on a CPU-only workstation (no GPU
training) but the integration points are verified:

```python
import openpi.training.config as c

# 1) YAML-loaded config is dataclass-equal to its _CONFIGS source
assert c.get_config("debug_pi05") == c._CONFIGS_DICT["debug_pi05"]
# ✓ PASS

# 2) BiFlexiv config loads from YAML, has correct fields
cfg = c.get_config("pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100")
assert cfg.data.repo_id == "Xense/shoe_insole_retrieval_and_packing0515"
assert cfg.num_train_steps == 40_000
assert cfg.fsdp_devices == 8
# ✓ PASS

# 3) Legacy lambda config still works via _CONFIGS fallback
cfg = c.get_config("pi0_droid")
assert cfg.name == "pi0_droid"
# ✓ PASS (loaded from _CONFIGS_DICT, not YAML)

# 4) Did-you-mean for misspellings
c.get_config("debug_pi006")
# → ValueError: Config 'debug_pi006' not found. Did you mean 'debug_pi05'?
# ✓ PASS
```

Downstream scripts (`compute_norm_stats.py`, `train.py`, `serve_policy.py`)
don't need any changes — they call `_config.get_config(name)` and now
transparently receive YAML-backed configs.

---

## Workflow comparison

### Adding a new task (before)

1. Edit `src/openpi/training/config.py` (919 lines, central file)
2. Append a 25–40 line `TrainConfig(...)` block to `_CONFIGS`
3. Open PR → **merge conflict with whoever else added a task this week**
4. Resolve conflict (often a rebase / re-apply)
5. Push → CI → merge

### Adding a new task (after)

1. Copy `configs/_examples/<closest>.yaml` to `configs/<my_task>.yaml`
2. Edit 3–5 lines (`repo_id`, `prompt`, maybe `num_train_steps`)
3. Run training. **No PR needed at all** for a private experiment.

If you want to share the config with the team later:

1. Move it to `configs/_examples/<my_task>.yaml`
2. **Read it once** to make sure no machine-local paths leaked in
3. PR (clean — only adds one new file, never touches the central registry)

---

## Open issues / future work

- **`weight_loader.params_path` is the main place machine-local paths leak in.**
  The earbuds config already has `/home/li/hubo/...` in it (preserved from the
  Python original — this branch doesn't change content, only format). The
  current mitigation is "PR review catches it." If that proves insufficient,
  we could add an OmegaConf env-var interpolation: `${env:PI05_BASE_CKPT:gs://...}`.
  Not done in this PR.

- **LoRA configs now work in YAML.** The follow-up change auto-derives
  `freeze_filter` from `model.paligemma_variant` / `model.action_expert_variant`
  when either contains `lora`. Internally it calls `Pi0Config.get_freeze_filter()`,
  which is exactly what every `_CONFIGS` LoRA entry did by hand. YAML authors
  just set the variant strings and `yaml_loader` fills in the rest. Configs
  whose freeze pattern doesn't follow the variant-string convention (rare)
  still need to stay in `_CONFIGS`.

- **No CLI-driven `train.py @configs/foo.yaml` yet.** Today you load YAML by
  passing the bare name (`train.py shoe_insole_...`), which works because
  `get_config` finds the file by stem. A more explicit `@path/to/file.yaml`
  syntax is feasible but wasn't required for this iteration.
