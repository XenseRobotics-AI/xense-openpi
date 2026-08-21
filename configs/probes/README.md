# Tactile counterfactual probe configs

These YAML files drive `scripts/tactile_counterfactual_probe.py` (offline
tactile counterfactual inference diagnostic for `Pi0TactileFastVit`
checkpoints). See `test/tactile_counterfactual/` for the implementation.

## water_weight_counterfactual.yaml

500 explicit pairs generated from `sampled_frames.yaml` by
`scripts/build_counterfactual_config.py`:

- `full`  <- `categories.water[i]`     (bottle placed in the far bin)
- `empty` <- `categories.no_water[i]`  (bottle placed in the near bin)

Uses `model.config_name` to resolve the TrainConfig. **Note:** the
`pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100` TrainConfig that
produced the checkpoint is not present in this repo checkout — either add it
(see todolist section 11) or use the inline variant below. Also edit
`model.checkpoint_dir` to the actual checkpoint step directory.

## water_weight_counterfactual_inline.yaml

Identical 500 pairs, but the model config (gemma_2b / gemma_300m, pi05, RTC,
FastViT-T12) and the data config (`LeRobotBiFlexivTactileDataConfig` on
`Xense/bottle-sorting-0810`) are inlined, so no TrainConfig lookup is needed.

## Regenerating the pairs

    python scripts/build_counterfactual_config.py \
        --samples sampled_frames.yaml \
        --output configs/probes/water_weight_counterfactual.yaml \
        --config-name pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100 \
        --checkpoint-dir checkpoints/<name>/<exp>/<step>

## Running

    # 2-pair test with full validation gates (equivalence + sanity checks)
    python scripts/tactile_counterfactual_probe.py \
        --config configs/probes/water_weight_counterfactual_inline.yaml \
        --max-pairs 2 --dry-run

    # full 500 pairs
    python scripts/tactile_counterfactual_probe.py \
        --config configs/probes/water_weight_counterfactual_inline.yaml
