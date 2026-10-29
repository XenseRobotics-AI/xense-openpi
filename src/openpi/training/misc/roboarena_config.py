"""RoboArena baseline policy configs.

These stay in Python rather than moving to `configs/*.yaml`: each one passes a
tokenizer *class* (`fast_model_tokenizer=BinningTokenizer`) as a config value,
which cannot be serialized, and they are authored as a family rather than one
file per task. Everything else lives in `configs/` - see `configs/README.md`.
"""

import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.training.optimizer as _optimizer


def get_roboarena_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import DroidInferenceDataConfig
    from openpi.training.config import TrainConfig

    return [
        #
        # RoboArena DROID baseline inference configs.
        #
        TrainConfig(
            # Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
            name="paligemma_binning_droid",
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                max_token_len=400,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=DroidInferenceDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
            name="paligemma_fast_droid",
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=15),
            data=DroidInferenceDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
            name="paligemma_fast_specialist_droid",
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FASTTokenizer,
                fast_model_tokenizer_kwargs={"fast_tokenizer_path": "KarlP/fast_droid_specialist"},
            ),
            data=DroidInferenceDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FSQ tokenizer.
            name="paligemma_vq_droid",
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FSQTokenizer,
                fast_model_tokenizer_kwargs={"fsq_tokenizer_path": "gs://openpi-assets/tokenizers/droid_fsq_tokenizer"},
            ),
            data=DroidInferenceDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
            name="paligemma_diffusion_droid",
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            model=pi0_config.Pi0Config(action_horizon=10, action_dim=8),
            data=DroidInferenceDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
    ]
