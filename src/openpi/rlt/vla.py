"""The frozen VLA behind RLT: its config, checkpoint-owned preprocessing and identity."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import jax.numpy as jnp

from openpi.models import pi0_config
import openpi.models.model as _model
from openpi.rlt import config as _rlt_config
from openpi.rlt import prefix_cache
from openpi.shared import download
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class FrozenVLA:
    train_config: _config.TrainConfig
    checkpoint_dir: pathlib.Path
    # Data pipeline with the norm stats bundled in the checkpoint - the ones serving uses.
    data_config: _config.DataConfig

    def load_model(self) -> _model.BaseModel:
        """Load the VLA for inference, as ``policy_config.create_trained_policy`` does."""
        model_config = self.train_config.model
        if isinstance(model_config, pi0_config.Pi0Config) and model_config.use_cudnn_attention:
            # A training-only optimization that JAX 0.5.3 only supports on Hopper; params are unaffected.
            model_config = dataclasses.replace(model_config, use_cudnn_attention=False)
        return model_config.load(_model.restore_params(self.checkpoint_dir / "params", dtype=jnp.bfloat16))

    def input_transforms(self, default_prompt: str | None = None) -> _transforms.DataTransformFn:
        """Robot observation -> model inputs; the serving policy's input pipeline."""
        data_config = self.data_config
        return _transforms.compose(
            [
                _transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )

    def output_transforms(self) -> _transforms.DataTransformFn:
        """Model outputs -> absolute robot actions; the serving policy's output pipeline."""
        data_config = self.data_config
        return _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )

    def vla_identity(self) -> dict[str, Any]:
        """The VLA's content identity: its config, weights and bundled norm stats."""
        return {
            "vla_config": self.train_config.name,
            "vla_checkpoint": str(self.checkpoint_dir),
            "params_fingerprint": prefix_cache.directory_fingerprint(self.checkpoint_dir / "params"),
            "norm_stats_fingerprint": prefix_cache.directory_fingerprint(
                self.checkpoint_dir / "assets" / self.data_config.asset_id
            ),
        }

    def cache_identity(self, token_training: _rlt_config.TokenTrainingConfig) -> dict[str, Any]:
        """What a prefix cache must have been built from to serve this config."""
        return {**self.vla_identity(), "repo_id": self.data_config.repo_id, "frame_stride": token_training.frame_stride}


def resolve(token_training: _rlt_config.TokenTrainingConfig) -> FrozenVLA:
    """Resolve the VLA config and checkpoint named by ``token_training``.

    Norm stats are loaded strictly from the checkpoint's ``assets/``, never the
    mutable repo-level assets tree: pi05 renders the *normalized* state into the
    prompt, so other stats would feed the frozen VLA a prefix it never saw.
    """
    train_config = _config.get_config(token_training.vla_config)
    if token_training.repo_id is not None:
        train_config = dataclasses.replace(
            train_config, data=dataclasses.replace(train_config.data, repo_id=token_training.repo_id)
        )
    checkpoint_dir = pathlib.Path(download.maybe_download(token_training.vla_checkpoint)).resolve()
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"{checkpoint_dir} is not an openpi checkpoint step dir (no params/).")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError(f"{train_config.name} has no asset id; cannot load its norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return FrozenVLA(train_config, checkpoint_dir, dataclasses.replace(data_config, norm_stats=norm_stats))
