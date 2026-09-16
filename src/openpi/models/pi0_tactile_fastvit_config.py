"""Config for Pi0/Pi05 + tactile FastViT-T12 encoder.

This config injects 4 tactile-image embeddings into ``embed_suffix``. It keeps
all of the base Pi0 features (RTC training, adaRMS for pi05, etc.) and only adds
fields related to the tactile branch.

The actual encoder is selected by name through the ``tactile_encoders`` registry,
so swapping FastViT for a different vision backbone in the future does not
require touching ``Pi0TactileFastVit`` or ``pi0.py``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, override

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_tactile_fastvit import Pi0TactileFastVit


@dataclasses.dataclass(frozen=True)
class Pi0TactileFastVitConfig(pi0_config.Pi0Config):
    """Pi0Config + 4-image tactile branch.

    Attributes:
        tactile_encoder_name: Registry key for the encoder (default ``"fastvit_t12"``).
        tactile_pretrained_path: Optional path to a Flax-format weight file
            (``params.safetensors``) produced by
            ``scripts/convert_fastvit_torch_to_flax.py``. ``None`` -> train from
            scratch (rare).
        tactile_image_keys: Names of tactile images in the ``Observation.images``
            dict. Order matches the suffix token order.
    """

    tactile_encoder_name: str = "fastvit_t12"
    tactile_pretrained_path: str | None = None
    tactile_image_keys: tuple[str, ...] = (
        "tactile_0_rgb",
        "tactile_1_rgb",
        "tactile_2_rgb",
        "tactile_3_rgb",
    )
    # Compute dtype for the tactile encoder's conv/BN/dense ops. Parameters
    # are always stored in fp32 (the optimizer master copy); only forward
    # activations and tensor-core matmuls run in this dtype. ``"bfloat16"``
    # roughly doubles encoder throughput on H100/A100 with no measurable
    # quality loss for FastViT-T12.
    tactile_compute_dtype: str = "bfloat16"

    # ---- Latent Tactile Predictor (LTP), training-only auxiliary head ----
    # docs/action-conditioned-tactile-pretraining.md section 2.3. ``None`` disables the
    # head entirely: no parameters are created and compute_loss is bit-identical to
    # the plain tactile model. Otherwise this is ``m``, the 1-based index of the action
    # expert block whose output residual stream the head reads (RATG: 5-9 of 18).
    tactile_future_layer: int | None = None
    # Frame offsets of the future tactile frames to predict (30 fps -> 0.33 s .. 1.67 s).
    # Must match the label store the data config injects.
    tactile_future_horizons: tuple[int, ...] = (10, 20, 30, 40, 50)
    # Dimension of the target per (horizon, pad): 256 for the PCA-whitened FastViT
    # latent, 16*16*3 = 768 for the pixel-field control.
    tactile_future_dim: int = 256
    tactile_future_num_heads: int = 8
    tactile_future_head_dim: int = 128
    tactile_future_mlp_dim: int = 4096
    # Which suffix positions the head may attend to. "action": only the action
    # tokens (the main design -- the head cannot copy the tactile tokens, so the
    # information has to travel tactile -> action-token stream). "all": every suffix
    # token including the tactile ones (the read-out control).
    tactile_future_kv: Literal["action", "all"] = "action"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tactile_future_layer is not None:
            depth = _gemma.get_config(self.action_expert_variant).depth
            if not 1 <= self.tactile_future_layer <= depth:
                raise ValueError(
                    f"tactile_future_layer must be in [1, {depth}] for {self.action_expert_variant}, "
                    f"got {self.tactile_future_layer}"
                )
            if not self.tactile_future_horizons:
                raise ValueError("tactile_future_horizons must not be empty when the LTP head is enabled")
            if self.enable_training_time_rtc:
                raise ValueError("the LTP head is not supported together with training-time RTC")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05_TACTILE if self.pi05 else _model.ModelType.PI0_TACTILE

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0TactileFastVit:
        from openpi.models.pi0_tactile_fastvit import Pi0TactileFastVit

        return Pi0TactileFastVit(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        images = {
            "base_0_rgb": image_spec,
            "left_wrist_0_rgb": image_spec,
            "right_wrist_0_rgb": image_spec,
        }
        image_masks = {
            "base_0_rgb": image_mask_spec,
            "left_wrist_0_rgb": image_mask_spec,
            "right_wrist_0_rgb": image_mask_spec,
        }
        for key in self.tactile_image_keys:
            images[key] = image_spec
            image_masks[key] = image_mask_spec

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=images,
                image_masks=image_masks,
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
