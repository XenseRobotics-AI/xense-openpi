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
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
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

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05_TACTILE if self.pi05 else _model.ModelType.PI0_TACTILE

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TactileFastVit":
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
