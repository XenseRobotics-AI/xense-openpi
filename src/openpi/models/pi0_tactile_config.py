"""Pi0 model configuration with tactile image support.

This extends the base Pi0 model to support additional tactile image inputs
while preserving the ability to load pretrained weights for the visual branch.
"""

import dataclasses
from typing import TYPE_CHECKING, override

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_tactile import Pi0Tactile


@dataclasses.dataclass(frozen=True)
class Pi0TactileConfig(_model.BaseModelConfig):
    """Extended Pi0 config that supports tactile images.

    This config adds support for tactile images while maintaining compatibility
    with pretrained Pi0/Pi0.5 weights for the visual branch.

    Architecture:
    - Visual branch (3 cameras): Uses pretrained SigLIP encoder
    - Tactile branch (2 sensors): Uses new SigLIP encoder (trained from scratch or with LoRA)
    - Both branches produce tokens that are concatenated before the LLM
    """

    # Whether to freeze the visual encoder (recommended when using pretrained weights)
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05_TACTILE
        return _model.ModelType.PI0_TACTILE

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Tactile":
        from openpi.models.pi0_tactile import Pi0Tactile

        return Pi0Tactile(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """Defines input spec with 5 images: 3 visual + 2 tactile."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    # Visual images (compatible with pretrained weights)
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                    # Tactile images (new)
                    "left_tactile_0_rgb": image_spec,
                    "right_tactile_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                    "left_tactile_0_rgb": image_mask_spec,
                    "right_tactile_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns freeze filter that handles both visual and tactile encoders.

        Strategy:
        - Visual encoder: Frozen or LoRA (based on parent config)
        - Tactile encoder: Always trainable (new parameters)
        - LLM: Follow parent config (LoRA or frozen)
        """
        filters = []
        has_lora = False

        # Get base filters from parent class
        base_filter = super().get_freeze_filter()
        if base_filter != nnx.Nothing:
            filters.append(base_filter)

        # Freeze visual encoder if requested
        if self.freeze_visual_encoder:
            filters.append(nnx_utils.PathRegex(".*PaliGemma.*img.*"))
            # But don't freeze tactile encoder
            filters.append(nnx.Not(nnx_utils.PathRegex(".*tactile_img.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
