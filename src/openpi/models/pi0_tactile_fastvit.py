"""Pi0/Pi05 variant that consumes 4 tactile images via a FastViT-T12 encoder.

The class subclasses ``Pi0`` and only changes two things:

* ``__init__`` instantiates the tactile encoder (via the registry) and a linear
  projection to the action-expert width.
* ``embed_suffix`` prepends 4 tactile tokens to the existing suffix tokens
  (``state`` + ``action+time``). Tactile features do not participate in
  ``adarms_cond``.

A small ``_preprocess_observation`` override swaps in the tactile-aware
preprocess so that the 4 tactile keys are correctly augmented during training.
"""

from __future__ import annotations

import einops
import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_tactile_fastvit_config
from openpi.models.tactile_encoders import build_tactile_encoder
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at


class Pi0TactileFastVit(pi0.Pi0):
    """Pi0/Pi05 with 4 tactile-image tokens injected into the suffix."""

    def __init__(self, config: pi0_tactile_fastvit_config.Pi0TactileFastVitConfig, rngs: nnx.Rngs) -> None:
        super().__init__(config, rngs=rngs)

        self.tactile_encoder = build_tactile_encoder(
            config.tactile_encoder_name,
            rngs=rngs,
            pretrained_path=config.tactile_pretrained_path,
        )

        action_expert_width = _gemma.get_config(config.action_expert_variant).width
        self.tactile_proj = nnx.Linear(
            self.tactile_encoder.feature_dim,
            action_expert_width,
            rngs=rngs,
        )

        self._tactile_keys: tuple[str, ...] = tuple(config.tactile_image_keys)
        self._num_tactile = len(self._tactile_keys)

    def _preprocess_observation(self, rng, observation, *, train):  # noqa: D401
        return _model.preprocess_observation_tactile(
            rng,
            observation,
            train=train,
            image_keys=_model.IMAGE_KEYS_TACTILE_4,
        )

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b s emb"] | None,
    ]:
        tactile_tokens: list = []
        tactile_mask_parts: list = []
        for key in self._tactile_keys:
            feat = self.tactile_encoder(obs.images[key])  # (b, feat_dim)
            feat = self.tactile_proj(feat)  # (b, action_expert_width)
            tactile_tokens.append(feat[:, None, :])
            tactile_mask_parts.append(
                einops.repeat(obs.image_masks[key], "b -> b s", s=1)
            )
        tactile_tokens_arr = jnp.concatenate(tactile_tokens, axis=1)  # (b, N, w)
        tactile_mask = jnp.concatenate(tactile_mask_parts, axis=1)  # (b, N)
        # tactile block: first token is a block boundary (cannot peek prefix),
        # the remaining tactile tokens are co-visible within the block.
        tactile_ar = jnp.asarray([True] + [False] * (self._num_tactile - 1))

        base_tokens, base_mask, base_ar, adarms_cond = super().embed_suffix(
            obs, noisy_actions, timestep
        )

        tokens = jnp.concatenate([tactile_tokens_arr, base_tokens], axis=1)
        input_mask = jnp.concatenate([tactile_mask, base_mask], axis=1)
        ar_mask = jnp.concatenate([tactile_ar, base_ar], axis=0)
        return tokens, input_mask, ar_mask, adarms_cond


__all__ = ["Pi0TactileFastVit"]
