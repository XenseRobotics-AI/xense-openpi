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

import flax.nnx as nnx
import jax
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

        # Resolve "bfloat16"/"float32"/"float16" strings to a JAX dtype so the
        # encoder's conv/BN/dense ops run in the requested compute precision.
        compute_dtype = jnp.dtype(config.tactile_compute_dtype)
        self.tactile_encoder = build_tactile_encoder(
            config.tactile_encoder_name,
            rngs=rngs,
            pretrained_path=config.tactile_pretrained_path,
            compute_dtype=compute_dtype,
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
        # Stack the 4 tactile images on a new axis and fold it into the batch
        # dim so the FastViT encoder runs as ONE call at batch B*N instead of
        # N sequential calls at batch B. BatchNorm uses frozen running stats
        # (use_running_average=True) and every other op in FastViT is
        # batch-invariant, so this is exactly equivalent to the per-key loop
        # but lets XLA fuse a single graph and gives the small depthwise convs
        # a much larger effective batch — the dominant win on H100.
        with jax.named_scope("suffix/tactile/stack"):
            tactile_imgs = jnp.stack(
                [obs.images[key] for key in self._tactile_keys], axis=1
            )  # (b, N, h, w, 3)
            tactile_mask = jnp.stack(
                [obs.image_masks[key] for key in self._tactile_keys], axis=1
            )  # (b, N)
            b, n, h, w, c = tactile_imgs.shape
        with jax.named_scope("suffix/tactile/fastvit"):
            feats = self.tactile_encoder(tactile_imgs.reshape(b * n, h, w, c))  # (b*n, feat)
        with jax.named_scope("suffix/tactile/proj"):
            feats = self.tactile_proj(feats)  # (b*n, action_expert_width)
            tactile_tokens_arr = feats.reshape(b, n, -1)  # (b, N, w)
        # tactile block: first token is a block boundary (cannot peek prefix),
        # the remaining tactile tokens are co-visible within the block.
        tactile_ar = jnp.asarray([True] + [False] * (self._num_tactile - 1))

        with jax.named_scope("suffix/base"):
            base_tokens, base_mask, base_ar, adarms_cond = super().embed_suffix(
                obs, noisy_actions, timestep
            )

        with jax.named_scope("suffix/concat"):
            tokens = jnp.concatenate([tactile_tokens_arr, base_tokens], axis=1)
            input_mask = jnp.concatenate([tactile_mask, base_mask], axis=1)
            ar_mask = jnp.concatenate([tactile_ar, base_ar], axis=0)

            # Per-token adarms_cond (training-time RTC path produces shape (b, ah, emb))
            # must be padded to match the new suffix length so RMSNorm's element-wise
            # modulation broadcasts correctly. Zero rows for tactile positions implement
            # the "tactile does not participate in adaRMS" design from the integration
            # doc -- at init the zero-init Dense yields a no-op; only the shared learned
            # bias can leak in, which matches the design intent. The (b, emb) and None
            # cases broadcast over all suffix tokens naturally and need no change.
            if adarms_cond is not None and adarms_cond.ndim == 3:
                tactile_cond = jnp.zeros(
                    (adarms_cond.shape[0], self._num_tactile, adarms_cond.shape[-1]),
                    dtype=adarms_cond.dtype,
                )
                adarms_cond = jnp.concatenate([tactile_cond, adarms_cond], axis=1)

        return tokens, input_mask, ar_mask, adarms_cond


__all__ = ["Pi0TactileFastVit"]
