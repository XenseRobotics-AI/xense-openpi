"""Pi0/Pi05 variant that consumes 4 tactile images via a FastViT-T12 encoder.

The class subclasses ``Pi0`` and only changes these things:

* ``__init__`` instantiates the tactile encoder (via the registry) and a linear
  projection to the action-expert width.
* ``embed_prefix`` filters tactile keys out of ``obs.images`` so they do *not*
  go through PaliGemma/SigLIP. Tactile is the FastViT branch only.
* ``embed_suffix`` prepends 4 tactile tokens to the existing suffix tokens
  (``state`` + ``action+time``). Tactile features do not participate in
  ``adarms_cond``.

A small ``_preprocess_observation`` override swaps in the tactile-aware
preprocess so that the 4 tactile keys are correctly augmented during training.

Optionally (``config.tactile_future_layer`` set) the model also carries a
training-only Latent Tactile Predictor head, ``tactile_future_head``, that reads
block ``m``'s residual stream at the action-token positions and regresses the
future tactile latents supplied in ``Observation.aux_targets``. ``compute_loss``
then returns a dict of losses instead of one array; see ``_compute_loss_with_ltp``.
Inference is untouched.
"""

from __future__ import annotations

import dataclasses
import itertools

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_tactile_fastvit_config
from openpi.models import tactile_future_head as _ltp
import openpi.models.gemma as _gemma
from openpi.models.tactile_encoders import build_tactile_encoder
from openpi.shared import array_typing as at

# Names of the entries the LTP head reads from ``Observation.aux_targets``.
FUTURE_TACTILE_Z = "future_tactile_z"
FUTURE_TACTILE_MASK = "future_tactile_mask"
# Edges of the flow-matching time bins that the LTP loss is reported by. Low tau =
# nearly clean actions in the suffix (action-conditioned prediction); high tau =
# noise (the head can only use what the action tokens pulled in from prefix/tactile).
TAC_TIME_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


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

        # ---- Latent Tactile Predictor (training-only) ----
        self._tactile_future_layer = config.tactile_future_layer
        self._tactile_future_kv = config.tactile_future_kv
        self._num_future_horizons = len(config.tactile_future_horizons)
        if config.tactile_future_layer is not None:
            self.tactile_future_head = _ltp.TactileFuturePredictor(
                width=action_expert_width,
                num_horizons=self._num_future_horizons,
                num_pads=self._num_tactile,
                out_dim=config.tactile_future_dim,
                num_heads=config.tactile_future_num_heads,
                head_dim=config.tactile_future_head_dim,
                mlp_dim=config.tactile_future_mlp_dim,
                rngs=rngs,
            )

    @property
    def has_tactile_future_head(self) -> bool:
        return self._tactile_future_layer is not None

    def _preprocess_observation(self, rng, observation, *, train):
        return _model.preprocess_observation_tactile(
            rng,
            observation,
            train=train,
            image_keys=_model.IMAGE_KEYS_TACTILE_4,
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # Tactile images are handled exclusively by the FastViT branch in
        # embed_suffix. Hide them from the base implementation so PaliGemma /
        # SigLIP only sees the 3 camera views; otherwise tactile gets encoded
        # twice and adds ~1024 prefix image tokens (256 per tactile view),
        # which dominates the LLM cost.
        filtered_images = {k: v for k, v in obs.images.items() if k not in self._tactile_keys}
        filtered_masks = {k: v for k, v in obs.image_masks.items() if k not in self._tactile_keys}
        filtered_obs = dataclasses.replace(obs, images=filtered_images, image_masks=filtered_masks)
        return super().embed_prefix(filtered_obs)

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
            tactile_imgs = jnp.stack([obs.images[key] for key in self._tactile_keys], axis=1)  # (b, N, h, w, 3)
            tactile_mask = jnp.stack([obs.image_masks[key] for key in self._tactile_keys], axis=1)  # (b, N)
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
            base_tokens, base_mask, base_ar, adarms_cond = super().embed_suffix(obs, noisy_actions, timestep)

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

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"] | dict[str, at.Array]:
        """Flow loss, plus the LTP loss when the head is enabled.

        Without the head this is exactly ``Pi0.compute_loss`` (one ``[b, ah]`` array).
        With it, returns a dict::

            flow        [b, ah]    per-token flow-matching loss (the main objective)
            tac         [b, K, S]  per-(sample, horizon, pad) LTP squared error, mean over Z
            tac_mask    [b, K, S]  which of those entries have a valid future frame
            tac_by_time [n_bins]   masked mean of ``tac`` per flow-time bin (NaN if empty)

        ``scripts/train.py`` combines ``flow`` and ``tac`` with the configured weight.
        """
        if not self.has_tactile_future_head:
            return super().compute_loss(rng, observation, actions, train=train)
        if observation.aux_targets is None or FUTURE_TACTILE_Z not in observation.aux_targets:
            raise ValueError(
                "tactile_future_layer is set but the batch carries no "
                f"aux_targets[{FUTURE_TACTILE_Z!r}]. Add InjectTactileFutureLabels to the data config "
                "(LeRobotBiFlexivTactileDataConfig.tactile_future_labels_path) or disable the head."
            )
        preprocess_rng, loss_rng = jax.random.split(rng)
        with jax.named_scope("loss/preprocess"):
            observation = self._preprocess_observation(preprocess_rng, observation, train=train)
        return self._compute_loss_with_ltp(loss_rng, observation, actions)

    def _compute_loss_with_ltp(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> dict[str, at.Array]:
        out = self._flow_forward(rng, observation, actions, return_suffix_hidden=True)
        suffix_hidden = out["suffix_hidden"]  # [depth, b, s, d]
        suffix_mask = out["suffix_mask"]  # [b, s]

        # h^(m): the output residual stream of block m (1-based), before block m+1's norm.
        with jax.named_scope("loss/ltp/select_layer"):
            hidden = suffix_hidden[self._tactile_future_layer - 1]
            if self._tactile_future_kv == "action":
                # The tactile tokens are the first ``num_tactile`` suffix positions; the
                # action tokens are the last ``action_horizon``. Only the latter may be
                # read so that the only route to the future is tactile -> action stream.
                hidden = hidden[:, -self.action_horizon :]
                key_mask = suffix_mask[:, -self.action_horizon :]
            else:
                key_mask = suffix_mask

        with jax.named_scope("loss/ltp/head"):
            z_hat = self.tactile_future_head(hidden, key_mask)  # [b, K, S, Z]

        with jax.named_scope("loss/ltp/target"):
            targets = observation.aux_targets
            z = jax.lax.stop_gradient(jnp.asarray(targets[FUTURE_TACTILE_Z]).astype(jnp.float32))
            valid = jnp.asarray(targets[FUTURE_TACTILE_MASK]).astype(jnp.bool_)  # [b, K]
            if z.shape != z_hat.shape:
                raise ValueError(
                    f"future_tactile_z has shape {z.shape}, but the LTP head predicts {z_hat.shape}. Check "
                    "tactile_future_horizons / tactile_future_dim against the label store."
                )
            if valid.shape != z.shape[:2]:
                raise ValueError(f"future_tactile_mask must be [b, K]={z.shape[:2]}, got {valid.shape}")
            # Mean over Z, so that a zero prediction of a whitened target scores exactly 1.
            tac = jnp.mean(jnp.square(z_hat - z), axis=-1)  # [b, K, S]
            tac_mask = jnp.broadcast_to(valid[:, :, None], tac.shape)
            tac_by_time = _masked_mean_by_bins(tac, tac_mask, out["time"], TAC_TIME_BIN_EDGES)

        return {"flow": out["loss"], "tac": tac, "tac_mask": tac_mask, "tac_by_time": tac_by_time}


def _masked_mean_by_bins(values: at.Array, mask: at.Array, time: at.Array, edges: tuple[float, ...]) -> at.Array:
    """Masked mean of ``values`` [b, ...] over the samples whose ``time`` [b] falls in each ``(lo, hi]``.

    Returns one entry per bin (``len(edges) - 1``), NaN where a bin has no valid entry.
    """
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1).astype(jnp.float32)
    means = []
    for lo, hi in itertools.pairwise(edges):
        in_bin = ((time > lo) & (time <= hi)).astype(jnp.float32)[:, None]
        weight = flat_mask * in_bin
        count = jnp.sum(weight)
        means.append(jnp.where(count > 0, jnp.sum(flat_values * weight) / jnp.maximum(count, 1.0), jnp.nan))
    return jnp.stack(means)


__all__ = ["FUTURE_TACTILE_MASK", "FUTURE_TACTILE_Z", "TAC_TIME_BIN_EDGES", "Pi0TactileFastVit"]
