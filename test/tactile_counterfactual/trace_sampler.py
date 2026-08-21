"""Trace-equivalent copy of the production sampling loop.

``trace_sample_actions`` re-implements the exact math of
``Pi0.sample_actions`` (inference_mode="standard") and
``Pi0.training_time_rtc_sample_actions`` (inference_mode="rtc", with the
first-inference dummy prefix / inference_delay=0, which is what the deployment
policy uses) while additionally capturing every intermediate quantity:

* FastViT features and tactile tokens (before the suffix concatenation),
* per-step suffix tokens / masks / adarms_cond,
* per-step transformer action hidden states ``suffix_out[:, -ah:, :]``,
* per-step velocity predictions ``v_t = action_out_proj(action_hidden)``,
* the full flow trajectory ``x_t_before -> v_t -> x_t_after`` with
  ``x_t_after = x_t_before + dt * v_t``.

The production sampler is *not* modified. Before an experiment is allowed to
run, ``verify_trace_equivalence`` must confirm that the traced final action is
allclose to the production sampler's output for the same fixed noise.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_tactile_fastvit import Pi0TactileFastVit

logger = logging.getLogger("tactile_counterfactual")

# Default tolerance for trace-vs-production equivalence. The trace runs the
# same op sequence as the production loop, so the difference is expected to be
# ~1e-6 (bfloat16 suffix activations) at most.
EQ_RTOL = 1e-4
EQ_ATOL = 1e-5


@dataclasses.dataclass
class StepTrace:
    step: int
    timestep: jax.Array  # scalar time *before* this update
    x_t_before: jax.Array  # (b, ah, ad) — after RTC prefix re-freeze if rtc
    time_masked: jax.Array | None  # (b, ah) per-action timestep for rtc, else None
    suffix_tokens: jax.Array  # (b, s, emb)
    suffix_input_mask: jax.Array  # (b, s)
    suffix_ar_mask: jax.Array  # (s,)
    adarms_cond: jax.Array | None  # (b, emb) | (b, s, emb) | None
    action_hidden: jax.Array  # (b, ah, emb)
    v_t: jax.Array  # (b, ah, ad)
    x_t_after: jax.Array  # (b, ah, ad) = x_t_before + dt * v_t (prefix frozen in rtc)


@dataclasses.dataclass
class ActionTrace:
    final_action: jax.Array  # (b, ah, ad)
    fastvit_features: jax.Array  # (b, N, fastvit_dim)
    tactile_tokens: jax.Array  # (b, N, action_expert_width) — tactile_proj output
    tactile_tokens_from_suffix: jax.Array  # first step suffix[:, :N] — must allclose tactile_tokens
    tactile_mask: jax.Array  # (b, N) bool
    steps: list[StepTrace]
    inference_mode: str
    num_steps: int


class TraceSampler:
    """JIT-compiled trace sampler. Build once, reuse across all probe runs.

    The module parameters are frozen at construction time (``nnx.split``), the
    same way ``openpi.shared.nnx_utils.module_jit`` freezes them for the
    production policy. All probe runs share the compiled prefix/step graphs.
    """

    def __init__(
        self,
        model: Pi0TactileFastVit,
        tactile_keys: Sequence[str],
        inference_mode: str = "rtc",
    ) -> None:
        if not isinstance(model, Pi0TactileFastVit):
            raise TypeError(
                f"TraceSampler requires a Pi0TactileFastVit model, got {type(model).__name__}. "
                "The probe is only defined for models with a tactile branch."
            )
        if inference_mode not in ("standard", "rtc"):
            raise ValueError(f"inference_mode must be 'standard' or 'rtc', got {inference_mode!r}")
        self._model = model
        self._tactile_keys = tuple(tactile_keys)
        self._inference_mode = inference_mode
        self._num_tactile = len(self._tactile_keys)

        graphdef, state = nnx.split(model)
        self._graphdef = graphdef
        self._state = state

        self._tactile_fn = jax.jit(self._tactile_fun)
        self._prefix_fn = jax.jit(self._prefix_fun)
        self._step_fn = jax.jit(self._step_fun)

    # ------------------------------------------------------------------ #
    # JIT-internal implementations (module state passed explicitly)      #
    # ------------------------------------------------------------------ #

    def _tactile_fun(self, state, obs: _model.Observation):
        """Manual reproduction of the FastViT branch of Pi0TactileFastVit.embed_suffix.

        Returns (fastvit_features, tactile_tokens, tactile_mask); the tokens
        must be allclose to the first ``N`` tokens returned by ``embed_suffix``
        in the first denoising step (verified by the runner's sanity checks).
        """
        module = nnx.merge(self._graphdef, state)
        imgs = jnp.stack([obs.images[key] for key in self._tactile_keys], axis=1)  # (b, N, h, w, 3)
        tactile_mask = jnp.stack([obs.image_masks[key] for key in self._tactile_keys], axis=1)  # (b, N)
        b, n, h, w, c = imgs.shape
        feats = module.tactile_encoder(imgs.reshape(b * n, h, w, c))  # (b*n, fastvit_dim)
        fastvit_features = feats.reshape(b, n, -1)  # (b, N, fastvit_dim)
        tactile_tokens = module.tactile_proj(feats).reshape(b, n, -1)  # (b, N, action_expert_width)
        return fastvit_features, tactile_tokens, tactile_mask

    def _prefix_fun(self, state, obs: _model.Observation):
        """Prefix forward pass; returns (kv_cache, prefix_mask)."""
        module = nnx.merge(self._graphdef, state)
        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = module.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return kv_cache, prefix_mask

    def _step_fun(
        self,
        state,
        kv_cache,
        obs: _model.Observation,
        prefix_mask,
        x_t,
        time,
        action_prefix,
        action_prefix_mask,
        dt,
    ):
        """One denoising step, line-for-line equivalent to the production loop."""
        module = nnx.merge(self._graphdef, state)
        batch_size = obs.state.shape[0]

        if self._inference_mode == "rtc":
            # Re-freeze the action prefix before every denoising step
            # (production: training_time_rtc_sample_actions).
            x_t = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_t)
            time_masked = jnp.where(action_prefix_mask, 0.0, time)
        else:
            time_masked = None

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = module.embed_suffix(
            obs,
            x_t,
            time_masked if self._inference_mode == "rtc" else jnp.broadcast_to(time, batch_size),
        )

        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = module.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=pos,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        action_hidden = suffix_out[:, -module.action_horizon :]
        v_t = module.action_out_proj(action_hidden)

        if self._inference_mode == "rtc":
            x_t_after = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_t + dt * v_t)
        else:
            x_t_after = x_t + dt * v_t

        return (
            x_t_after,
            time + dt,
            (
                x_t,
                time_masked,
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
                action_hidden,
                v_t,
                x_t_after,
            ),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> Pi0TactileFastVit:
        return self._model

    @property
    def inference_mode(self) -> str:
        return self._inference_mode

    def __call__(
        self,
        rng,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise=None,
        prev_chunk_left_over=None,
        inference_delay: int = 0,
    ) -> ActionTrace:
        """Trace one sampling rollout with a FIXED noise (caller-provided).

        ``noise`` must be a (b, ah, ad) array. When None, it is drawn from
        ``rng`` (only for equivalence/dry-run tests, never for the paired
        counterfactual conditions — those always pass the fixed paired noise).
        """
        # Production does the same preprocessing at the top of the sampler.
        observation = self._model._preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self._model.action_horizon, self._model.action_dim))
        noise = jnp.asarray(noise)
        if noise.shape != (batch_size, self._model.action_horizon, self._model.action_dim):
            raise ValueError(
                f"noise shape {noise.shape} does not match expected "
                f"{(batch_size, self._model.action_horizon, self._model.action_dim)}"
            )

        # RTC prefix handling, mirroring production training_time_rtc_sample_actions.
        if self._inference_mode == "rtc":
            if prev_chunk_left_over is None:
                prev_chunk_left_over = jnp.zeros((batch_size, 20, self._model.action_dim))
                inference_delay = 0
            inference_delay = jnp.atleast_1d(jnp.asarray(inference_delay))
            if inference_delay.shape[0] == 1 and batch_size > 1:
                inference_delay = jnp.broadcast_to(inference_delay, (batch_size,))
            b, remaining_len, ad = prev_chunk_left_over.shape
            if remaining_len < self._model.action_horizon:
                padding = jnp.zeros((b, self._model.action_horizon - remaining_len, ad))
                action_prefix = jnp.concatenate([prev_chunk_left_over, padding], axis=1)
            else:
                action_prefix = prev_chunk_left_over[:, : self._model.action_horizon, :]
            action_prefix_mask = jnp.arange(self._model.action_horizon)[None, :] < inference_delay[:, None]
            action_prefix = jnp.asarray(action_prefix)
            action_prefix_mask = jnp.asarray(action_prefix_mask)
        else:
            action_prefix = jnp.zeros((batch_size, self._model.action_horizon, self._model.action_dim))
            action_prefix_mask = jnp.zeros((batch_size, self._model.action_horizon), dtype=jnp.bool_)

        # FastViT branch (manual reproduction for saving; sanity-checked against
        # the actual embed_suffix tokens from step 0).
        fastvit_features, tactile_tokens, tactile_mask = self._tactile_fn(self._state, observation)
        tactile_mask = jnp.asarray(tactile_mask, dtype=jnp.bool_)

        # Prefix forward pass (fills the KV cache), same as production.
        kv_cache, prefix_mask = self._prefix_fn(self._state, observation)

        x_t = noise
        time = jnp.asarray(1.0)
        steps: list[StepTrace] = []
        for step in range(num_steps):
            x_t_new, time_new, vals = self._step_fn(
                self._state,
                kv_cache,
                observation,
                prefix_mask,
                x_t,
                time,
                action_prefix,
                action_prefix_mask,
                dt,
            )
            (
                x_t_before,
                time_masked,
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
                action_hidden,
                v_t,
                x_t_after,
            ) = vals
            steps.append(
                StepTrace(
                    step=step,
                    timestep=time,
                    x_t_before=x_t_before,
                    time_masked=time_masked,
                    suffix_tokens=suffix_tokens,
                    suffix_input_mask=suffix_mask,
                    suffix_ar_mask=suffix_ar_mask,
                    adarms_cond=adarms_cond,
                    action_hidden=action_hidden,
                    v_t=v_t,
                    x_t_after=x_t_after,
                )
            )
            x_t, time = x_t_new, time_new

        return ActionTrace(
            final_action=x_t,
            fastvit_features=fastvit_features,
            tactile_tokens=tactile_tokens,
            tactile_tokens_from_suffix=steps[0].suffix_tokens[:, : self._num_tactile, :],
            tactile_mask=tactile_mask,
            steps=steps,
            inference_mode=self._inference_mode,
            num_steps=num_steps,
        )


def verify_trace_equivalence(
    model: Pi0TactileFastVit,
    observation: _model.Observation,
    *,
    num_steps: int,
    noise,
    inference_mode: str,
    rtol: float = EQ_RTOL,
    atol: float = EQ_ATOL,
    sampler: TraceSampler | None = None,
) -> dict[str, float]:
    """Run the production sampler and the trace sampler with the same noise and
    assert the final actions are allclose.

    Returns a dict with the achieved max/mean absolute difference for logging.
    Raises AssertionError if they diverge beyond the tolerance — the caller
    must fail the whole experiment in that case (never run 2000 inferences
    against an unverified trace implementation).
    """
    if sampler is None:
        sampler = TraceSampler(model, model._tactile_keys, inference_mode=inference_mode)
    traced_action = sampler(jax.random.key(0), observation, num_steps=num_steps, noise=noise).final_action

    if inference_mode == "rtc":
        production_action = model.training_time_rtc_sample_actions(
            jax.random.key(0), observation, num_steps=num_steps, noise=noise
        )
    else:
        production_action = model.sample_actions(jax.random.key(0), observation, num_steps=num_steps, noise=noise)

    traced_np = np.asarray(traced_action)
    production_np = np.asarray(production_action)
    diff = np.abs(traced_np - production_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    logger.info(
        "trace vs production equivalence (%s, %d steps): max abs diff=%.3e mean=%.3e (rtol=%g atol=%g)",
        inference_mode,
        num_steps,
        max_diff,
        mean_diff,
        rtol,
        atol,
    )
    np.testing.assert_allclose(traced_np, production_np, rtol=rtol, atol=atol)
    return {"max_abs_diff": max_diff, "mean_abs_diff": mean_diff, "rtol": rtol, "atol": atol}
