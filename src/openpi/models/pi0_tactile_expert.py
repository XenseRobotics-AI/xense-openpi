"""Synchronous cascaded flow: tactile-blind action expert, then tactile expert.

Both experts predict epsilon - actions. The second stage reads a frozen
prefix + action KV snapshot refreshed at the split time; it predicts the
remaining velocity, not a residual action correction.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models.tactile_encoders import build_tactile_encoder


class Pi0TactileExpert(pi0.Pi0):
    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self._init_tactile_expert(rngs)
        self.cascade_total_steps = config.cascade_total_steps
        self.cascade_split_step = config.cascade_split_step
        self.tactile_loss_weight = config.tactile_loss_weight
        self.tactile_dropout = config.tactile_dropout
        self._tactile_keys = tuple(config.tactile_image_keys)
        width = gemma.get_config(config.tactile_expert_variant).width
        self.tactile_encoder = build_tactile_encoder(
            config.tactile_encoder_name,
            rngs=rngs,
            pretrained_path=config.tactile_pretrained_path,
            compute_dtype=jnp.dtype(config.tactile_compute_dtype),
        )
        linear = {"rngs": rngs, "kernel_init": nnx.initializers.xavier_uniform()}
        self.tactile_proj = nnx.Linear(self.tactile_encoder.feature_dim, width, **linear)
        self.action_in_proj_tac = nnx.Linear(config.action_dim, width, **linear)
        self.action_out_proj_tac = nnx.Linear(width, config.action_dim, **linear)
        self.time_mlp_in_tac = nnx.Linear(width, width, **linear)
        self.time_mlp_out_tac = nnx.Linear(width, width, **linear)

    def _init_tactile_expert(self, rngs):
        """Xavier-init expert 2 only, respecting scanned layers and head axes.

        Include adaRMS kernels: Gemma's default zero residual gates otherwise
        block all tactile-encoder gradients at initialization. Biases stay zero.
        The pretrained experts and their parameter names are untouched.
        """
        state = nnx.state(self.PaliGemma.llm, nnx.Param)
        for path, variable in state.flat_state().items():
            if not any(str(part).endswith("_2") for part in path):
                continue
            if path[-1] not in ("w", "kernel", "gating_einsum", "linear"):
                continue
            shape = variable.value.shape
            in_axis = (-3, -2) if "attn_vec_einsum_2" in path else (-2,)
            input_axes = tuple(axis % len(shape) for axis in in_axis)
            output_axis = len(shape) - 1
            batch_axes = tuple(i for i in range(len(shape)) if i not in (*input_axes, output_axis))
            init = nnx.initializers.xavier_uniform(in_axis=input_axes, out_axis=output_axis, batch_axis=batch_axes)
            variable.value = init(rngs.params(), shape, variable.value.dtype)
        nnx.update(self.PaliGemma.llm, state)

    def _preprocess_observation(self, rng, observation, *, train):
        return _model.preprocess_observation_tactile(
            rng, observation, train=train, image_keys=(*_model.IMAGE_KEYS, *self._tactile_keys)
        )

    def embed_prefix(self, obs):
        return super().embed_prefix(
            dataclasses.replace(
                obs,
                images={k: v for k, v in obs.images.items() if k not in self._tactile_keys},
                image_masks={k: v for k, v in obs.image_masks.items() if k not in self._tactile_keys},
            )
        )

    def embed_tactile(self, obs, *, rng=None, train=False):
        images = jnp.stack([obs.images[k] for k in self._tactile_keys], axis=1)
        batch, views, height, width, channels = images.shape
        features = self.tactile_encoder(images.reshape(batch * views, height, width, channels))
        tokens = self.tactile_proj(features).reshape(batch, views, -1)
        if train and self.tactile_dropout:
            keep = jax.random.bernoulli(rng, 1 - self.tactile_dropout, (batch, 1, 1))
            tokens = jnp.where(keep, tokens, 0)
        mask = jnp.stack([obs.image_masks[k] for k in self._tactile_keys], axis=1)
        return tokens, mask

    def _cached_forward(self, tokens, mask, ar, cond, context_mask, cache, expert):
        """Append one expert stream to an immutable cache; return its new KV."""
        to_context = jnp.broadcast_to(context_mask[:, None, :], (*mask.shape, context_mask.shape[1]))
        attention = jnp.concatenate([to_context, pi0.make_attn_mask(mask, ar)], axis=-1)
        positions = context_mask.sum(axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        streams, conditions = [None, None, None], [None, None, None]
        streams[expert], conditions[expert] = tokens, cond
        outputs, cache = self.PaliGemma.llm(
            streams, mask=attention, positions=positions, kv_cache=cache, adarms_cond=conditions
        )
        return outputs[expert][:, -self.action_horizon :], cache

    def _action_forward(self, obs, x, time, prefix_mask, cache):
        tokens, mask, ar, cond = self.embed_suffix(obs, x, jnp.broadcast_to(time, (x.shape[0],)))
        output, cache = self._cached_forward(tokens, mask, ar, cond, prefix_mask, cache, 1)
        return self.action_out_proj(output), cache

    def _tactile_velocity(self, x, time, tactile, context_mask, cache):
        tactile_tokens, tactile_mask = tactile
        actions = self.action_in_proj_tac(x)
        time = jnp.broadcast_to(time, (x.shape[0],))
        cond = pi0.posemb_sincos(time, actions.shape[-1], min_period=4e-3, max_period=4.0)
        cond = nnx.swish(self.time_mlp_out_tac(nnx.swish(self.time_mlp_in_tac(cond))))
        tokens = jnp.concatenate([tactile_tokens, actions], axis=1)
        mask = jnp.concatenate([tactile_mask, jnp.ones(x.shape[:2], dtype=jnp.bool_)], axis=1)
        ar = jnp.asarray(
            [True] + [False] * (tactile_tokens.shape[1] - 1) + [True] + [False] * (self.action_horizon - 1)
        )
        output, _ = self._cached_forward(tokens, mask, ar, cond, context_mask, cache, 2)
        return self.action_out_proj_tac(output)

    def _rollout_action(self, obs, noise, prefix_mask, cache, num_steps):
        dt = -1.0 / num_steps

        def step(_, carry):
            x, time = carry
            velocity, _ = self._action_forward(obs, x, time, prefix_mask, cache)
            return x + dt * velocity, time + dt

        return jax.lax.fori_loop(0, self.cascade_split_step, step, (noise, jnp.asarray(1.0)))

    def _split_context(self, obs, x_split, time, prefix_mask, cache):
        _, cache = self._action_forward(obs, x_split, time, prefix_mask, cache)
        mask = jnp.concatenate([prefix_mask, jnp.ones(x_split.shape[:2], dtype=jnp.bool_)], axis=1)
        return mask, cache

    def sample_actions(self, rng, observation, *, num_steps=None, noise=None, **kwargs):
        if kwargs:
            raise ValueError("The tactile cascade supports standard denoising only")
        if num_steps is None:
            num_steps = self.cascade_total_steps
        if not isinstance(num_steps, jax.core.Tracer) and (
            int(num_steps) <= 0 or int(num_steps) < self.cascade_split_step
        ):
            raise ValueError("num_steps must be positive and >= cascade_split_step")
        observation = self._preprocess_observation(None, observation, train=False)
        if noise is None:
            noise = jax.random.normal(rng, (observation.state.shape[0], self.action_horizon, self.action_dim))
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix, None, None],
            mask=pi0.make_attn_mask(prefix_mask, prefix_ar),
            positions=jnp.cumsum(prefix_mask, axis=1) - 1,
        )
        x, time = self._rollout_action(observation, noise, prefix_mask, cache, num_steps)
        if not isinstance(num_steps, jax.core.Tracer) and self.cascade_split_step == int(num_steps):
            return x
        context_mask, cache = self._split_context(observation, x, time, prefix_mask, cache)
        tactile = self.embed_tactile(observation)
        dt = -1.0 / num_steps

        def step(_, carry):
            x, time = carry
            return x + dt * self._tactile_velocity(x, time, tactile, context_mask, cache), time + dt

        return jax.lax.fori_loop(self.cascade_split_step, num_steps, step, (x, time))[0]

    def compute_loss(self, rng, observation, actions, *, train=False):
        return self.compute_loss_with_aux(rng, observation, actions, train=train)[0]

    def compute_loss_with_aux(self, rng, observation, actions, *, train=False):
        # Preserve the base pi05 action-loss random stream for direct comparisons.
        preprocess_rng, loss_rng = jax.random.split(rng)
        noise_rng, time_rng = jax.random.split(loss_rng)
        # Use a separate root: fold_in(rng, 1) can equal loss_rng with JAX's
        # partitionable Threefry split, accidentally reusing the action noise.
        kv_rng, tactile_noise_rng, tactile_time_rng, dropout_rng = jax.random.split(jax.random.fold_in(rng, 2), 4)
        observation = self._preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, x, time)
        mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        outputs, cache = self.PaliGemma.llm(
            [prefix, suffix, None],
            mask=pi0.make_attn_mask(mask, jnp.concatenate([prefix_ar, suffix_ar])),
            positions=jnp.cumsum(mask, axis=1) - 1,
            adarms_cond=[None, cond, None],
        )
        loss_a = jnp.mean(
            (self.action_out_proj(outputs[1][:, -self.action_horizon :]) - (noise - actions)) ** 2, axis=-1
        )
        if self.cascade_split_step == self.cascade_total_steps:
            return loss_a, {"loss_action": loss_a.mean(), "loss_tactile": jnp.asarray(0.0)}
        # Cache axes are [layer, batch, sequence, kv_head, head_dim]. Prefix
        # Valid prefix queries never attend to suffix, so their KV equals a
        # separate prefix forward. Padding KV stays masked in every stage.
        cache = jax.tree.map(lambda v: jax.lax.stop_gradient(v[:, :, : prefix.shape[1]]), cache)
        x_split, split_time = self._rollout_action(
            observation, jax.random.normal(kv_rng, actions.shape), prefix_mask, cache, self.cascade_total_steps
        )
        context_mask, cache = self._split_context(
            observation, jax.lax.stop_gradient(x_split), split_time, prefix_mask, cache
        )
        cache = jax.tree.map(jax.lax.stop_gradient, cache)
        noise_t = jax.random.normal(tactile_noise_rng, actions.shape)
        tau_split = 1 - self.cascade_split_step / self.cascade_total_steps
        time_t = (jax.random.beta(tactile_time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001) * tau_split
        x_t = time_t[:, None, None] * noise_t + (1 - time_t[:, None, None]) * actions
        tactile = self.embed_tactile(observation, rng=dropout_rng, train=train)
        velocity = self._tactile_velocity(x_t, time_t, tactile, context_mask, cache)
        loss_t = jnp.mean((velocity - (noise_t - actions)) ** 2, axis=-1)
        return loss_a + self.tactile_loss_weight * loss_t, {
            "loss_action": loss_a.mean(),
            "loss_tactile": loss_t.mean(),
        }
