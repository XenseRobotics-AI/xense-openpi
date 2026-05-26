import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import Unpack, override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    # Cumsum before broadcasting: when mask_ar is 1-D (a compile-time constant),
    # this cuts XLA constant-folding cost from O(B*N^2) to O(N^2).
    if mask_ar.ndim < input_mask.ndim:
        cumsum = jnp.broadcast_to(jnp.cumsum(mask_ar, axis=-1), input_mask.shape)
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, "*shape"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "*shape {embedding_dim}"]:
    """Computes sine-cosine positional embeddings for scalar or token-wise positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    pos = jnp.asarray(pos)
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = pos[..., None] * (1.0 / period * 2 * jnp.pi)
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(
        self,
        config: pi0_config.Pi0Config,
        rngs: nnx.Rngs,
    ):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
        )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        self._enable_training_time_rtc = config.enable_training_time_rtc
        self._max_delay = config.max_delay

    def _preprocess_observation(
        self,
        rng: at.KeyArrayLike | None,
        observation: _model.Observation,
        *,
        train: bool,
    ) -> _model.Observation:
        """Subclass hook for swapping in a different preprocess (e.g. tactile keys)."""
        return _model.preprocess_observation(rng, observation, train=train)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            with jax.named_scope(f"prefix/img/{name}"):
                image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            with jax.named_scope("prefix/prompt_embed"):
                tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

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
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            with jax.named_scope("suffix/state_token"):
                state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        with jax.named_scope("suffix/action_in_proj"):
            action_tokens = self.action_in_proj(noisy_actions)
        if timestep.ndim == 1:
            if timestep.shape[0] != noisy_actions.shape[0]:
                raise ValueError(f"Expected timestep batch dimension {noisy_actions.shape[0]}, got {timestep.shape[0]}")
        elif timestep.ndim == 2:
            expected_shape = noisy_actions.shape[:2]
            if timestep.shape != expected_shape:
                raise ValueError(f"Expected per-action timestep shape {expected_shape}, got {timestep.shape}")
        else:
            raise ValueError(f"Expected timestep ndim 1 or 2, got {timestep.ndim}")

        with jax.named_scope("suffix/time_embed"):
            # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            if self.pi05:
                # time MLP (for adaRMS)
                time_emb = self.time_mlp_in(time_emb)
                time_emb = nnx.swish(time_emb)
                time_emb = self.time_mlp_out(time_emb)
                time_emb = nnx.swish(time_emb)
                action_expert_tokens = action_tokens
                adarms_cond = time_emb
            else:
                # mix timestep + action information using an MLP (no adaRMS)
                if time_emb.ndim == 2:
                    time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
                else:
                    time_tokens = time_emb
                action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
                action_time_tokens = self.action_time_mlp_in(action_time_tokens)
                action_time_tokens = nnx.swish(action_time_tokens)
                action_time_tokens = self.action_time_mlp_out(action_time_tokens)
                action_expert_tokens = action_time_tokens
                adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        # Split RNG and preprocess observation
        preprocess_rng, loss_rng = jax.random.split(rng)
        with jax.named_scope("loss/preprocess"):
            observation = self._preprocess_observation(preprocess_rng, observation, train=train)

        # Delegate to specific implementation
        if not self._enable_training_time_rtc:
            return self._compute_loss_standard(loss_rng, observation, actions)
        return self._compute_loss_training_time_rtc(loss_rng, observation, actions)

    def _compute_loss_standard(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> at.Float[at.Array, "*b ah"]:
        """Standard Pi0 loss computation."""
        with jax.named_scope("loss/rng_noise_time"):
            noise_rng, time_rng = jax.random.split(rng)
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]

            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        with jax.named_scope("loss/embed_prefix"):
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        with jax.named_scope("loss/embed_suffix"):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        with jax.named_scope("loss/attn_mask"):
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
        with jax.named_scope("loss/llm_forward"):
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )
        with jax.named_scope("loss/action_out"):
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _compute_loss_training_time_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> at.Float[at.Array, "*b ah"]:
        """Training-time RTC loss computation."""
        with jax.named_scope("loss/rng_noise_time_delay"):
            noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
            b, ah, ad = actions.shape  # (batch_size, action_horizon, action_dim)
            time = jax.random.uniform(time_rng, (b,))
            noise = jax.random.normal(noise_rng, (b, ah, ad))

            # sample delays from some distribution of choice:
            # here, we use Uniform[0, max_delay), as in our real-world experiments
            delay = jax.random.randint(delay_rng, (b,), 0, self._max_delay)

            # Create action prefix mask (True for prefix actions, False for postfix actions)
            action_prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]

            # Compute x_t: prefix uses clean actions (time=0.0), postfix uses noisy actions
            time_masked = jnp.where(action_prefix_mask, 0.0, time[:, None])
            x_t = time_masked[:, :, None] * noise + (1 - time_masked[:, :, None]) * actions
            u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        with jax.named_scope("loss/embed_prefix"):
            prefix_tokens, prefix_input_mask, prefix_ar_mask = self.embed_prefix(observation)
        with jax.named_scope("loss/embed_suffix"):
            suffix_tokens, suffix_input_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                time_masked,
            )
        with jax.named_scope("loss/attn_mask"):
            input_mask = jnp.concatenate([prefix_input_mask, suffix_input_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
        with jax.named_scope("loss/llm_forward"):
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )
        with jax.named_scope("loss/action_out"):
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            loss = (v_t - u_t) ** 2

            # compute the loss on the postfix only
            # Use action_prefix_mask (not prefix_input_mask which is for transformer)
            action_postfix_mask = jnp.logical_not(action_prefix_mask)[:, :, None]
            loss = jnp.sum(loss * action_postfix_mask, axis=-1) / (jnp.sum(action_postfix_mask, axis=-1) + 1e-8)
            return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        **kwargs: Unpack[_model.ActionSelectKwargs],
    ) -> _model.Actions:
        observation = self._preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def get_v_t(x_t, time, obs):  # equivalent to denoise_step in PyTorch
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                obs, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=pos,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry
            v_t = get_v_t(x_t, time, observation)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _log_rtc_prefix_diagnostics(
        self,
        x_0: jax.Array,
        action_prefix: jax.Array,
        action_prefix_mask: jax.Array,
        inference_delay: jax.Array,
    ) -> None:
        """Host-side print of max/mean |x_0[:delay] - action_prefix[:delay]|.

        If RTC prefix freezing works as intended in `training_time_rtc_sample_actions`,
        this should be exactly 0 in the prefix region. A nonzero value points at
        a sampling bug (prefix drifted during denoising) rather than a
        client-side serialization issue. Printed from inside a jit via
        `jax.debug.print`, which does not break compilation.
        """
        delay_scalar = inference_delay[0]
        # For the host print, reduce over batch/time/dim using the prefix mask,
        # so padding past the delay does not pollute the stats.
        mask3d = action_prefix_mask[:, :, None]
        diff = jnp.abs(x_0 - action_prefix) * mask3d
        denom = jnp.maximum(jnp.sum(mask3d), 1.0)
        max_diff = jnp.max(diff)
        mean_diff = jnp.sum(diff) / denom
        jax.debug.print(
            "RTC sampler prefix diagnostics: delay={d} max|x0 - prefix|={mx:.6f} "
            "mean|x0 - prefix|={me:.6f} (expected ~0 if prefix is frozen)",
            d=delay_scalar,
            mx=max_diff,
            me=mean_diff,
        )

    def training_time_rtc_sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        **kwargs: Unpack[_model.ActionSelectKwargs],
    ) -> _model.Actions:
        observation = self._preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Get action_prefix and inference_delay from kwargs
        prev_chunk_left_over = kwargs.get("prev_chunk_left_over")  # shape: (b, remaining_len, ad)
        inference_delay = kwargs.get("inference_delay")  # shape: (b,) or scalar

        # Handle first inference: if prev_chunk_left_over is None, this is the first inference
        # Create a dummy prev_chunk and set inference_delay to 0 so it won't be used
        if prev_chunk_left_over is None:
            logging.info(
                "RTC: First inference detected (prev_chunk_left_over is None), "
                "using dummy prev_chunk with inference_delay=0"
            )
            prev_chunk_left_over = jnp.zeros(
                (
                    batch_size,
                    20,
                    self.action_dim,
                )  # 20 is the default action_queue_size_to_get_new_actions
            )  # dummy shape
            inference_delay = 0  # No prefix constraint for first inference

        if inference_delay is None:
            logging.warning("RTC: inference_delay is None, defaulting to 0")
            inference_delay = 0

        # Ensure inference_delay has batch dimension (b,)
        # It may come as a scalar from the client
        inference_delay = jnp.atleast_1d(inference_delay)

        # Pad or truncate prev_chunk_left_over to action_horizon
        # action_prefix should be padded to (b, action_horizon, action_dim)
        # Only the first delay actions are valid, controlled by action_prefix_mask
        b, remaining_len, ad = prev_chunk_left_over.shape

        # Broadcast inference_delay to batch size if needed
        if inference_delay.shape[0] == 1 and b > 1:
            inference_delay = jnp.broadcast_to(inference_delay, (b,))

        if remaining_len < self.action_horizon:
            # Pad with zeros
            padding = jnp.zeros((b, self.action_horizon - remaining_len, ad))
            action_prefix = jnp.concatenate([prev_chunk_left_over, padding], axis=1)
        else:
            # Truncate to action_horizon
            action_prefix = prev_chunk_left_over[:, : self.action_horizon, :]

        # Create prefix mask: True where index < delay (these positions use action_prefix)
        action_prefix_mask = jnp.arange(self.action_horizon)[None, :] < inference_delay[:, None]

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry

            # Re-freeze the action prefix before every denoising step so sampling
            # stays aligned with the current RTC training loss semantics: the
            # prefix is always treated as clean conditioning, while only the
            # postfix is denoised.
            x_t = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_t)
            time_masked = jnp.where(action_prefix_mask, 0.0, time)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_masked)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            x_t = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_t + dt * v_t)
            return x_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # Server-side RTC diagnostic: without a post-loop re-clamp of the
        # prefix, x_0[:, :delay] = action_prefix + dt * v_t_last[prefix], which
        # the client observes as a nonzero "frozen prefix" diff at merge time.
        # This host-print confirms whether drift is sampler-side (here) or
        # serialization-side (client).
        self._log_rtc_prefix_diagnostics(x_0, action_prefix, action_prefix_mask, inference_delay)
        return x_0
