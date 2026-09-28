"""Chunked TD critic loss and BC-regularized actor loss (RLinf's RLTAC, as tuned in TacXense).

A transition is one C-step window: ``curr_obs`` at its first step, the C
executed actions (normalized), per-step rewards, ``next_obs`` C steps later and
``terminated`` for a labeled success/failure end. The critic target is

    sum_l gamma**l r_l + gamma**C * (1 - terminated) * min_i Q'_i(s', canon(pi(s')))

and the actor minimizes ``-q_weight * Q1(s, canon(pi(s))) + bc_weight * MSE`` (plus an
optional smoothness penalty on its chunk) against the training reference: the VLA reference, with human actions at
intervened steps. Replay keeps the raw VLA reference; the substitution happens
here, and the same tensor is both the actor's input and its BC target.
"""

import jax
import jax.numpy as jnp

from openpi.rlt import action_space as _action_space
from openpi.rlt import mlp_policy


def discounted_chunk_rewards(rewards: jax.Array, gamma: float) -> jax.Array:
    """``sum_l gamma**l * r_l`` over the chunk axis, ``(B, C) -> (B,)``."""
    return jnp.sum(rewards * gamma ** jnp.arange(rewards.shape[-1]), axis=-1)


def training_reference(batch: dict) -> jax.Array:
    """Reference chunk with the first C steps replaced by the human action wherever intervened."""
    ref = batch["curr_obs"]["ref_chunk"]
    num_chunks = batch["actions"].shape[1]
    head = jnp.where(batch["intervention_mask"][..., None], batch["actions"], ref[:, :num_chunks])
    return jnp.concatenate([head, ref[:, num_chunks:]], axis=1)


def smoothness(chunk: jax.Array, order_weights: tuple[float, float, float]) -> jax.Array:
    """``sum_k w_k * mean(diff^k(chunk)**2)`` over the chunk axis: velocity, acceleration, jerk.

    Measured in the normalized action space the actor emits, over the whole chunk.
    """
    total = jnp.zeros((), chunk.dtype)
    diffs = chunk
    for weight in order_weights:
        diffs = jnp.diff(diffs, axis=1)
        if weight > 0:
            total = total + weight * jnp.mean(jnp.square(diffs))
    return total


def critic_loss(
    critic: mlp_policy.Critic,
    target_critic: mlp_policy.Critic,
    actor: mlp_policy.Actor,
    space: _action_space.ActionSpace,
    batch: dict,
    rng: jax.Array,
    *,
    gamma: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    next_obs = batch["next_obs"]
    next_actions = space.canonicalize(actor(next_obs, noise_rng=rng), next_obs["state"])
    q_next = jnp.min(target_critic(next_obs, next_actions), axis=-1)
    bootstrap = gamma ** batch["actions"].shape[1] * jnp.where(batch["terminated"], 0.0, q_next)
    target = jax.lax.stop_gradient(discounted_chunk_rewards(batch["chunk_rewards"], gamma) + bootstrap)

    q = critic(batch["curr_obs"], batch["actions"])
    loss = jnp.mean(jnp.square(q - target[:, None]))
    return loss, {
        "critic_loss": loss,
        "q_data": q.mean(),
        "q_data_min": q.min(),
        "q_data_max": q.max(),
        "q_disagreement": jnp.abs(q[:, 0] - q[:, 1]).mean(),
        "q_target": target.mean(),
        "q_target_std": target.std(),
        "terminal_ratio": batch["terminated"].mean(),
    }


def actor_loss(
    actor: mlp_policy.Actor,
    critic: mlp_policy.Critic,
    space: _action_space.ActionSpace,
    batch: dict,
    rng: jax.Array,
    *,
    q_weight: float | jax.Array,
    bc_weight: float | jax.Array,
    reference_dropout_prob: float,
    smooth_weight: float = 0.0,
    smooth_order_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[jax.Array, dict[str, jax.Array]]:
    curr_obs = batch["curr_obs"]
    reference = training_reference(batch)
    noise_rng, dropout_rng = jax.random.split(rng)
    raw = actor(
        {**curr_obs, "ref_chunk": reference},
        noise_rng=noise_rng,
        dropout_rng=dropout_rng,
        reference_dropout_prob=reference_dropout_prob,
    )
    pi = space.canonicalize(raw, curr_obs["state"])
    # Only Q1 drives the actor, as in RLinf.
    q_pi = critic(curr_obs, pi)[:, 0].mean()
    bc = jnp.mean(jnp.square(pi - reference[:, : pi.shape[1]]))
    smooth = smoothness(pi, smooth_order_weights)
    loss = -q_weight * q_pi + bc_weight * bc + smooth_weight * smooth
    return loss, {
        "actor_loss": loss,
        "q_pi": q_pi,
        "bc_loss": bc,
        "weighted_q": q_weight * q_pi,
        "weighted_bc": bc_weight * bc,
        "smooth_loss": smooth,
        "weighted_smooth": smooth_weight * smooth,
        "action_projection_abs_mean": jnp.abs(pi - raw).mean(),
        "human_mask_ratio": batch["intervention_mask"].mean(),
    }
