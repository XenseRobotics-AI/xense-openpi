"""Chunked TD critic loss and BC-regularized actor loss (RLinf's RLTAC, as tuned in TacXense).

A transition is one C-step window: ``curr_obs`` at its first step, the C
executed actions (normalized), per-step rewards, ``next_obs`` C steps later and
``terminated`` for a labeled success/failure end. The critic target is

    mc:        the row's true discounted return to the labeled phase end (no bootstrap)
    td:        sum_l gamma**l r_l + gamma**C * (1 - terminated) * min_i Q'_i(s', canon(pi(s')))
    mix:       mc_lambda * mc + (1 - mc_lambda) * td

and the actor minimizes ``-q_weight * Q1(s, canon(pi(s))) + bc_weight * MSE``
against the BC target: the VLA reference, with human actions at intervened
steps. The actor's reference input is always the raw VLA reference - the same
input it sees at inference - so intervened steps teach it to turn the VLA
reference into the human correction. Replay keeps the raw VLA reference; the
substitution happens here, at loss time, and only builds the BC target.
"""

import jax
import jax.numpy as jnp

from openpi.rlt import action_space as _action_space
from openpi.rlt import mlp_policy


def discounted_chunk_rewards(rewards: jax.Array, gamma: float) -> jax.Array:
    """``sum_l gamma**l * r_l`` over the chunk axis, ``(B, C) -> (B,)``."""
    return jnp.sum(rewards * gamma ** jnp.arange(rewards.shape[-1]), axis=-1)


def training_reference(batch: dict) -> jax.Array:
    """BC target: the VLA reference, first C steps replaced by the human action wherever intervened.

    Only the BC target is substituted; the actor's reference input stays the raw
    VLA reference (``curr_obs["ref_chunk"]``), matching what it sees at inference.
    """
    ref = batch["curr_obs"]["ref_chunk"]
    num_chunks = batch["actions"].shape[1]
    head = jnp.where(batch["intervention_mask"][..., None], batch["actions"], ref[:, :num_chunks])
    return jnp.concatenate([head, ref[:, num_chunks:]], axis=1)


def critic_loss(
    critic: mlp_policy.Critic,
    target_critic: mlp_policy.Critic,
    actor: mlp_policy.Actor,
    space: _action_space.ActionSpace,
    batch: dict,
    rng: jax.Array,
    *,
    gamma: float,
    critic_target: str = "td",
    mc_lambda: float = 0.5,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    td_target = None
    if critic_target in ("td", "mix"):
        next_obs = batch["next_obs"]
        next_actions = space.canonicalize(actor(next_obs, noise_rng=rng), next_obs["state"])
        q_next = jnp.min(target_critic(next_obs, next_actions), axis=-1)
        bootstrap = gamma ** batch["actions"].shape[1] * jnp.where(batch["terminated"], 0.0, q_next)
        td_target = discounted_chunk_rewards(batch["chunk_rewards"], gamma) + bootstrap
    if critic_target == "mc":
        target = batch["mc_return"]
    elif critic_target == "td":
        target = td_target
    elif critic_target == "mix":
        target = mc_lambda * batch["mc_return"] + (1.0 - mc_lambda) * td_target
    else:
        raise ValueError(f"critic_target must be mc, td or mix, got {critic_target!r}.")
    target = jax.lax.stop_gradient(target)

    q = critic(batch["curr_obs"], batch["actions"])
    loss = jnp.mean(jnp.square(q - target[:, None]))
    # R^2 of the return regression, averaged over heads: ~0 means the critic collapses to
    # predicting the mean (constant output); ~1 means it tracks the true returns.
    target_var = jnp.maximum(target.var(), 1e-8)
    explained_variance = (1.0 - jnp.square(target[:, None] - q).mean(axis=0) / target_var).mean()
    return loss, {
        "critic_loss": loss,
        "q_data": q.mean(),
        "q_data_std": q.std(),
        "q_data_min": q.min(),
        "q_data_max": q.max(),
        "q_disagreement": jnp.abs(q[:, 0] - q[:, 1]).mean(),
        "q_target": target.mean(),
        "q_target_std": target.std(),
        "explained_variance": explained_variance,
        "mc_return": batch["mc_return"].mean(),
        "terminal_ratio": batch["terminated"].mean(),
    }


def actor_loss(
    actor: mlp_policy.Actor,
    critic: mlp_policy.Critic,
    space: _action_space.ActionSpace,
    batch: dict,
    rng: jax.Array,
    *,
    q_weight: float,
    bc_weight: float,
    reference_dropout_prob: float,
    input_reference: str = "corrected",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    curr_obs = batch["curr_obs"]
    # Human corrections always supervise BC; conditioning is selected independently.
    bc_target = training_reference(batch)
    if input_reference not in ("corrected", "proposal"):
        raise ValueError("input_reference must be corrected or proposal.")
    actor_obs = {**curr_obs, "ref_chunk": bc_target} if input_reference == "corrected" else curr_obs
    noise_rng, dropout_rng = jax.random.split(rng)
    raw = actor(
        actor_obs,
        noise_rng=noise_rng,
        dropout_rng=dropout_rng,
        reference_dropout_prob=reference_dropout_prob,
    )
    pi = space.canonicalize(raw, curr_obs["state"])
    # Only Q1 drives the actor, as in RLinf.
    q_pi = critic(curr_obs, pi)[:, 0].mean()
    bc = jnp.mean(jnp.square(pi - bc_target[:, : pi.shape[1]]))
    loss = -q_weight * q_pi + bc_weight * bc
    return loss, {
        "actor_loss": loss,
        "q_pi": q_pi,
        "bc_loss": bc,
        "weighted_q": q_weight * q_pi,
        "weighted_bc": bc_weight * bc,
        "action_projection_abs_mean": jnp.abs(pi - raw).mean(),
        "human_mask_ratio": batch["intervention_mask"].mean(),
    }
