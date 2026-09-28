import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.rlt import action_space_test
from openpi.rlt import config as _config
from openpi.rlt import mlp_policy
from openpi.rlt import tacxense_reference
from openpi.rlt import td

Z, S, A, C, R, B = 16, 20, 20, 4, 6, 8
# A negligible std makes the Gaussian actor deterministic, so both implementations sample the same action.
_CONFIG = _config.RLConfig(
    num_action_chunks=C,
    ref_num_action_chunks=R,
    actor_hidden_dims=(32, 32),
    critic_hidden_dims=(32, 32),
    fixed_std=1e-9,
)


def _heads(config=_CONFIG, seed=0):
    rngs = nnx.Rngs(seed)
    dims = {"z_dim": Z, "state_dim": S, "action_dim": A}
    return mlp_policy.Actor(config, **dims, rngs=rngs), mlp_policy.Critic(config, **dims, rngs=rngs)


def _batch(space, seed=0):
    rng = np.random.default_rng(seed)

    def obs():
        state = rng.normal(size=(B, S)).astype(np.float32)
        return {
            "z_rl": rng.normal(size=(B, Z)).astype(np.float32),
            "state": state,
            "proprio": np.asarray(space.normalize_state(state)),
            "ref_chunk": rng.uniform(-1, 1, (B, R, A)).astype(np.float32),
        }

    human = rng.uniform(size=(B, C)) < 0.3
    return {
        "curr_obs": obs(),
        "next_obs": obs(),
        "actions": rng.uniform(-1, 1, (B, C, A)).astype(np.float32),
        "chunk_rewards": (rng.uniform(size=(B, C)) < 0.2).astype(np.float32),
        "terminated": rng.uniform(size=B) < 0.5,
        "intervention_mask": human,
    }


def test_training_reference_substitutes_human_steps():
    batch = _batch(action_space_test.bi_flexiv_space())
    ref = np.asarray(td.training_reference(batch))
    human = batch["intervention_mask"]
    np.testing.assert_array_equal(ref[:, :C][human], batch["actions"][human])
    np.testing.assert_array_equal(ref[:, :C][~human], batch["curr_obs"]["ref_chunk"][:, :C][~human])
    np.testing.assert_array_equal(ref[:, C:], batch["curr_obs"]["ref_chunk"][:, C:])


def test_reference_dropout_only_zeroes_the_reference():
    actor, _ = _heads()
    obs = _batch(action_space_test.bi_flexiv_space())["curr_obs"]
    dropped = actor(obs, dropout_rng=jax.random.key(0), reference_dropout_prob=1.0)
    np.testing.assert_allclose(dropped, actor({**obs, "ref_chunk": jnp.zeros_like(obs["ref_chunk"])}), atol=1e-6)


def test_losses_are_differentiable():
    space = action_space_test.bi_flexiv_space()
    actor, critic = _heads(dataclasses.replace(_CONFIG, fixed_std=0.002))
    batch = _batch(space)
    grads = nnx.grad(lambda c: td.critic_loss(c, c, actor, space, batch, jax.random.key(0), gamma=0.99)[0])(critic)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    grads = nnx.grad(
        lambda a: td.actor_loss(
            a, critic, space, batch, jax.random.key(1), q_weight=0.1, bc_weight=5.0, reference_dropout_prob=0.5
        )[0]
    )(actor)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))


def _copy_mlp(ours: mlp_policy._MLP, theirs) -> None:
    import torch

    linears = [m for m in theirs if isinstance(m, torch.nn.Linear)]
    norms = [m for m in theirs if isinstance(m, torch.nn.LayerNorm)]
    for layer, ref in zip(ours.linears, linears, strict=True):
        layer.kernel.value = jnp.asarray(ref.weight.detach().numpy().T)
        layer.bias.value = jnp.asarray(ref.bias.detach().numpy())
    for norm, ref in zip(ours.norms or [], norms, strict=True):
        norm.scale.value = jnp.asarray(ref.weight.detach().numpy())
        norm.bias.value = jnp.asarray(ref.bias.detach().numpy())


def test_matches_tacxense():
    torch = pytest.importorskip("torch")
    rlt = tacxense_reference.import_tacxense()
    space = action_space_test.bi_flexiv_space()
    torch.manual_seed(0)
    reference = rlt.RLTActorCritic(
        rlt.RLTActor(
            z_dim=Z,
            state_dim=S,
            action_dim=A,
            num_action_chunks=C,
            ref_num_action_chunks=R,
            hidden_dims=(32, 32),
            fixed_std=1e-9,
        ),
        rlt.TwinQ(z_dim=Z, state_dim=S, action_chunk_dim=C * A, hidden_dims=(32, 32)),
    )
    with torch.no_grad():  # a distinct target critic, so the bootstrap path is really exercised
        for p in reference.target_critic.parameters():
            p.add_(0.05 * torch.randn_like(p))
    codec = rlt.ActionCodec(
        rlt.ActionLayout.bi_flexiv(tuple(bool(x) for x in space.delta_mask)),
        state_q01=space.state_q01,
        state_q99=space.state_q99,
        action_q01=space.action_q01,
        action_q99=space.action_q99,
    )

    actor, critic = _heads()
    _, target = _heads(seed=1)
    _copy_mlp(actor.net, reference.actor.net)
    for ours, theirs in ((critic, reference.critic), (target, reference.target_critic)):
        _copy_mlp(ours.heads[0], theirs.q1)
        _copy_mlp(ours.heads[1], theirs.q2)

    batch = _batch(space)
    t_batch = jax.tree.map(torch.from_numpy, batch)
    t_batch.update(valid_action_mask=torch.ones(B, C, dtype=torch.bool), executed_length=torch.full((B,), C))

    loss, info = td.critic_loss(critic, target, actor, space, batch, jax.random.key(0), gamma=0.9)
    ref_loss, ref_info = rlt.critic_loss(t_batch, reference, 0.9, codec=codec)
    np.testing.assert_allclose(loss, ref_loss.item(), rtol=1e-4)
    np.testing.assert_allclose(info["q_target"], ref_info["q_target"], rtol=1e-4, atol=1e-6)

    loss, info = td.actor_loss(
        actor, critic, space, batch, jax.random.key(0), q_weight=0.1, bc_weight=5.0, reference_dropout_prob=0.0
    )
    ref_loss, ref_info = rlt.actor_loss(t_batch, reference, q_weight=0.1, bc_weight=5.0, codec=codec)
    np.testing.assert_allclose(loss, ref_loss.item(), rtol=1e-4)
    np.testing.assert_allclose(info["bc_loss"], ref_info["bc_loss"], rtol=1e-4)


def test_smoothness_and_weight_schedule_match_tacxense():
    torch = pytest.importorskip("torch")
    rlt = tacxense_reference.import_tacxense()
    rng = np.random.default_rng(3)
    chunk = rng.normal(size=(5, C + 3, A)).astype(np.float32)
    orders = (1.0, 0.5, 0.25)
    ours = td.smoothness(jnp.asarray(chunk), orders)
    # TacXense's actor_loss term with every step valid (no RTC): sum over masked diffs / (count * A).
    expected = 0.0
    t = torch.from_numpy(chunk)
    for order, weight in enumerate(orders, start=1):
        diffs = t.diff(n=order, dim=1)
        expected += weight * diffs.square().sum().item() / (diffs.shape[0] * diffs.shape[1] * A)
    np.testing.assert_allclose(ours, expected, rtol=1e-5)

    schedule = _config.ActorWeightSchedule(
        enable=True, warmup_updates=3, ramp_updates=4, warmup_bc_weight=2.0, warmup_q_weight=0.0
    )
    reference = rlt.ActorWeightSchedule(
        rlt.ActorWeightScheduleConfig(**dataclasses.asdict(schedule)), bc_weight=5.0, q_weight=0.1
    )
    for step in range(10):
        np.testing.assert_allclose(schedule.weights(step, bc_weight=5.0, q_weight=0.1), reference.weights(step))
    assert _config.ActorWeightSchedule().weights(7, bc_weight=5.0, q_weight=0.1) == (5.0, 0.1)


def test_smooth_weight_adds_to_the_actor_loss():
    space = action_space_test.bi_flexiv_space()
    actor, critic = _heads()
    batch = _batch(space)
    kwargs = {"q_weight": 0.1, "bc_weight": 5.0, "reference_dropout_prob": 0.0}
    base, _ = td.actor_loss(actor, critic, space, batch, jax.random.key(0), **kwargs)
    smoothed, info = td.actor_loss(actor, critic, space, batch, jax.random.key(0), smooth_weight=0.5, **kwargs)
    np.testing.assert_allclose(smoothed - base, 0.5 * info["smooth_loss"], atol=1e-6)
    assert info["smooth_loss"] > 0
