"""Trace sampler tests: production equivalence, shapes, trajectory math.

Uses a small dummy-variant Pi0TactileFastVit model (gemma dummy width=64,
random-init FastViT) so the tests run on a laptop GPU/CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
from test.tactile_counterfactual.trace_sampler import TraceSampler
from test.tactile_counterfactual.trace_sampler import verify_trace_equivalence

NUM_STEPS = 3


@pytest.fixture(scope="module")
def model_cfg():
    return Pi0TactileFastVitConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        enable_training_time_rtc=True,
        max_delay=10,
        tactile_pretrained_path=None,
    )


@pytest.fixture(scope="module")
def model(model_cfg):
    return model_cfg.create(jax.random.key(0))


@pytest.fixture(scope="module")
def obs(model_cfg):
    observation, _ = model_cfg.inputs_spec(batch_size=1)
    return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), observation)


@pytest.fixture(scope="module")
def noise(model):
    return jnp.asarray(
        jax.random.normal(jax.random.key(7), (1, model.action_horizon, model.action_dim)),
        dtype=jnp.float32,
    )


def test_trace_equivalent_to_production_standard(model, obs, noise):
    result = verify_trace_equivalence(model, obs, num_steps=NUM_STEPS, noise=noise, inference_mode="standard")
    assert result["max_abs_diff"] < 1e-4


def test_trace_equivalent_to_production_rtc(model, obs, noise):
    # With prev_chunk_left_over=None the production RTC sampler uses the dummy
    # prefix + inference_delay=0, which is the first-inference deployment path.
    result = verify_trace_equivalence(model, obs, num_steps=NUM_STEPS, noise=noise, inference_mode="rtc")
    assert result["max_abs_diff"] < 1e-4


def test_shape_validation(model, obs, noise):
    sampler = TraceSampler(model, model._tactile_keys, inference_mode="rtc")
    trace = sampler(jax.random.key(0), obs, num_steps=NUM_STEPS, noise=noise)

    fastvit_dim = model.tactile_encoder.feature_dim
    expert_width = model.action_in_proj.out_features
    ah, ad = model.action_horizon, model.action_dim
    n_tactile = len(model._tactile_keys)

    assert trace.fastvit_features.shape == (1, n_tactile, fastvit_dim)
    assert trace.tactile_tokens.shape == (1, n_tactile, expert_width)
    assert trace.tactile_tokens_from_suffix.shape == (1, n_tactile, expert_width)

    assert len(trace.steps) == NUM_STEPS
    suffix_len = n_tactile + ah  # pi05: no state token in the suffix
    for s in trace.steps:
        assert s.suffix_tokens.shape == (1, suffix_len, expert_width)
        assert s.suffix_input_mask.shape == (1, suffix_len)
        assert s.suffix_ar_mask.shape == (suffix_len,)
        assert s.adarms_cond is not None
        assert s.adarms_cond.shape == (1, suffix_len, expert_width)
        assert s.action_hidden.shape == (1, ah, expert_width)
        assert s.v_t.shape == (1, ah, ad)
        assert s.x_t_before.shape == (1, ah, ad)
        assert s.x_t_after.shape == (1, ah, ad)
    assert trace.final_action.shape == (1, ah, ad)


def test_euler_update_math(model, obs, noise):
    """x_t_after == x_t_before + dt * v_t (standard mode)."""
    sampler = TraceSampler(model, model._tactile_keys, inference_mode="standard")
    trace = sampler(jax.random.key(0), obs, num_steps=NUM_STEPS, noise=noise)
    dt = -1.0 / NUM_STEPS
    for s in trace.steps:
        expected = np.asarray(s.x_t_before) + dt * np.asarray(s.v_t)
        np.testing.assert_allclose(np.asarray(s.x_t_after), expected, rtol=1e-5, atol=1e-6)


def test_tactile_tokens_match_suffix(model, obs, noise):
    sampler = TraceSampler(model, model._tactile_keys, inference_mode="standard")
    trace = sampler(jax.random.key(0), obs, num_steps=NUM_STEPS, noise=noise)
    np.testing.assert_allclose(
        np.asarray(trace.tactile_tokens), np.asarray(trace.tactile_tokens_from_suffix), rtol=1e-5, atol=1e-6
    )


def test_fixed_noise_reproducible(model, obs, noise):
    sampler = TraceSampler(model, model._tactile_keys, inference_mode="standard")
    a = sampler(jax.random.key(0), obs, num_steps=NUM_STEPS, noise=noise)
    b = sampler(jax.random.key(1), obs, num_steps=NUM_STEPS, noise=noise)
    np.testing.assert_array_equal(np.asarray(a.final_action), np.asarray(b.final_action))


def test_noise_shape_mismatch_rejected(model, obs):
    sampler = TraceSampler(model, model._tactile_keys, inference_mode="standard")
    with pytest.raises(ValueError, match="noise shape"):
        sampler(jax.random.key(0), obs, num_steps=NUM_STEPS, noise=jnp.zeros((1, 3, 3)))
