"""Tests for Pi0TactileFastVit's Latent Tactile Predictor (LTP) path.

Everything runs on the ``dummy`` Gemma variant (4 blocks, width 64) so the suite is
fast; the FastViT encoder is randomly initialised.
"""

import dataclasses

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models import pi0_tactile_fastvit as _tactile
from openpi.models import pi0_tactile_fastvit_config as _config
from openpi.shared import nnx_utils

_BASE = {
    "paligemma_variant": "dummy",
    "action_expert_variant": "dummy",
    "pi05": True,
    "action_dim": 8,
    "action_horizon": 6,
    "max_token_len": 8,
}
_HORIZONS = (1, 2, 3)
_DIM = 16
_LAYER = 2
_BATCH = 2


def _ltp_config(**overrides) -> _config.Pi0TactileFastVitConfig:
    kwargs = {
        **_BASE,
        "tactile_future_layer": _LAYER,
        "tactile_future_dim": _DIM,
        "tactile_future_horizons": _HORIZONS,
        **overrides,
    }
    return _config.Pi0TactileFastVitConfig(**kwargs)


def _future_mask() -> jnp.ndarray:
    # Sample 0 has two valid horizons, sample 1 one: exercises the per-(k, s) masking.
    return jnp.array([[True, True, False], [True, False, False]])


def _obs_with_aux(config, key):
    obs = config.fake_obs(_BATCH)
    z = jax.random.normal(key, (_BATCH, len(_HORIZONS), len(config.tactile_image_keys), _DIM))
    return dataclasses.replace(
        obs,
        aux_targets={_tactile.FUTURE_TACTILE_Z: z, _tactile.FUTURE_TACTILE_MASK: _future_mask()},
    )


def _grad_path_norms(grads: nnx.State) -> dict[str, np.ndarray]:
    """Per-leaf gradient norms; leaves under the scanned Gemma blocks keep their layer axis."""
    out = {}
    for path, leaf in grads.flat_state().items():
        name = "/".join(str(p) for p in path)
        value = jnp.asarray(leaf.value, dtype=jnp.float32)
        if "/layers/" in name:
            out[name] = np.asarray(jnp.sqrt(jnp.sum(jnp.square(value), axis=tuple(range(1, value.ndim)))))
        else:
            out[name] = np.asarray(jnp.linalg.norm(value))
    return out


def test_config_validates_layer_and_rtc():
    with pytest.raises(ValueError, match="tactile_future_layer"):
        _ltp_config(tactile_future_layer=99)
    with pytest.raises(ValueError, match="RTC"):
        _ltp_config(enable_training_time_rtc=True)
    # Off by default, and then the RTC restriction does not apply.
    _config.Pi0TactileFastVitConfig(**_BASE, enable_training_time_rtc=True)


def test_compute_loss_returns_flow_and_masked_tac():
    key = jax.random.key(0)
    config = _ltp_config()
    model = config.create(key)
    obs = _obs_with_aux(config, jax.random.key(1))
    act = config.fake_act(_BATCH)

    losses = nnx_utils.module_jit(model.compute_loss)(key, obs, act)

    assert set(losses) == {"flow", "tac", "tac_mask", "tac_by_time"}
    assert losses["flow"].shape == (_BATCH, config.action_horizon)
    assert losses["tac"].shape == (_BATCH, len(_HORIZONS), len(config.tactile_image_keys))
    assert losses["tac_mask"].shape == losses["tac"].shape
    expected_mask = np.broadcast_to(np.asarray(_future_mask())[:, :, None], losses["tac"].shape)
    assert np.array_equal(np.asarray(losses["tac_mask"]), expected_mask)
    assert losses["tac_by_time"].shape == (len(_tactile.TAC_TIME_BIN_EDGES) - 1,)
    # Every sample lands in exactly one bin, so the bins' masked means average back to the total.
    valid = np.asarray(losses["tac_mask"])
    total = float(np.sum(np.asarray(losses["tac"]) * valid) / valid.sum())
    bins = np.asarray(losses["tac_by_time"])
    assert np.isfinite(bins).any()
    assert np.nanmin(bins) <= total + 1e-5
    assert np.nanmax(bins) >= total - 1e-5


def test_compute_loss_requires_aux_targets_when_head_enabled():
    key = jax.random.key(0)
    config = _ltp_config()
    model = config.create(key)
    with pytest.raises(ValueError, match="aux_targets"):
        model.compute_loss(key, config.fake_obs(_BATCH), config.fake_act(_BATCH))


def test_head_disabled_is_bitwise_identical_to_baseline():
    key = jax.random.key(0)
    base_config = _config.Pi0TactileFastVitConfig(**_BASE)
    ltp_config = _ltp_config()
    base = base_config.create(key)
    ltp = ltp_config.create(key)

    # The LTP model only adds `tactile_future_head/*` parameters.
    base_flat = traverse_util.flatten_dict(nnx.state(base, nnx.Param).to_pure_dict())
    ltp_flat = traverse_util.flatten_dict(nnx.state(ltp, nnx.Param).to_pure_dict())
    extra = {"/".join(map(str, k)) for k in set(ltp_flat) - set(base_flat)}
    assert extra
    assert all(name.startswith("tactile_future_head/") for name in extra)
    assert not set(base_flat) - set(ltp_flat)

    # Share the common parameters; the flow loss and the sampled actions must then match exactly.
    graphdef, state = nnx.split(ltp)
    merged = {**ltp_flat, **base_flat}
    state.replace_by_pure_dict(traverse_util.unflatten_dict(merged))
    ltp = nnx.merge(graphdef, state)

    obs = _obs_with_aux(ltp_config, jax.random.key(1))
    act = ltp_config.fake_act(_BATCH)
    base_loss = nnx_utils.module_jit(base.compute_loss)(key, obs, act)
    ltp_losses = nnx_utils.module_jit(ltp.compute_loss)(key, obs, act)
    assert np.array_equal(np.asarray(base_loss), np.asarray(ltp_losses["flow"]))

    plain_obs = dataclasses.replace(obs, aux_targets=None)
    base_actions = nnx_utils.module_jit(base.sample_actions)(key, plain_obs, num_steps=2)
    ltp_actions = nnx_utils.module_jit(ltp.sample_actions)(key, plain_obs, num_steps=2)
    assert np.array_equal(np.asarray(base_actions), np.asarray(ltp_actions))


def test_tac_loss_gradient_stops_at_layer_m():
    """L_tac must reach blocks 1..m and the head, and nothing downstream of h^(m)."""
    key = jax.random.key(0)
    config = _ltp_config()
    model = config.create(key)
    obs = _obs_with_aux(config, jax.random.key(1))
    act = config.fake_act(_BATCH)

    def tac_loss(model, rng, obs, act):
        losses = model.compute_loss(rng, obs, act, train=True)
        return jnp.sum(losses["tac"] * losses["tac_mask"]) / jnp.sum(losses["tac_mask"])

    norms = _grad_path_norms(nnx.grad(tac_loss)(model, key, obs, act))

    # pi05 zero-initialises the adaRMS gates, so at init the attention/FFN weights get no
    # gradient from any loss; the gate projections are where the per-block signal shows.
    gate_kernels = [v for k, v in norms.items() if "/layers/" in k and "_1/Dense_0/kernel" in k]
    assert gate_kernels
    for per_layer in gate_kernels:
        assert per_layer.shape[0] == 4
        assert (per_layer[:_LAYER] > 0).all(), per_layer
        assert (per_layer[_LAYER:] == 0).all(), per_layer

    assert all(v > 0 for k, v in norms.items() if k.startswith("tactile_future_head/") and "bias" not in k)
    assert all(v == 0 for k, v in norms.items() if k.startswith("action_out_proj/"))
    assert all(v == 0 for k, v in norms.items() if "final_norm_1" in k)
    assert norms["action_in_proj/kernel"] > 0  # the noisy action tokens feed h^(m)


def test_kv_all_variant_reads_the_whole_suffix():
    key = jax.random.key(0)
    config = _ltp_config(tactile_future_kv="all")
    model = config.create(key)
    obs = _obs_with_aux(config, jax.random.key(1))
    losses = nnx_utils.module_jit(model.compute_loss)(key, obs, config.fake_act(_BATCH))
    assert losses["tac"].shape == (_BATCH, len(_HORIZONS), len(config.tactile_image_keys))


def test_suffix_hidden_is_the_pre_final_norm_residual_stream():
    key = jax.random.key(0)
    config = _ltp_config()
    model = config.create(key)
    obs = model._preprocess_observation(None, config.fake_obs(_BATCH), train=False)
    actions = config.fake_act(_BATCH)
    time = jnp.full((_BATCH,), 0.5)

    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar, adarms_cond = model.embed_suffix(obs, actions, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attn_mask = pi0.make_attn_mask(input_mask, jnp.concatenate([prefix_ar, suffix_ar], axis=0))
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (_, suffix_out), _, hidden = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
        return_suffix_hidden=True,
    )
    depth = 4
    assert hidden.shape == (depth, _BATCH, suffix_tokens.shape[1], suffix_tokens.shape[2])

    # The final RMSNorm's scale is zero-initialised, so normalising the last block's
    # residual stream must reproduce the module output (bf16 rounding aside).
    last = hidden[-1].astype(jnp.float32)
    normed = last * jax.lax.rsqrt(jnp.mean(jnp.square(last), axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(np.asarray(normed), np.asarray(suffix_out, dtype=np.float32), atol=2e-2, rtol=2e-2)

    # Without the switch the call signature is unchanged.
    outputs, _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    assert np.array_equal(np.asarray(outputs[1]), np.asarray(suffix_out))
