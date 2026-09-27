"""Exercise real three-expert Gemma with inexpensive image encoders on CPU."""

import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_tactile_expert as cascade
from openpi.models.pi0_tactile_expert_config import Pi0TactileExpertConfig
from openpi.shared import nnx_utils
from openpi.training import weight_loaders


class TinyImage(nn.Module):
    width: int

    @nn.compact
    def __call__(self, images, train=False):
        return nn.Dense(self.width)(images.mean((1, 2)))[:, None, :], None


class TinyTactile(nnx.Module):
    feature_dim = 8

    def __init__(self, rngs, **kwargs):
        self.proj = nnx.Linear(3, self.feature_dim, rngs=rngs)

    def __call__(self, images):
        return self.proj(images.mean((1, 2)))


@pytest.fixture
def tiny(monkeypatch, request):
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(pi0._siglip, "Module", lambda num_classes, **kwargs: TinyImage(num_classes))
    monkeypatch.setattr(cascade, "build_tactile_encoder", lambda name, **kwargs: TinyTactile(**kwargs))
    config = Pi0TactileExpertConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        tactile_expert_variant="dummy",
        dtype=getattr(request, "param", "float32"),
        action_dim=3,
        action_horizon=2,
        max_token_len=2,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    # Tiny spatial sizes avoid image overhead; bypass only preprocessing.
    obs = dataclasses.replace(obs, images={k: v[:, :2, :2] for k, v in obs.images.items()})
    monkeypatch.setattr(cascade.Pi0TactileExpert, "_preprocess_observation", lambda self, rng, obs, train: obs)
    return config, model, obs


def test_config_validation():
    assert Pi0TactileExpertConfig().cascade_split_step == 6
    for kwargs in (
        {"pi05": False},
        {"enable_training_time_rtc": True},
        {"cascade_split_step": 11},
        {"cascade_total_steps": 0},
        {"tactile_dropout": 1},
        {"tactile_loss_weight": -1},
        {"tactile_expert_variant": "dummy"},
        {"action_expert_variant": "gemma_300m_lora"},
    ):
        with pytest.raises(ValueError, match=r".+"):
            Pi0TactileExpertConfig(**kwargs)


@pytest.mark.parametrize("tiny", ["float32", "bfloat16"], indirect=True)
def test_jitted_loss_and_sampling(tiny):
    config, model, obs = tiny
    actions = config.fake_act(batch_size=2)
    loss, aux = nnx_utils.module_jit(model.compute_loss_with_aux)(jax.random.key(1), obs, actions)
    assert loss.shape == (2, 2)
    assert np.isfinite(loss).all()
    np.testing.assert_allclose(loss.mean(), aux["loss_action"] + aux["loss_tactile"], rtol=1e-6)
    sample = nnx_utils.module_jit(model.sample_actions)(jax.random.key(2), obs, num_steps=jnp.asarray(10))
    assert sample.shape == actions.shape
    assert np.isfinite(sample).all()


def test_six_plus_four_and_refresh(tiny, monkeypatch):
    _, model, obs = tiny
    recorded = []

    def velocity(self, obs, x, time, mask, cache):
        jax.debug.callback(lambda t: recorded.append(float(t)), time, ordered=True)
        return jnp.ones_like(x), cache

    def tactile_velocity(self, x, time, tactile, mask, cache):
        return jnp.full_like(x, 2)

    monkeypatch.setattr(cascade.Pi0TactileExpert, "_action_forward", velocity)
    monkeypatch.setattr(cascade.Pi0TactileExpert, "_tactile_velocity", tactile_velocity)
    noise = jnp.zeros((2, 2, 3))
    sample = nnx_utils.module_jit(model.sample_actions)(jax.random.key(1), obs, noise=noise)
    np.testing.assert_allclose(sample, -1.4, atol=1e-6)
    np.testing.assert_allclose(recorded, [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], atol=1e-6)


def test_action_blindness_and_masked_tactile(tiny):
    config, model, obs = tiny
    changed = dataclasses.replace(
        obs,
        images={
            k: v + jnp.array([2.0, -1.0, 0.5]) if k in config.tactile_image_keys else v for k, v in obs.images.items()
        },
    )
    actions = config.fake_act(batch_size=2)
    loss = nnx_utils.module_jit(model.compute_loss_with_aux)
    _, original = loss(jax.random.key(1), obs, actions)
    _, altered = loss(jax.random.key(1), changed, actions)
    np.testing.assert_array_equal(original["loss_action"], altered["loss_action"])
    assert not np.allclose(original["loss_tactile"], altered["loss_tactile"])
    masks = {k: jnp.zeros_like(v) if k in config.tactile_image_keys else v for k, v in obs.image_masks.items()}
    sample = nnx_utils.module_jit(model.sample_actions)
    a = sample(jax.random.key(2), dataclasses.replace(obs, image_masks=masks))
    b = sample(jax.random.key(2), dataclasses.replace(changed, image_masks=masks))
    np.testing.assert_array_equal(a, b)


def test_tactile_loss_gradient_isolation(tiny):
    config, model, obs = tiny

    def loss_t(model):
        return model.compute_loss_with_aux(jax.random.key(1), obs, config.fake_act(batch_size=2))[1]["loss_tactile"]

    grads = nnx.jit(nnx.grad(loss_t))(model)
    flat = grads.flat_state()
    tactile_norm = 0.0
    encoder_norm = 0.0
    for path, value in flat.items():
        names = [str(p) for p in path]
        array = np.asarray(value.value)
        assert np.isfinite(array).all(), path
        if "tactile_encoder" in names or "tactile_proj" in names:
            encoder_norm += np.abs(array).sum()
        if any(n.endswith(("_2", "_tac")) for n in names):
            tactile_norm += np.abs(array).sum()
        elif "tactile_encoder" not in names and "tactile_proj" not in names:
            np.testing.assert_array_equal(array, np.zeros_like(array), err_msg=str(path))
    assert tactile_norm > 0
    assert encoder_norm > 0


def test_all_action_matches_base(tiny):
    config, model, obs = tiny
    model.cascade_split_step = 10
    # Use the same weights in a genuine two-expert base model.
    base_config = pi0.pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=3,
        action_horizon=2,
        max_token_len=2,
    )
    base = base_config.create(jax.random.key(9))
    reference = nnx.state(base).to_pure_dict()
    loaded = weight_loaders._merge_params(nnx.state(model).to_pure_dict(), reference, missing_regex=".*")
    state = nnx.state(base)
    state.replace_by_pure_dict(loaded)
    nnx.update(base, state)
    # Prefix filtering/preprocessing in cascade is tested separately.
    clean_obs = dataclasses.replace(
        obs,
        images={k: v for k, v in obs.images.items() if k not in config.tactile_image_keys},
        image_masks={k: v for k, v in obs.image_masks.items() if k not in config.tactile_image_keys},
    )
    base._preprocess_observation = lambda rng, observation, train: observation
    noise = jax.random.normal(jax.random.key(5), (2, 2, 3))
    actual = nnx_utils.module_jit(model.sample_actions)(jax.random.key(2), obs, noise=noise)
    expected = nnx_utils.module_jit(base.sample_actions)(jax.random.key(2), clean_obs, noise=noise)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_loader_preserves_new_leaves_and_copy_precedence(tiny, monkeypatch):
    _, model, _ = tiny
    params = nnx.state(model).to_pure_dict()
    import flax.traverse_util

    flat = flax.traverse_util.flatten_dict(params, sep="/")
    base_flat = {
        k: np.asarray(v)
        for k, v in flat.items()
        if "tactile" not in k and "_tac/" not in k and not any(s.endswith("_2") for s in k.split("/"))
    }
    base = flax.traverse_util.unflatten_dict(base_flat, sep="/")
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *a, **kw: base)
    loaded = weight_loaders.CascadeInitWeightLoader("unused").load(params)
    assert jax.tree.structure(loaded) == jax.tree.structure(params)
    copied = weight_loaders.CascadeInitWeightLoader("unused", tactile_expert_init="copy_action").load(params)
    np.testing.assert_array_equal(copied["action_in_proj_tac"]["kernel"], base["action_in_proj"]["kernel"])
    full = weight_loaders._copy_action_expert(params, params)
    for actual, expected in zip(jax.tree.leaves(full), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_prefix_cache_reuse_is_equivalent(tiny):
    config, model, obs = tiny
    obs = dataclasses.replace(obs, tokenized_prompt_mask=jnp.array([[True, False], [False, True]]))
    prefix, pm, pa = model.embed_prefix(obs)
    suffix, sm, sa, cond = model.embed_suffix(obs, config.fake_act(batch_size=2), jnp.ones(2))
    mask = jnp.concatenate([pm, sm], axis=1)
    _, full = model.PaliGemma.llm(
        [prefix, suffix, None],
        mask=pi0.make_attn_mask(mask, jnp.concatenate([pa, sa])),
        positions=mask.cumsum(axis=1) - 1,
        adarms_cond=[None, cond, None],
    )
    _, separate = model.PaliGemma.llm(
        [prefix, None, None],
        mask=pi0.make_attn_mask(pm, pa),
        positions=pm.cumsum(axis=1) - 1,
    )
    for a, b in zip(full, separate, strict=True):
        # Fully masked padding queries softmax to a uniform distribution in
        # explicit_attention. Their unused KV can differ with sequence length;
        # only valid keys must agree, and padding stays masked in both stages.
        valid = pm[None, :, :, None, None]
        np.testing.assert_allclose(
            jnp.where(valid, a[:, :, : prefix.shape[1]], 0), jnp.where(valid, b, 0), atol=1e-6, rtol=1e-6
        )
    reused = jax.tree.map(lambda v: v[:, :, : prefix.shape[1]], full)
    actions = config.fake_act(batch_size=2)
    # Check the nonzero-gated tactile expert, not the untrained action gates.
    context_a, cache_a = model._split_context(obs, actions, jnp.asarray(0.4), pm, reused)
    context_b, cache_b = model._split_context(obs, actions, jnp.asarray(0.4), pm, separate)
    tactile = model.embed_tactile(obs)
    np.testing.assert_allclose(
        model._tactile_velocity(actions, jnp.asarray(0.4), tactile, context_a, cache_a),
        model._tactile_velocity(actions, jnp.asarray(0.4), tactile, context_b, cache_b),
        atol=1e-6,
        rtol=1e-6,
    )
