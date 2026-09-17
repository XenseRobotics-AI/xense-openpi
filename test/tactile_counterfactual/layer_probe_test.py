"""Tests for the layer-probe machinery on the ``dummy`` Gemma variant (4 blocks, width 64).

The FastViT encoder is randomly initialised; no dataset or checkpoint is touched.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_tactile_fastvit_config as _config
from test.tactile_counterfactual import layer_probe as lp

_BATCH = 4
_DEPTH = 4


@pytest.fixture(scope="module")
def model():
    config = _config.Pi0TactileFastVitConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_dim=8,
        action_horizon=6,
        max_token_len=8,
    )
    return config, config.create(jax.random.key(0))


@pytest.fixture(scope="module")
def obs(model):
    config, _ = model
    obs = config.fake_obs(_BATCH)
    # Distinct rows so the roll-based variants change something.
    rng = np.random.default_rng(0)
    images = {k: jnp.asarray(rng.uniform(-1, 1, size=np.shape(v)).astype(np.float32)) for k, v in obs.images.items()}
    prompt = jnp.asarray(rng.integers(1, 100, size=np.shape(obs.tokenized_prompt)).astype(np.int32))
    return dataclasses.replace(obs, images=images, tokenized_prompt=prompt)


def test_forward_shapes_and_determinism(model, obs):
    config, module = model
    fwd = lp.LayerForward(module)
    assert fwd.layer_names == [f"l{i}" for i in range(_DEPTH + 1)]
    actions = np.asarray(config.fake_act(_BATCH))
    noise = lp.fixed_noise(0, _BATCH, config.action_horizon, config.action_dim)
    out = fwd(obs, actions, noise, 0.5)
    n_tac = len(config.tactile_image_keys)
    assert out["tactile_tokens"].shape == (_BATCH, n_tac, 64)
    assert out["action_stream"].shape == (_DEPTH + 1, _BATCH, config.action_horizon, 64)
    assert out["prefix_pooled"].shape[0] == _BATCH
    assert out["v_t"].shape == (_BATCH, config.action_horizon, config.action_dim)
    again = fwd(obs, actions, noise, 0.5)
    assert np.array_equal(np.asarray(out["action_stream"]), np.asarray(again["action_stream"]))
    # A different tau changes the suffix input (x_t) and therefore layer 0.
    other = fwd(obs, actions, noise, 1.0)
    assert not np.array_equal(np.asarray(out["action_stream"][0]), np.asarray(other["action_stream"][0]))


def test_variants_touch_only_their_fields(model, obs):
    config, _ = model
    keys = tuple(config.tactile_image_keys)
    rgb = [k for k in obs.images if k not in keys]

    shuffled = lp.make_variant(obs, "tac-shuffle", keys)
    lp.assert_variant_differs(obs, shuffled, keys)
    for k in rgb:
        assert np.array_equal(np.asarray(obs.images[k]), np.asarray(shuffled.images[k]))
    assert np.array_equal(np.asarray(shuffled.images[keys[0]]), np.roll(np.asarray(obs.images[keys[0]]), -1, axis=0))

    null = lp.make_variant(obs, "null", keys)
    assert not np.asarray(null.image_masks[keys[0]]).any()
    assert np.array_equal(np.asarray(null.images[keys[0]]), np.asarray(obs.images[keys[0]]))

    vl = lp.make_variant(obs, "vl-swap", keys)
    for k in keys:
        assert np.array_equal(np.asarray(obs.images[k]), np.asarray(vl.images[k]))
    assert np.array_equal(np.asarray(vl.images[rgb[0]]), np.roll(np.asarray(obs.images[rgb[0]]), -1, axis=0))
    assert np.array_equal(np.asarray(vl.tokenized_prompt), np.roll(np.asarray(obs.tokenized_prompt), -1, axis=0))

    pad = lp.make_variant(obs, "pad-pert", keys, pad_extra=2)
    lengths = np.asarray(obs.tokenized_prompt_mask).sum(1)
    assert np.array_equal(np.asarray(pad.tokenized_prompt_mask).sum(1), np.maximum(lengths - 2, 1))

    zero = lp.make_variant(obs, "tac-zero", keys)
    assert not np.asarray(zero.images[keys[0]]).any()

    swapped = lp.make_variant(obs, "pad-swap", keys)
    assert np.array_equal(np.asarray(swapped.images[keys[0]]), np.asarray(obs.images[keys[2]]))
    assert np.array_equal(np.asarray(swapped.images[keys[3]]), np.asarray(obs.images[keys[1]]))

    donor = lp.make_variant(obs, "tac-shuffle", keys)
    shifted = lp.make_variant(obs, "tac-timeshift", keys, donor=donor)
    assert np.array_equal(np.asarray(shifted.images[keys[1]]), np.asarray(donor.images[keys[1]]))
    with pytest.raises(ValueError, match="donor"):
        lp.make_variant(obs, "tac-timeshift", keys)
    with pytest.raises(ValueError, match="unknown variant"):
        lp.make_variant(obs, "bogus", keys)


def test_pad_pert_is_a_non_event_for_rgb_and_tactile(model, obs):
    """Masking the tail of the prompt leaves every other field alone (the red-herring control)."""
    config, _ = model
    keys = tuple(config.tactile_image_keys)
    pad = lp.make_variant(obs, "pad-pert", keys, pad_extra=1)
    for k in obs.images:
        assert obs.images[k] is pad.images[k]
    assert obs.tokenized_prompt is pad.tokenized_prompt


def test_sensitivity_metrics():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(6, 3, 5)).astype(np.float32)
    assert np.allclose(lp.sensitivity(real, real), 0.0, atol=1e-6)
    assert lp.cross_sample_sensitivity(real).shape == (6,)
    # A shared constant must not hide the per-sample change (raw cosine would).
    shifted = real + 1000.0
    other = shifted.copy()
    other[0] += 5.0
    s = lp.sensitivity(shifted, other)
    assert s[0] > 0.1
    assert np.allclose(s[1:], 0.0, atol=1e-3)
    layered = lp.layered_sensitivity(np.stack([real, real]), np.stack([real, other - 1000.0]))
    assert layered.shape == (2, 6)
    assert np.allclose(layered[0], 0.0, atol=1e-6)
    assert layered[1, 0] > 0.1
    stats = lp.summarize(np.arange(4.0), np.array([True, False, True, False]))
    assert stats == {"mean": 1.0, "std": 1.0, "n": 2}
    assert lp.summarize(np.arange(4.0), np.zeros(4, dtype=bool)) is None


def test_ridge_recovers_a_linear_map_and_reports_groups():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 32))
    w = rng.normal(size=(32, 2 * 3 * 4))
    y = x @ w + 0.01 * rng.normal(size=(300, 24))
    model, scores = lp.fit_ridge(x[:200], y[:200], x[200:], y[200:])
    assert set(scores) == set(lp.DEFAULT_ALPHAS)
    ev = lp.evaluate_ridge(model, x[200:], y[200:], (2, 3, 4))
    assert ev["nmse"] < 0.01
    assert len(ev["per_horizon"]) == 2
    assert len(ev["per_pad"]) == 3
    assert ev["n"] == 100
    # Dual form (more features than rows) equals the textbook primal solution.
    x = rng.normal(size=(30, 40))
    y = x @ rng.normal(size=(40, 3))
    x_tr, y_tr = x[:20], y[:20]
    dual, _ = lp.fit_ridge(x_tr, y_tr, x[20:], y[20:], alphas=(0.5,))
    xs = (x_tr - x_tr.mean(0)) / x_tr.std(0)
    lam = 0.5 * float(np.sum(np.square(xs)) / xs.shape[1])
    reference = np.linalg.solve(xs.T @ xs + lam * np.eye(40), xs.T @ (y_tr - y_tr.mean(0)))
    assert dual.lam == pytest.approx(lam)
    np.testing.assert_allclose(dual.weight, reference, rtol=1e-8, atol=1e-8)
    assert dual.weight.shape == (40, 3)


def test_split_episodes_holds_out_whole_episodes():
    episodes = np.repeat(np.arange(8), 3)
    is_val = lp.split_episodes(episodes, 0.25, np.random.default_rng(0))
    assert is_val.shape == episodes.shape
    for ep in np.unique(episodes):
        assert len(np.unique(is_val[episodes == ep])) == 1
    assert 0 < is_val.sum() < len(episodes)


class _FakeDataset:
    def __init__(self, lengths: dict[int, int]) -> None:
        self._lengths = lengths

    @property
    def episodes(self) -> list[int]:
        return sorted(self._lengths)

    def episode_length(self, ep: int) -> int:
        return self._lengths[ep]

    def has_sample(self, ep: int, frame: int) -> bool:
        return 0 <= frame < self._lengths[ep]


def test_sample_frames_distinct_episodes_per_batch_and_tail():
    ds = _FakeDataset({e: 100 + e for e in range(12)})
    refs = lp.sample_frames(ds, num_frames=20, batch_size=4, rng=np.random.default_rng(0), min_tail=50)
    assert len(refs) == 20
    assert refs.num_batches == 5
    for b in range(refs.num_batches):
        eps = refs.episode[refs.batch(b)]
        assert len(np.unique(eps)) == len(eps)
    assert np.all(refs.frame + 50 < np.array([ds.episode_length(int(e)) for e in refs.episode]))
    assert refs.contact is None
    with pytest.raises(ValueError, match="batch_size"):
        lp.sample_frames(ds, num_frames=4, batch_size=1, rng=np.random.default_rng(0))


def test_future_tactile_store_targets_and_contact(tmp_path):
    n_ep, length, pads = 3, 40, 4
    offsets = np.arange(0, (n_ep + 1) * length, length, dtype=np.int64)
    rng = np.random.default_rng(0)
    field = rng.integers(0, 256, size=(n_ep * length, pads, 4, 4, 3), dtype=np.uint8)
    # Episode 1 is constant, so its delta proxy is exactly 0.
    field[length : 2 * length] = 7
    np.save(tmp_path / "pixel_field.npy", field)
    np.save(tmp_path / "episode_offsets.npy", offsets)
    horizons = [5, 10]
    meta = {
        "num_pads": pads,
        "horizons": horizons,
        "pca_dim": 8,
        "y_delta_rms": {str(k): [0.1] * pads for k in horizons},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    store = lp.FutureTactileStore(tmp_path, horizons, target="pixel_delta")
    assert store.num_episodes == n_ep
    assert store.num_pads == pads
    assert store.target_dim() == 48
    z, mask = store.targets(0, 33)
    assert z.shape == (2, pads, 48)
    assert mask.tolist() == [True, False]
    scores = store.contact_scores(np.array([0, 1]), np.array([3, 3]), "delta")
    assert scores.shape == (2,)
    assert scores[0] > 0
    assert scores[1] == 0
    state = store.contact_scores(np.array([1, 0]), np.array([0, 0]), "state")
    assert np.all(state == 0)  # frame 0 vs itself
    with pytest.raises(ValueError, match="proxy"):
        store.contact_scores(np.array([0]), np.array([0]), "bogus")

    ds = _FakeDataset(dict.fromkeys(range(n_ep), length))
    refs = lp.sample_frames(
        ds, num_frames=6, batch_size=2, rng=np.random.default_rng(0), min_tail=10, store=store, pool_size=64
    )
    assert refs.contact is not None
    assert refs.contact.shape == (6,)
    assert refs.contact_threshold is not None
    payload = refs.to_dict()
    assert len(payload["contact"]) == 6


def test_markdown_and_json_helpers():
    table = lp.markdown_table(["a", "b"], [["x", 0.5], ["y", None]])
    assert table.splitlines()[0] == "| a | b |"
    assert "n/a" in table
    out = lp.to_jsonable({"a": np.float32(1.5), "b": np.arange(2), "c": (1, 2)})
    assert out == {"a": 1.5, "b": [0, 1], "c": [1, 2]}
