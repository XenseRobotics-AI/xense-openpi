import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import optimizer as _optimizer


class _TwoLinear(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.body = nnx.Linear(4, 4, rngs=rngs)
        self.tactile_future_head = nnx.Linear(4, 2, rngs=rngs)


def test_path_mask_selects_nnx_state_leaves_by_joined_path():
    params = nnx.state(_TwoLinear(nnx.Rngs(0)), nnx.Param)
    mask = _optimizer._path_mask_fn(".*tactile_future_head.*")(params)
    selected = {_optimizer.join_path(path) for path, value in jax.tree_util.tree_leaves_with_path(mask) if bool(value)}
    assert selected == {"tactile_future_head/kernel", "tactile_future_head/bias"}


def test_lr_scales_multiply_the_update_of_matching_params():
    params = {"body": {"kernel": jnp.ones((3,))}, "tactile_future_head": {"kernel": jnp.ones((3,))}}
    grads = jax.tree.map(jnp.ones_like, params)
    schedule = _optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=1e-3, decay_steps=10)

    plain = _optimizer.create_optimizer(_optimizer.AdamW(), schedule)
    scaled = _optimizer.create_optimizer(_optimizer.AdamW(), schedule, lr_scales={".*tactile_future_head.*": 4.0})

    plain_updates, _ = plain.update(grads, plain.init(params), params)
    scaled_updates, _ = scaled.update(grads, scaled.init(params), params)

    np.testing.assert_allclose(scaled_updates["body"]["kernel"], plain_updates["body"]["kernel"])
    np.testing.assert_allclose(
        scaled_updates["tactile_future_head"]["kernel"], 4.0 * plain_updates["tactile_future_head"]["kernel"]
    )


def test_lr_scales_rejects_non_positive():
    schedule = _optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=1e-3, decay_steps=10)
    with pytest.raises(ValueError, match="positive"):
        _optimizer.create_optimizer(_optimizer.AdamW(), schedule, lr_scales={".*": 0.0})
