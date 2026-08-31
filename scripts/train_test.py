import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import config as _config

from . import train


def test_add_relative_gradient_noise():
    grads = {
        "a": jnp.arange(1, 13, dtype=jnp.float32).reshape(3, 4),
        "b": jnp.asarray([-3.0, 0.0, 5.0], dtype=jnp.float32),
    }
    scale = 0.015
    key = jax.random.key(7)

    noisy, reported_rel_norm = train._add_relative_gradient_noise(grads, key, scale)
    noise = jax.tree.map(lambda perturbed, clean: perturbed - clean, noisy, grads)
    measured_rel_norm = np.sqrt(sum(float(jnp.sum(x**2)) for x in jax.tree.leaves(noise))) / np.sqrt(
        sum(float(jnp.sum(x**2)) for x in jax.tree.leaves(grads))
    )

    assert float(reported_rel_norm) == pytest.approx(scale, rel=1e-6)
    assert measured_rel_norm == pytest.approx(scale, rel=1e-6)
    same_noise, _ = train._add_relative_gradient_noise(grads, key, scale)
    different_noise, _ = train._add_relative_gradient_noise(grads, jax.random.key(8), scale)
    for actual, expected in zip(jax.tree.leaves(same_noise), jax.tree.leaves(noisy)):
        np.testing.assert_array_equal(actual, expected)
    assert any(
        not np.array_equal(actual, expected)
        for actual, expected in zip(jax.tree.leaves(different_noise), jax.tree.leaves(noisy))
    )


@pytest.mark.parametrize("config_name", ["debug_pi05"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config.get_config(config_name),
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
