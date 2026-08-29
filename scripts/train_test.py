import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp

from openpi.training import config as _config

from . import train


def test_nonfinite_update_is_rejected():
    loss = jnp.asarray(1.0)
    grads = {"finite": jnp.ones((2,)), "bad": jnp.array([0.0, jnp.nan])}
    updates = {"finite": jnp.ones((2,)), "bad": jnp.zeros((2,))}

    update_is_finite = train._all_finite((loss, grads, updates))

    assert not bool(update_is_finite)


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
