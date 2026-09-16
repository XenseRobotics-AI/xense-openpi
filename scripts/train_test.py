import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


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


def test_aux_loss_weight_warms_up_linearly():
    import jax.numpy as jnp

    config = dataclasses.replace(_config.get_config("debug_pi05"), aux_loss_weight=2.0, aux_loss_warmup_steps=10)
    assert float(train._aux_loss_weight(config, jnp.asarray(0))) == pytest.approx(0.2)
    assert float(train._aux_loss_weight(config, jnp.asarray(9))) == pytest.approx(2.0)
    assert float(train._aux_loss_weight(config, jnp.asarray(500))) == pytest.approx(2.0)
    no_ramp = dataclasses.replace(config, aux_loss_warmup_steps=0)
    assert float(train._aux_loss_weight(no_ramp, jnp.asarray(0))) == pytest.approx(2.0)


def test_combine_losses_weights_masked_tac():
    import jax.numpy as jnp
    import numpy as np

    losses = {
        "flow": jnp.array([[1.0, 3.0]]),
        "tac": jnp.array([[[2.0, 4.0], [100.0, 100.0]]]),
        "tac_mask": jnp.array([[[True, True], [False, False]]]),
        "tac_by_time": jnp.array([np.nan, 3.0]),
    }
    total, metrics = train._combine_losses(losses, jnp.asarray(0.5))
    assert float(total) == pytest.approx(2.0 + 0.5 * 3.0)
    assert float(metrics["loss/flow"]) == pytest.approx(2.0)
    assert float(metrics["loss/tac"]) == pytest.approx(3.0)
    assert np.isnan(float(metrics["tac/time_bin0"]))
    assert float(metrics["tac/time_bin1"]) == pytest.approx(3.0)
    assert train._reduce_with_nanmean("tac/time_bin0")
    assert train._reduce_with_nanmean("aux/grad_ratio")
    assert not train._reduce_with_nanmean("loss/tac")
