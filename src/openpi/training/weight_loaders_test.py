import numpy as np
import pytest

import openpi.models.model as _model
import openpi.shared.download as download
import openpi.training.weight_loaders as weight_loaders


def _checkpoint() -> dict:
    """A pi05-shaped checkpoint trained at action_dim 32."""
    return {
        "PaliGemma": {"llm": {"w": np.ones((4, 4), np.float32)}},
        "action_in_proj": {"kernel": np.ones((32, 8), np.float32), "bias": np.ones((8,), np.float32)},
        "action_out_proj": {"kernel": np.ones((8, 32), np.float32), "bias": np.ones((32,), np.float32)},
        "time_mlp_in": {"kernel": np.ones((8, 8), np.float32)},
    }


def _model_params(action_dim: int = 58) -> dict:
    """The model's own init at a wider action_dim, marked with zeros."""
    return {
        "PaliGemma": {"llm": {"w": np.zeros((4, 4), np.float32)}},
        "action_in_proj": {"kernel": np.zeros((action_dim, 8), np.float32), "bias": np.zeros((8,), np.float32)},
        "action_out_proj": {
            "kernel": np.zeros((8, action_dim), np.float32),
            "bias": np.zeros((action_dim,), np.float32),
        },
        "time_mlp_in": {"kernel": np.zeros((8, 8), np.float32)},
    }


@pytest.fixture
def checkpoint(monkeypatch):
    ckpt = _checkpoint()
    monkeypatch.setattr(download, "maybe_download", lambda path: path)
    monkeypatch.setattr(_model, "restore_params", lambda path, restore_type: ckpt)
    return ckpt


def test_wuji_loader_keeps_model_init_for_action_projections(checkpoint):
    params = _model_params()

    loaded = weight_loaders.WujiWeightLoader("ckpt").load(params)

    for name in ("action_in_proj", "action_out_proj"):
        for leaf in ("kernel", "bias"):
            assert loaded[name][leaf] is params[name][leaf]
    np.testing.assert_array_equal(loaded["PaliGemma"]["llm"]["w"], 1.0)
    np.testing.assert_array_equal(loaded["time_mlp_in"]["kernel"], 1.0)


def test_wuji_loader_rejects_other_shape_mismatches(checkpoint):
    params = _model_params()
    params["time_mlp_in"]["kernel"] = np.zeros((8, 16), np.float32)

    with pytest.raises(ValueError, match="time_mlp_in/kernel"):
        weight_loaders.WujiWeightLoader("ckpt").load(params)


def test_wuji_loader_rejects_a_regex_that_matches_nothing(checkpoint):
    with pytest.raises(ValueError, match="matches no model parameter"):
        weight_loaders.WujiWeightLoader("ckpt", reinit_regex="action_proj/.*").load(_model_params())
