import pathlib

import pytest

from openpi.rlt import config as _config


def test_example_parses():
    config = _config.load(pathlib.Path(_config.__file__).resolve().parents[3] / "configs" / "rlt" / "_example.yaml")
    assert config.name == "_example"
    assert config.model == _config.RLTModelConfig()
    assert config.rl == _config.RLConfig()
    assert config.token_training.frame_stride == 1


def test_unknown_field_fails():
    with pytest.raises(TypeError, match="image_only"):
        _config.loads(
            "token_training: {vla_config: a, vla_checkpoint: b, prefix_cache_dir: c}\nmodel: {image_only: true}", "x"
        )


def test_missing_config_suggests_close_name():
    with pytest.raises(ValueError, match="not found"):
        _config.get_config("definitely_not_a_config")
