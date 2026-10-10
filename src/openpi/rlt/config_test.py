import pathlib

import pytest

from openpi.rlt import config as _config


def test_example_parses():
    config = _config.load(pathlib.Path(_config.__file__).resolve().parents[3] / "configs" / "rlt" / "_example.yaml")
    assert config.name == "_example"
    assert config.model == _config.RLTModelConfig()
    assert config.rl == _config.RLConfig()
    assert config.rl.input_reference == "corrected"
    assert config.token_training.frame_stride == 1


def test_unknown_field_fails():
    with pytest.raises(TypeError, match="image_only"):
        _config.loads(
            "token_training: {vla_config: a, vla_checkpoint: b, prefix_cache_dir: c}\nmodel: {image_only: true}", "x"
        )


def test_missing_config_suggests_close_name():
    with pytest.raises(ValueError, match="not found"):
        _config.get_config("definitely_not_a_config")


@pytest.mark.parametrize("mode", ["corrected", "proposal"])
def test_input_reference_yaml_and_training_contract(mode):
    config = _config.loads(
        f"token_training: {{vla_config: a, vla_checkpoint: b, prefix_cache_dir: c}}\nrl: {{input_reference: {mode}}}",
        "x",
    )
    assert config.rl.input_reference == mode
    assert config.rl.training_contract()["input_reference"] == mode


def test_invalid_input_reference_fails():
    with pytest.raises(ValueError, match="input_reference"):
        _config.loads(
            "token_training: {vla_config: a, vla_checkpoint: b, prefix_cache_dir: c}\nrl: {input_reference: typo}", "x"
        )
