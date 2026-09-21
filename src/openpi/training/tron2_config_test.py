import dataclasses

import numpy as np
import pytest

from openpi.training import config as _config
import openpi.transforms as transforms


@pytest.mark.parametrize("use_delta", [True, False])
def test_tron2_dataset_pipeline_roundtrip(monkeypatch, tmp_path, use_delta):
    # Keep this test offline; tokenizer/model transforms are shared and unchanged.
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    config = _config.get_config("pi05_base_tron2rt_pnp_0918")
    assert isinstance(config.data, _config.LeRobotTron2DataConfig)
    assert config.data.repo_id == "Xense/tron2rt-pnp-0918"
    factory = dataclasses.replace(config.data, use_delta_cartesian_actions=use_delta)
    data_config = factory.create(tmp_path, config.model)
    assert data_config.prompt_from_task
    assert not data_config.tactile
    assert data_config.action_sequence_keys == ("action",)

    state = np.arange(20, dtype=np.float32)
    actions = np.stack([state + 1, state + 2])
    sample = {
        "observation.state": state,
        "action": actions.copy(),
        "task": "Pick and place.",
        **{
            f"observation.images.{camera}": np.full((3, 8, 8), value, dtype=np.uint8)
            for value, camera in enumerate(("top", "left_wrist", "right_wrist"), start=1)
        },
    }
    result = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])(sample)
    for value, slot in enumerate(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), start=1):
        np.testing.assert_array_equal(result["image"][slot], np.full((8, 8, 3), value, dtype=np.uint8))
        assert result["image_mask"][slot]
    assert result["prompt"] == sample["task"]
    np.testing.assert_array_equal(result["state"], state)
    expected = actions.copy()
    if use_delta:
        expected[:, :18] -= state[:18]
    np.testing.assert_array_equal(result["actions"], expected)

    result["actions"] = np.pad(result["actions"], ((0, 0), (0, 12)))
    restored = transforms.compose(data_config.data_transforms.outputs)(result)
    np.testing.assert_allclose(restored["actions"], actions)
