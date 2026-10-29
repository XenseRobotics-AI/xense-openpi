import dataclasses

import numpy as np
import pytest

from openpi.training import config as _config
import openpi.transforms as transforms


@pytest.mark.parametrize("use_delta", [True, False])
def test_bi_elite_dataset_pipeline_roundtrip(monkeypatch, tmp_path, use_delta):
    # Keep this test offline; tokenizer/model transforms are shared and unchanged.
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    config = _config.get_config("pi05_base_bi_elite_egg_carton_to_tray_1012_h100")
    assert isinstance(config.data, _config.LeRobotBiEliteDataConfig)
    assert config.data.repo_id == "Xense/bi_elite_cs66_rt_egg_carton_to_tray"
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
        "task": "Open the egg carton.",
        **{
            f"observation.images.{camera}": np.full((3, 8, 8), value, dtype=np.uint8)
            for value, camera in enumerate(("head", "left_wrist", "right_wrist"), start=1)
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
        # Per-side-grouped: TCPs relative, grippers (dims 9 and 19) absolute.
        expected[:, 0:9] -= state[0:9]
        expected[:, 10:19] -= state[10:19]
    np.testing.assert_array_equal(result["actions"], expected)

    result["actions"] = np.pad(result["actions"], ((0, 0), (0, 12)))
    restored = transforms.compose(data_config.data_transforms.outputs)(result)
    np.testing.assert_allclose(restored["actions"], actions)
