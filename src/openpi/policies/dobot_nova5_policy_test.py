import pathlib

import numpy as np
import pytest

import openpi.models.pi0_config as pi0_config
from openpi.policies import dobot_nova5_policy
from openpi.training import config as _config
from openpi.training import registry
import openpi.transforms as _transforms


def _example(*, with_actions: bool = False) -> dict:
    example = dobot_nova5_policy.make_dobot_nova5_example()
    if with_actions:
        example["actions"] = np.zeros((50, dobot_nova5_policy.STATE_DIM), dtype=np.float32)
    return example


def test_inputs_map_single_wrist_to_right_slot_and_mask_left_slot():
    result = dobot_nova5_policy.DobotNova5Inputs()(_example(with_actions=True))

    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.True_,
    }
    assert not result["image"]["left_wrist_0_rgb"].any()
    assert result["state"].shape == (10,)
    assert result["actions"].shape == (50, 10)


def test_inputs_require_exact_dataset_camera_contract():
    example = _example()
    del example["images"]["wrist"]
    with pytest.raises(ValueError, match="Missing required Nova5 cameras"):
        dobot_nova5_policy.DobotNova5Inputs()(example)


def test_outputs_trim_model_actions_to_ten_dims():
    padded = np.random.rand(50, 32)
    result = dobot_nova5_policy.DobotNova5Outputs()({"actions": padded})

    assert result["actions"].shape == (50, 10)
    np.testing.assert_array_equal(result["actions"], padded[:, :10])


def test_delta_mask_keeps_only_gripper_absolute():
    mask = _transforms.make_bool_mask(9, -1)

    assert len(mask) == dobot_nova5_policy.STATE_DIM
    assert all(mask[dobot_nova5_policy.TCP])
    assert not any(mask[dobot_nova5_policy.GRIPPER])


def test_data_config_matches_lerobot_dataset_columns(tmp_path: pathlib.Path):
    factory = _config.LeRobotDobotNova5DataConfig(repo_id="Xense/loreal_returns_sorting_0831")
    data_config = factory.create(tmp_path, pi0_config.Pi0Config(pi05=True))

    assert registry.resolve(registry.all_registries()["DATA_CONFIGS"], "LeRobotDobotNova5DataConfig") is type(factory)
    assert data_config.repack_transforms.inputs[0].structure == {
        "images": {
            "head": "observation.images.head",
            "wrist": "observation.images.wrist",
        },
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
    }
    assert [type(transform) for transform in data_config.data_transforms.inputs] == [
        dobot_nova5_policy.DobotNova5Inputs,
        _transforms.DeltaActions,
    ]
    assert [type(transform) for transform in data_config.data_transforms.outputs] == [
        _transforms.AbsoluteActions,
        dobot_nova5_policy.DobotNova5Outputs,
    ]
