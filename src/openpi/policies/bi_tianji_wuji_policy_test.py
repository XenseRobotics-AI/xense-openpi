"""Pins the Tianji/Wuji 58D layout: 86D recorded state truncated, TCPs delta, fingers absolute."""

import pathlib

import numpy as np
import pytest

import openpi.models.pi0_config as pi0_config
from openpi.policies import bi_flexiv_policy
from openpi.policies import bi_tianji_wuji_policy
import openpi.training.config as _config
import openpi.transforms as _transforms


def _example(*, state_dim: int = 86, horizon: int = 4) -> dict:
    example = bi_flexiv_policy.make_bi_flexiv_example()
    example["state"] = np.arange(state_dim, dtype=np.float32)
    example["actions"] = np.full((horizon, 58), 100.0, dtype=np.float32)
    return example


def _data_transforms(*, action_dim: int = 58) -> _transforms.Group:
    factory = _config.LeRobotBiTianjiWujiDataConfig(repo_id="Xense/TW-block-sort-0918")
    model = pi0_config.Pi0Config(pi05=True, action_dim=action_dim)
    return factory.create(pathlib.Path("/nonexistent"), model).data_transforms


def test_delta_mask_covers_both_tcps_only():
    mask = _transforms.make_bool_mask(18, -40)

    assert len(mask) == 58
    assert [i for i, is_delta in enumerate(mask) if is_delta] == list(range(18))


def test_pipeline_truncates_state_and_deltas_tcps_only():
    result = _transforms.compose(_data_transforms().inputs)(_example())

    np.testing.assert_array_equal(result["state"], np.arange(58))
    # TCP dims become action - state, finger dims pass through unchanged.
    np.testing.assert_array_equal(result["actions"][:, :18], 100.0 - np.arange(18)[None, :].repeat(4, 0))
    np.testing.assert_array_equal(result["actions"][:, 18:], 100.0)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}


def test_inputs_reject_untruncated_state():
    with pytest.raises(ValueError, match="TruncateState"):
        bi_tianji_wuji_policy.BiTianjiWujiInputs()(_example())


def test_inputs_reject_wrong_action_width():
    example = _example(state_dim=58)
    example["actions"] = np.zeros((4, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="58"):
        bi_tianji_wuji_policy.BiTianjiWujiInputs()(example)


def test_outputs_trim_to_58_dims():
    padded = np.random.rand(10, 64)
    result = bi_tianji_wuji_policy.BiTianjiWujiOutputs()({"actions": padded})

    np.testing.assert_array_equal(result["actions"], padded[:, :58])


def test_data_config_rejects_narrow_model():
    with pytest.raises(ValueError, match="action_dim >= 58"):
        _data_transforms(action_dim=32)
