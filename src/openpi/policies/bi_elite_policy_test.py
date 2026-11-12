"""Pins the BiElite 20D per-side-grouped dim layout that its docstrings describe."""

import numpy as np
import pytest

from openpi.policies import bi_elite_policy
import openpi.transforms as _transforms


def test_delta_mask_keeps_per_side_grippers_absolute():
    mask = bi_elite_policy.DELTA_ACTION_MASK

    assert len(mask) == bi_elite_policy.STATE_DIM
    absolute = [index for index, is_delta in enumerate(mask) if not is_delta]
    assert absolute == [bi_elite_policy.LEFT_GRIPPER.start, bi_elite_policy.RIGHT_GRIPPER.start] == [9, 19]
    # Guard against copying BiFlexiv's mask, which would delta-encode the left gripper.
    assert mask != _transforms.make_bool_mask(18, -1, -1)


def test_outputs_trim_to_20_dims():
    padded = np.random.rand(10, 32)
    result = bi_elite_policy.BiEliteOutputs()({"actions": padded})

    assert result["actions"].shape == (10, 20)
    np.testing.assert_array_equal(result["actions"], padded[:, :20])


def test_inputs_map_cameras_and_fill_missing_wrist():
    example = bi_elite_policy.make_bi_elite_example()
    del example["images"]["left_wrist"]

    result = bi_elite_policy.BiEliteInputs()(example)

    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.False_
    assert not result["image"]["left_wrist_0_rgb"].any()
    assert result["image_mask"]["right_wrist_0_rgb"] == np.True_
    assert result["prompt"] == "do something"


def test_inputs_require_head_and_reject_unknown_cameras():
    example = bi_elite_policy.make_bi_elite_example()
    del example["images"]["head"]
    with pytest.raises(ValueError, match="head"):
        bi_elite_policy.BiEliteInputs()(example)

    example = bi_elite_policy.make_bi_elite_example()
    example["images"]["left_tactile_left"] = example["images"]["head"]
    with pytest.raises(ValueError, match="Expected images"):
        bi_elite_policy.BiEliteInputs()(example)
