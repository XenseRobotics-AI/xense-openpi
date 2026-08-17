"""Unit tests for the XTac-UMI transforms and the layout they agree on.

The layout is the whole contract here: the recording rig, the data config's delta
mask and the Flexiv deployment example all have to agree on which of the 20 dims
is which, and an off-by-nine between them is silent — it trains and it runs, it
just moves the wrong arm. These tests pin the agreement.
"""

import numpy as np
import pytest

from openpi.policies import xtac_umi_policy
from openpi.training import config as _config
import openpi.transforms as _transforms


def _example(*, with_head: bool = False, with_actions: bool = False) -> dict:
    example = xtac_umi_policy.make_xtac_umi_example()
    if with_head:
        example["images"]["head"] = np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)
    if with_actions:
        example["actions"] = np.zeros((10, xtac_umi_policy.STATE_DIM))
    return example


def test_inputs_mask_base_slot_by_default():
    """No head camera: base_0_rgb is a black image the model is told to ignore."""
    result = xtac_umi_policy.XTacUmiInputs()(_example())

    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image_mask"]["base_0_rgb"] == np.False_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
    assert result["image_mask"]["right_wrist_0_rgb"] == np.True_
    # Black, and the same shape as a real frame so the model sees a consistent slot.
    assert not result["image"]["base_0_rgb"].any()
    assert result["image"]["base_0_rgb"].shape == result["image"]["left_wrist_0_rgb"].shape


def test_inputs_use_head_camera_fills_and_unmasks_base_slot():
    result = xtac_umi_policy.XTacUmiInputs(use_head_camera=True)(_example(with_head=True))

    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image"]["base_0_rgb"].any()


def test_inputs_head_camera_ignored_unless_enabled():
    """A client may send `head` for compatibility; it must not leak into the slot."""
    result = xtac_umi_policy.XTacUmiInputs()(_example(with_head=True))

    assert result["image_mask"]["base_0_rgb"] == np.False_
    assert not result["image"]["base_0_rgb"].any()


def test_inputs_reject_missing_wrist_and_missing_head():
    example = _example()
    del example["images"]["left_wrist"]
    with pytest.raises(ValueError, match="Missing required wrist cameras"):
        xtac_umi_policy.XTacUmiInputs()(example)

    with pytest.raises(ValueError, match="use_head_camera=True but no 'head' image"):
        xtac_umi_policy.XTacUmiInputs(use_head_camera=True)(_example())

    example = _example()
    example["images"]["belly"] = np.zeros((3, 224, 224), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unexpected cameras"):
        xtac_umi_policy.XTacUmiInputs()(example)


def test_inputs_convert_images_to_hwc_uint8():
    example = _example()
    example["images"]["left_wrist"] = np.ones((3, 224, 224), dtype=np.float32)
    result = xtac_umi_policy.XTacUmiInputs()(example)

    assert result["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert result["image"]["left_wrist_0_rgb"].max() == 255


def test_inputs_pass_rotations_through_untouched():
    """6D rotation is already continuous, so nothing may rewrite the state."""
    example = _example(with_actions=True)
    example["state"] = np.arange(xtac_umi_policy.STATE_DIM, dtype=np.float32)
    result = xtac_umi_policy.XTacUmiInputs()(example)

    np.testing.assert_array_equal(result["state"], example["state"])
    np.testing.assert_array_equal(result["actions"], example["actions"])


def test_outputs_trim_padded_action_dim():
    padded = np.random.rand(10, 32)
    result = xtac_umi_policy.XTacUmiOutputs()({"actions": padded})

    assert result["actions"].shape == (10, xtac_umi_policy.STATE_DIM)
    np.testing.assert_array_equal(result["actions"], padded[:, : xtac_umi_policy.STATE_DIM])


def test_delta_mask_matches_the_per_side_grouped_layout():
    """TCP dims delta-encoded, both gripper dims absolute — at 9 and 19, not 18/19."""
    mask = _transforms.make_bool_mask(9, -1, 9, -1)

    assert len(mask) == xtac_umi_policy.STATE_DIM
    absolute = [index for index, is_delta in enumerate(mask) if not is_delta]
    assert absolute == [9, 19]
    assert all(mask[xtac_umi_policy.LEFT_TCP])
    assert all(mask[xtac_umi_policy.RIGHT_TCP])
    assert not any(mask[xtac_umi_policy.LEFT_GRIPPER])
    assert not any(mask[xtac_umi_policy.RIGHT_GRIPPER])


def test_data_config_wires_the_xtac_umi_transforms():
    """The shipped example config must resolve to these transforms, not BiFlexiv's."""
    train_config = _config.get_config("pi05_base_xtac_umi_sort_defective_parts_0710")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    input_types = [type(t) for t in data_config.data_transforms.inputs]
    output_types = [type(t) for t in data_config.data_transforms.outputs]
    assert xtac_umi_policy.XTacUmiInputs in input_types
    assert xtac_umi_policy.XTacUmiOutputs in output_types
    # use_delta_cartesian_actions defaults on, so the delta pair must be wired too.
    assert _transforms.DeltaActions in input_types
    assert _transforms.AbsoluteActions in output_types


def test_data_config_repack_tracks_use_head_camera():
    """The head column is requested only when the flag asks for it."""
    train_config = _config.get_config("pi05_base_xtac_umi_sort_defective_parts_0710")

    def repack_images(*, use_head_camera: bool) -> dict:
        data = _config.LeRobotXTacUmiDataConfig(repo_id=train_config.data.repo_id, use_head_camera=use_head_camera)
        config = data.create(train_config.assets_dirs, train_config.model)
        return config.repack_transforms.inputs[0].structure["images"]

    assert set(repack_images(use_head_camera=False)) == {"left_wrist", "right_wrist"}
    assert set(repack_images(use_head_camera=True)) == {"head", "left_wrist", "right_wrist"}
