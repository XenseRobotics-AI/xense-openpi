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

# The shipped absolute-state config: trained on data re-based onto the arms' start TCPs.
_REBASED_EXAMPLE_CONFIG = "pi05_base_xtac_umi_g1_clean_plates_fwd04_rebased_1014_h100"


def _example(*, with_head: bool = False, with_actions: bool = False) -> dict:
    example = xtac_umi_policy.make_xtac_umi_example()
    if with_head:
        example["images"]["head"] = np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)
    if with_actions:
        example["actions"] = np.zeros((10, xtac_umi_policy.STATE_DIM))
    return example


def test_inputs_mask_base_slot_by_default():
    """No head camera: base_0_rgb is a black image the model is told to ignore."""
    result = xtac_umi_policy.XtacUmiInputs()(_example())

    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image_mask"]["base_0_rgb"] == np.False_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
    assert result["image_mask"]["right_wrist_0_rgb"] == np.True_
    # Black, and the same shape as a real frame so the model sees a consistent slot.
    assert not result["image"]["base_0_rgb"].any()
    assert result["image"]["base_0_rgb"].shape == result["image"]["left_wrist_0_rgb"].shape


def test_inputs_use_head_camera_fills_and_unmasks_base_slot():
    result = xtac_umi_policy.XtacUmiInputs(use_head_camera=True)(_example(with_head=True))

    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image"]["base_0_rgb"].any()


def test_inputs_head_camera_ignored_unless_enabled():
    """A client may send `head` for compatibility; it must not leak into the slot."""
    result = xtac_umi_policy.XtacUmiInputs()(_example(with_head=True))

    assert result["image_mask"]["base_0_rgb"] == np.False_
    assert not result["image"]["base_0_rgb"].any()


def test_inputs_reject_missing_wrist_and_missing_head():
    example = _example()
    del example["images"]["left_wrist"]
    with pytest.raises(ValueError, match="Missing required wrist cameras"):
        xtac_umi_policy.XtacUmiInputs()(example)

    with pytest.raises(ValueError, match="use_head_camera=True but no 'head' image"):
        xtac_umi_policy.XtacUmiInputs(use_head_camera=True)(_example())

    example = _example()
    example["images"]["belly"] = np.zeros((3, 224, 224), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unexpected cameras"):
        xtac_umi_policy.XtacUmiInputs()(example)


def test_inputs_convert_images_to_hwc_uint8():
    example = _example()
    example["images"]["left_wrist"] = np.ones((3, 224, 224), dtype=np.float32)
    result = xtac_umi_policy.XtacUmiInputs()(example)

    assert result["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert result["image"]["left_wrist_0_rgb"].max() == 255


def test_inputs_pass_rotations_through_untouched():
    """6D rotation is already continuous, so nothing may rewrite the state."""
    example = _example(with_actions=True)
    example["state"] = np.arange(xtac_umi_policy.STATE_DIM, dtype=np.float32)
    result = xtac_umi_policy.XtacUmiInputs()(example)

    np.testing.assert_array_equal(result["state"], example["state"])
    np.testing.assert_array_equal(result["actions"], example["actions"])


def test_outputs_trim_padded_action_dim():
    padded = np.random.rand(10, 32)
    result = xtac_umi_policy.XtacUmiOutputs()({"actions": padded})

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
    train_config = _config.get_config(_REBASED_EXAMPLE_CONFIG)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    input_types = [type(t) for t in data_config.data_transforms.inputs]
    output_types = [type(t) for t in data_config.data_transforms.outputs]
    # The 20D layout check runs first, before anything can train on a raw recording.
    assert input_types[0] is xtac_umi_policy.XtacUmiStateActions
    assert xtac_umi_policy.XtacUmiInputs in input_types
    assert xtac_umi_policy.XtacUmiOutputs in output_types
    # The re-based config trains the absolute TCP state, so the delta pair must be wired too.
    assert _transforms.DeltaActions in input_types
    assert _transforms.AbsoluteActions in output_types


def test_data_config_repack_tracks_use_head_camera():
    """The head column is requested only when the flag asks for it."""
    train_config = _config.get_config(_REBASED_EXAMPLE_CONFIG)

    def repack_images(*, use_head_camera: bool) -> dict:
        data = _config.LeRobotXtacUmiDataConfig(repo_id=train_config.data.repo_id, use_head_camera=use_head_camera)
        config = data.create(train_config.assets_dirs, train_config.model)
        return config.repack_transforms.inputs[0].structure["images"]

    assert set(repack_images(use_head_camera=False)) == {"left_wrist", "right_wrist"}
    assert set(repack_images(use_head_camera=True)) == {"head", "left_wrist", "right_wrist"}


def test_rebased_pipeline_round_trip_and_stats_match(tmp_path, monkeypatch):
    """Re-based 20D data: training deltas, parquet normalization and the output inverse agree."""
    from scripts import compute_norm_stats

    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model: _transforms.Group())
    factory = _config.LeRobotXtacUmiDataConfig(
        use_head_camera=True,
        head_camera_key="observation.images.head_left",
    )
    config = factory.create(tmp_path, _config.pi0_config.Pi0Config(pi05=True))
    assert config.repack_transforms.inputs[0].structure["images"]["head"] == "observation.images.head_left"
    state = np.arange(20, dtype=np.float32)
    actions = np.stack([state + 1, state + 2])
    sample = _example(with_head=True)
    sample.update(state=state, actions=actions)
    trained = _transforms.compose(config.data_transforms.inputs)(sample)
    assert trained["state"].shape == (20,)
    assert trained["actions"].shape == (2, 20)
    # TCP dims are deltas against the current state; the two gripper dims stay absolute.
    np.testing.assert_array_equal(trained["actions"][0, xtac_umi_policy.LEFT_TCP], np.ones(9))
    np.testing.assert_array_equal(trained["actions"][1, xtac_umi_policy.RIGHT_TCP], np.full(9, 2.0))
    np.testing.assert_array_equal(trained["actions"][:, [9, 19]], actions[:, [9, 19]])
    stats_batch = _transforms.compose(compute_norm_stats._state_action_transforms(config.data_transforms.inputs))(
        {"state": state[None], "actions": actions[None]}
    )
    np.testing.assert_array_equal(stats_batch["state"][0], trained["state"])
    np.testing.assert_array_equal(stats_batch["actions"][0], trained["actions"])
    restored = _transforms.compose(config.data_transforms.outputs)(trained)
    np.testing.assert_allclose(restored["actions"], actions)
    # Delta transforms must not mutate the caller's recording.
    np.testing.assert_array_equal(sample["actions"], actions)


def test_raw_29d_recording_is_rejected_with_a_rebase_hint():
    """A raw G1 recording's poses are not in the robot's frame; it must be re-based first."""
    state_actions = xtac_umi_policy.XtacUmiStateActions()

    with pytest.raises(ValueError, match="rebase_umi_dataset"):
        state_actions({"state": np.zeros(29)})
    with pytest.raises(ValueError, match="rebase_umi_dataset"):
        state_actions({"state": np.zeros(20), "actions": np.zeros((4, 29))})


@pytest.mark.parametrize("dims", [19, 21, 28])
def test_state_actions_reject_unexpected_recording_layout(dims):
    with pytest.raises(ValueError, match="must be 20D"):
        xtac_umi_policy.XtacUmiStateActions()({"state": np.zeros(dims)})


def test_state_actions_pass_20d_through_as_a_copy():
    sample = {"state": np.arange(20, dtype=np.float32), "actions": np.ones((3, 20), dtype=np.float32)}
    result = xtac_umi_policy.XtacUmiStateActions()(sample)

    np.testing.assert_array_equal(result["state"], sample["state"])
    np.testing.assert_array_equal(result["actions"], sample["actions"])
    # Later delta transforms write in place; the caller's arrays must not be touched.
    assert result["state"] is not sample["state"]
    assert result["actions"] is not sample["actions"]
