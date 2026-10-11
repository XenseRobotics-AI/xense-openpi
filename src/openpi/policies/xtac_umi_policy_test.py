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
    train_config = _config.get_config("pi05_base_xtac_umi_pick_up_cube_0807_h200")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    input_types = [type(t) for t in data_config.data_transforms.inputs]
    output_types = [type(t) for t in data_config.data_transforms.outputs]
    assert xtac_umi_policy.XtacUmiInputs in input_types
    assert xtac_umi_policy.XtacUmiOutputs in output_types
    # use_delta_cartesian_actions defaults on, so the delta pair must be wired too.
    assert _transforms.DeltaActions in input_types
    assert _transforms.AbsoluteActions in output_types


def test_data_config_repack_tracks_use_head_camera():
    """The head column is requested only when the flag asks for it."""
    train_config = _config.get_config("pi05_base_xtac_umi_pick_up_cube_0807_h200")

    def repack_images(*, use_head_camera: bool) -> dict:
        data = _config.LeRobotXtacUmiDataConfig(repo_id=train_config.data.repo_id, use_head_camera=use_head_camera)
        config = data.create(train_config.assets_dirs, train_config.model)
        return config.repack_transforms.inputs[0].structure["images"]

    assert set(repack_images(use_head_camera=False)) == {"left_wrist", "right_wrist"}
    assert set(repack_images(use_head_camera=True)) == {"head", "left_wrist", "right_wrist"}


def test_g1_hand_only_pipeline_round_trip_and_stats_match(tmp_path, monkeypatch):
    """29D targets must be sliced before deltas, including in parquet normalization."""
    from scripts import compute_norm_stats

    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model: _transforms.Group())
    factory = _config.LeRobotXtacUmiDataConfig(
        drop_head_pose=True,
        use_head_camera=True,
        head_camera_key="observation.images.head_left",
    )
    config = factory.create(tmp_path, _config.pi0_config.Pi0Config(pi05=True))
    assert config.repack_transforms.inputs[0].structure["images"]["head"] == "observation.images.head_left"
    state = np.arange(29, dtype=np.float32)
    actions = np.stack([state + 1, state + 2])
    sample = _example(with_head=True)
    sample.update(state=state, actions=actions)
    trained = _transforms.compose(config.data_transforms.inputs)(sample)
    assert trained["state"].shape == (20,)
    assert trained["actions"].shape == (2, 20)
    np.testing.assert_array_equal(trained["actions"][:, [9, 19]], actions[:, [9, 19]])
    stats_batch = _transforms.compose(compute_norm_stats._state_action_transforms(config.data_transforms.inputs))(
        {"state": state[None], "actions": actions[None]}
    )
    np.testing.assert_array_equal(stats_batch["state"][0], trained["state"])
    np.testing.assert_array_equal(stats_batch["actions"][0], trained["actions"])
    restored = _transforms.compose(config.data_transforms.outputs)(trained)
    np.testing.assert_allclose(restored["actions"], actions[:, :20])
    # Delta transforms must not mutate the caller's unsliced recording.
    np.testing.assert_array_equal(sample["actions"], actions)


@pytest.mark.parametrize(("dims", "drop_head_pose"), [(29, False), (28, True)])
def test_state_actions_reject_unexpected_recording_layout(dims, drop_head_pose):
    with pytest.raises(ValueError, match="XTac-UMI state must have"):
        xtac_umi_policy.XtacUmiStateActions(drop_head_pose=drop_head_pose)({"state": np.zeros(dims)})


def test_g1_hand_only_config_accepts_20d_state_at_inference():
    sample = {"state": np.arange(20, dtype=np.float32)}
    result = xtac_umi_policy.XtacUmiStateActions(drop_head_pose=True)(sample)
    np.testing.assert_array_equal(result["state"], sample["state"])


def _random_pose9(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    rotation = np.linalg.qr(rng.normal(size=(*shape, 3, 3)))[0]
    return np.concatenate((rng.normal(size=(*shape, 3)), rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def _random_hands(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    grippers = rng.uniform(size=(*shape, 2))
    return np.concatenate(
        (_random_pose9(rng, shape), grippers[..., :1], _random_pose9(rng, shape), grippers[..., 1:]), axis=-1
    )


def _transform_hands(world_from_old: np.ndarray, hands: np.ndarray) -> np.ndarray:
    """Re-express both TCP poses in another world frame: T -> W @ T."""
    out = hands.copy()
    for tcp in (xtac_umi_policy.LEFT_TCP, xtac_umi_policy.RIGHT_TCP):
        pose = hands[..., tcp]
        rotation = world_from_old[:3, :3]
        out[..., tcp] = np.concatenate(
            (
                pose[..., :3] @ rotation.T + world_from_old[:3, 3],
                pose[..., 3:6] @ rotation.T,
                pose[..., 6:9] @ rotation.T,
            ),
            axis=-1,
        )
    return out


def test_relative_actions_examples():
    """Pushing 3 cm along the fingertips and yawing 90 deg left read the same in any frame."""
    # Fingertips (gripper x) along world +y, gripper y along world -x.
    state = np.zeros(20)
    state[xtac_umi_policy.LEFT_TCP] = [0.5, 0.2, 0.3, 0, 1, 0, -1, 0, 0]
    state[xtac_umi_policy.RIGHT_TCP] = [0.5, -0.2, 0.3, 0, 1, 0, -1, 0, 0]
    actions = np.repeat(state[None], 2, axis=0)
    actions[0, 1] += 0.03  # Left: 3 cm along world +y, i.e. along its own fingertips.
    actions[1, 13:19] = [-1, 0, 0, 0, -1, 0]  # Right: fingertips now along world -x.

    relative = xtac_umi_policy.XtacUmiRelativeActions()({"state": state, "actions": actions})["actions"]

    np.testing.assert_allclose(relative[0, xtac_umi_policy.LEFT_TCP], [0.03, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(relative[1, xtac_umi_policy.RIGHT_TCP], [0, 0, 0, 0, 1, 0, -1, 0, 0], atol=1e-6)


def test_relative_actions_are_world_frame_invariant_and_state_is_grippers():
    rng = np.random.default_rng(0)
    state = _random_hands(rng, ())
    actions = _random_hands(rng, (5,))
    world_from_old = np.eye(4)
    world_from_old[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0] * np.array([1, 1, -1])
    if np.linalg.det(world_from_old[:3, :3]) < 0:
        world_from_old[:3, 0] *= -1
    world_from_old[:3, 3] = rng.normal(size=3)

    transform = xtac_umi_policy.XtacUmiRelativeActions()
    original = transform({"state": state, "actions": actions})
    moved = transform(
        {"state": _transform_hands(world_from_old, state), "actions": _transform_hands(world_from_old, actions)}
    )

    np.testing.assert_allclose(original["actions"], moved["actions"], atol=1e-6)
    np.testing.assert_array_equal(original["state"], state[[9, 19]])
    assert _transforms.INFERENCE_CONTEXT_KEY not in original


def test_relative_actions_round_trip_through_inference_context():
    rng = np.random.default_rng(1)
    state = _random_hands(rng, ())
    actions = _random_hands(rng, (4,))
    relative = xtac_umi_policy.XtacUmiRelativeActions()({"state": state, "actions": actions})["actions"]

    inference = xtac_umi_policy.XtacUmiRelativeActions()({"state": state})
    assert inference["state"].shape == (2,)
    # The model emits padded actions; padding must survive untouched.
    padded = np.concatenate((relative, np.zeros((4, 12))), axis=-1)
    outputs = xtac_umi_policy.XtacUmiAbsoluteActions()(
        {"actions": padded, _transforms.INFERENCE_CONTEXT_KEY: inference[_transforms.INFERENCE_CONTEXT_KEY]}
    )

    np.testing.assert_allclose(outputs["actions"][:, :20], actions, atol=1e-6)
    assert _transforms.INFERENCE_CONTEXT_KEY not in outputs


def test_relative_config_pipeline_and_norm_stats_match(tmp_path, monkeypatch):
    from scripts import compute_norm_stats

    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model: _transforms.Group())
    factory = _config.LeRobotXtacUmiDataConfig(
        drop_head_pose=True, use_delta_cartesian_actions=False, use_relative_pose_actions=True
    )
    config = factory.create(tmp_path, _config.pi0_config.Pi0Config(pi05=True))
    rng = np.random.default_rng(2)
    head = rng.normal(size=9)
    state = np.concatenate((_random_hands(rng, ()), head))
    actions = np.concatenate((_random_hands(rng, (3,)), np.tile(head, (3, 1))), axis=-1)
    sample = _example()
    sample.update(state=state, actions=actions)

    trained = _transforms.compose(config.data_transforms.inputs)(sample)
    stats_batch = _transforms.compose(compute_norm_stats._state_action_transforms(config.data_transforms.inputs))(
        {"state": state[None], "actions": actions[None]}
    )
    np.testing.assert_allclose(stats_batch["state"][0], trained["state"])
    np.testing.assert_allclose(stats_batch["actions"][0], trained["actions"])
    assert trained["state"].shape == (2,)

    inference = _transforms.compose(config.data_transforms.inputs)({**_example(), "state": state})
    restored = _transforms.compose(config.data_transforms.outputs)(
        {"actions": trained["actions"], _transforms.INFERENCE_CONTEXT_KEY: inference[_transforms.INFERENCE_CONTEXT_KEY]}
    )
    np.testing.assert_allclose(restored["actions"], actions[:, :20], atol=1e-6)


def test_relative_and_delta_actions_are_exclusive(tmp_path):
    factory = _config.LeRobotXtacUmiDataConfig(use_relative_pose_actions=True)
    with pytest.raises(ValueError, match="use_relative_pose_actions replaces"):
        factory.create(tmp_path, _config.pi0_config.Pi0Config(pi05=True))
