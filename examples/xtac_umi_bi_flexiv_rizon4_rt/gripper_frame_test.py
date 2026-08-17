"""Tests for the gripper end-frame change of basis.

The rotation is applied to every action the arms execute, so its properties are
worth pinning: get the handedness or the multiplication side wrong and the arms
still move, just to the wrong orientation.
"""

import numpy as np
import pytest

from examples.xtac_umi_bi_flexiv_rizon4_rt import gripper_frame

_IDENTITY_POSE9 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _random_pose9(rng: np.random.Generator) -> np.ndarray:
    """A valid pose: random translation plus the first two columns of a rotation."""
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:  # QR can return a reflection
        rotation[:, 0] *= -1
    return np.concatenate((rng.standard_normal(3), rotation[:, 0], rotation[:, 1]))


def _random_state(rng: np.random.Generator) -> np.ndarray:
    return np.concatenate((_random_pose9(rng), [0.3], _random_pose9(rng), [0.7]))


def test_change_of_basis_is_a_proper_rotation_and_self_inverse():
    matrix = gripper_frame._FLEXIV_GRIPPER_FROM_UMI_GRIPPER
    rotation = matrix[:3, :3]

    # A reflection here would mirror every commanded orientation.
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    # Self-inverse is what lets one function serve both directions.
    np.testing.assert_allclose(matrix @ matrix, np.eye(4), atol=1e-12)
    # No translation: both conventions name the same physical TCP point.
    np.testing.assert_array_equal(matrix[:3, 3], np.zeros(3))


def test_documented_axis_mapping():
    """UMI x = flexiv +z, UMI y = flexiv -y, UMI z = flexiv +x."""
    columns = gripper_frame._FLEXIV_GRIPPER_FROM_UMI_GRIPPER[:3, :3]

    np.testing.assert_array_equal(columns[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(columns[:, 1], [0, -1, 0])
    np.testing.assert_array_equal(columns[:, 2], [1, 0, 0])


def test_align_gripper_frames_round_trips():
    rng = np.random.default_rng(0)
    state = _random_state(rng)

    round_tripped = gripper_frame.align_gripper_frames(gripper_frame.align_gripper_frames(state))
    np.testing.assert_allclose(round_tripped, state, atol=1e-6)


def test_align_gripper_frames_leaves_grippers_and_translation_alone():
    rng = np.random.default_rng(1)
    state = _random_state(rng)
    converted = gripper_frame.align_gripper_frames(state)

    # Gripper dims are at 9 and 19 in the per-side-grouped layout.
    assert converted[9] == pytest.approx(state[9])
    assert converted[19] == pytest.approx(state[19])
    # Same physical point, so only the orientation columns move.
    np.testing.assert_allclose(converted[0:3], state[0:3], atol=1e-6)
    np.testing.assert_allclose(converted[10:13], state[10:13], atol=1e-6)
    assert not np.allclose(converted[3:9], state[3:9])


def test_align_gripper_frames_is_batched_and_per_side_independent():
    rng = np.random.default_rng(2)
    batch = np.stack([_random_state(rng) for _ in range(4)])

    converted = gripper_frame.align_gripper_frames(batch)
    assert converted.shape == batch.shape
    for index in range(len(batch)):
        np.testing.assert_allclose(converted[index], gripper_frame.align_gripper_frames(batch[index]), atol=1e-6)

    # Changing the left arm must not move the right one.
    perturbed = batch.copy()
    perturbed[:, 0:9] = _random_pose9(rng)
    np.testing.assert_allclose(gripper_frame.align_gripper_frames(perturbed)[:, 10:20], converted[:, 10:20], atol=1e-6)


def test_pose9_matrix_round_trip_and_reorthonormalization():
    rng = np.random.default_rng(3)
    pose = _random_pose9(rng)

    np.testing.assert_allclose(gripper_frame.matrix_to_pose9(gripper_frame.pose9_to_matrix(pose)), pose, atol=1e-6)

    # A slightly non-orthonormal network output must still yield a valid rotation.
    sloppy = pose.copy()
    sloppy[3:6] *= 1.02
    sloppy[6:9] += 0.01 * sloppy[3:6]
    rotation = gripper_frame.pose9_to_matrix(sloppy)[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_identity_pose_maps_to_the_relabelled_axes():
    """Sanity: the identity orientation picks up the basis, it is not a fixed point."""
    state = np.concatenate((_IDENTITY_POSE9, [0.0], _IDENTITY_POSE9, [1.0]))
    converted = gripper_frame.align_gripper_frames(state)

    # Identity rotation @ M = M, so the first two columns become M's.
    np.testing.assert_allclose(converted[3:6], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(converted[6:9], [0, -1, 0], atol=1e-6)


def test_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="Expected last dimension 20"):
        gripper_frame.align_gripper_frames(np.zeros(18))
    with pytest.raises(ValueError, match="Expected pose last dimension 9"):
        gripper_frame.pose9_to_matrix(np.zeros(7))
    with pytest.raises(ValueError, match="first column has zero norm"):
        gripper_frame.pose9_to_matrix(np.zeros(9))
    with pytest.raises(ValueError, match="linearly dependent"):
        gripper_frame.pose9_to_matrix(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]))
