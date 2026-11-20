"""Tests for the 9D TCP pose helpers.

Re-basing and replay build every arm target through these, so a wrong column order
or a reflection would still move the arms, just to the wrong orientation.
"""

import numpy as np
import pytest

from examples.xtac_umi_bi_flexiv_rizon4_rt import gripper_frame


def _random_pose9(rng: np.random.Generator) -> np.ndarray:
    """A valid pose: random translation plus the first two columns of a rotation."""
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:  # QR can return a reflection
        rotation[:, 0] *= -1
    return np.concatenate((rng.standard_normal(3), rotation[:, 0], rotation[:, 1]))


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


def test_pose9_columns_and_translation_land_where_documented():
    """r1-r3 / r4-r6 are the first two rotation COLUMNS; the third is their cross product."""
    pose = np.array([0.1, 0.2, 0.3, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])  # 90 deg about z
    matrix = gripper_frame.pose9_to_matrix(pose)

    np.testing.assert_allclose(matrix[:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(matrix[:3, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(matrix[:3, 1], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(matrix[:3, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])


def test_pose9_helpers_are_batched():
    rng = np.random.default_rng(4)
    batch = np.stack([[_random_pose9(rng) for _ in range(3)] for _ in range(2)])

    matrices = gripper_frame.pose9_to_matrix(batch)
    assert matrices.shape == (2, 3, 4, 4)
    np.testing.assert_allclose(matrices[1, 2], gripper_frame.pose9_to_matrix(batch[1, 2]), atol=1e-6)
    np.testing.assert_allclose(gripper_frame.matrix_to_pose9(matrices), batch, atol=1e-6)


def test_rejects_invalid_poses():
    with pytest.raises(ValueError, match="Expected pose last dimension 9"):
        gripper_frame.pose9_to_matrix(np.zeros(7))
    with pytest.raises(ValueError, match="first column has zero norm"):
        gripper_frame.pose9_to_matrix(np.zeros(9))
    with pytest.raises(ValueError, match="linearly dependent"):
        gripper_frame.pose9_to_matrix(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]))
