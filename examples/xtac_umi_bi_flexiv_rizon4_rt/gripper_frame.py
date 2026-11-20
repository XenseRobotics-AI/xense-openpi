"""9D TCP pose helpers shared by the XTac-UMI re-basing, replay and deployment code.

A pose is ``[x, y, z, r1..r6]``, where r1-r3 / r4-r6 are the first two COLUMNS of
the rotation matrix — the same convention as the recording rig, the Flexiv driver
and ``openpi.policies.xtac_umi_policy``.

Nothing here converts between gripper end-frame conventions. Training data is
re-based onto the arms' start TCPs (``rebase_umi_dataset.py``), which composes the
recorded world-frame motion onto the robot's own TCP orientation, so datasets,
checkpoints and the Flexiv driver all share the Flexiv TCP convention.
"""

import numpy as np


def pose9_to_matrix(pose: np.ndarray) -> np.ndarray:
    """``[x, y, z, r1..r6]`` -> 4x4 homogeneous matrix (batched).

    The 6D rotation is re-orthonormalized with Gram-Schmidt (normalize the first
    column, orthogonalize the second against it, third = cross product), so
    slightly non-orthonormal network outputs still yield a valid rotation.
    """
    pose = np.asarray(pose)
    if pose.ndim == 0 or pose.shape[-1] != 9:
        raise ValueError(f"Expected pose last dimension 9, got {pose.shape}")

    first = pose[..., 3:6]
    second = pose[..., 6:9]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Invalid 6D rotation: first column has zero norm")
    first = first / first_norm

    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Invalid 6D rotation: columns are linearly dependent")
    second = second / second_norm
    third = np.cross(first, second)

    matrix = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.result_type(pose.dtype, np.float32))
    matrix[..., :3, :3] = np.stack((first, second, third), axis=-1)
    matrix[..., :3, 3] = pose[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    """4x4 homogeneous matrix -> ``[x, y, z, r1..r6]`` (batched)."""
    return np.concatenate((matrix[..., :3, 3], matrix[..., :3, 0], matrix[..., :3, 1]), axis=-1)
