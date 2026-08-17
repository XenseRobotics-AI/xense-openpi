"""Gripper end-frame change-of-basis between the Flexiv driver and XTac-UMI data.

Dim layout is NOT converted here. Both sides of this boundary use the XTac-UMI
per-side-grouped 20D layout::

    [left_tcp(0-8), left_gripper(9), right_tcp(10-18), right_gripper(19)]

``real_env.py`` assembles the robot's state in exactly that order from the
driver's named keys, so nothing in this example regroups dims. Each 9D TCP pose
is ``[x, y, z, r1..r6]`` where r1-r3 / r4-r6 are the first two COLUMNS of the
rotation matrix — the same convention as the recording rig, the Flexiv driver,
and ``openpi.policies.xtac_umi_policy``.

What DOES differ is the axis convention of the gripper's own end frame for the
same physical TCP point:

- XTac-UMI (policy/dataset side): x along the fingertips (forward), y left, z up.
- Flexiv driver (robot side): z along the fingertips (forward), y right, x up.

So UMI x = flexiv +z, UMI y = flexiv -y, UMI z = flexiv +x. :func:`align_gripper_frames`
right-multiplies each TCP orientation by that change of basis; translation is
untouched (both conventions name the same physical point).

**This is a hardware/CAD claim, not something this repo can verify.** It comes
from the UMI rig's ``ee_transform.py`` (EE frame at the two-finger midpoint, x
forward) versus the Flexiv flange->TCP setup. If a bench's driver already reports
UMI-convention poses, disable it with ``--args.no-align-gripper-frames``.

Nothing here is stateful: the change of basis is a constant, and it is its own
inverse, so the same function converts in both directions.
"""

import numpy as np

# Columns are the XTac-UMI gripper axes expressed in the Flexiv gripper frame:
# UMI x = flexiv +z, UMI y = flexiv -y, UMI z = flexiv +x. det = +1, and the
# matrix is symmetric and therefore self-inverse (M @ M = I), so one function
# serves both conversion directions. Translation is zero — same TCP point.
_FLEXIV_GRIPPER_FROM_UMI_GRIPPER = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


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


def _reexpress_pose9(pose: np.ndarray, offset_in_pose_frame: np.ndarray) -> np.ndarray:
    """Right-multiply each pose by a constant offset expressed in the pose's own frame.

    Change of basis of the TCP end frame: same physical point, relabelled axes —
    i.e. ``pose @ offset``, NOT ``offset @ pose``.
    """
    reexpressed = np.matmul(pose9_to_matrix(pose), offset_in_pose_frame)
    return matrix_to_pose9(reexpressed).astype(np.result_type(np.asarray(pose).dtype, np.float32), copy=False)


def align_gripper_frames(vector: np.ndarray) -> np.ndarray:
    """Convert a 20D state/action between the Flexiv and XTac-UMI gripper frames.

    Self-inverse: the same call converts flexiv->UMI and UMI->flexiv. Dim layout
    is unchanged (per-side-grouped in, per-side-grouped out) and both gripper
    dims pass through untouched.
    """
    vector = np.asarray(vector)
    if vector.ndim == 0 or vector.shape[-1] != 20:
        raise ValueError(f"Expected last dimension 20, got {vector.shape}")
    return np.concatenate(
        (
            _reexpress_pose9(vector[..., 0:9], _FLEXIV_GRIPPER_FROM_UMI_GRIPPER),
            vector[..., 9:10],
            _reexpress_pose9(vector[..., 10:19], _FLEXIV_GRIPPER_FROM_UMI_GRIPPER),
            vector[..., 19:20],
        ),
        axis=-1,
    )
