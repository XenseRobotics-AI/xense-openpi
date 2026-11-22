"""Policy transforms for XTac-UMI bimanual datasets.

XTac-UMI data is collected with the handheld ``bi_taccap_gripper`` rig (see the
``xense-taccap-lerobot`` repo): two TacCap grippers, each tracked by a Pico4
tracker, with a wrist camera and tactile sensors per side. There is no arm, so
there is no third-person view and no robot base frame.

State/action layout (20D) follows the recording rig's feature order verbatim
(``BiTaccapGripper.observation_features``, which emits per side in turn)::

    [ left_tcp.x, left_tcp.y, left_tcp.z, left_tcp.r1..r6,      # dims 0-8
      left_gripper.pos,                                         # dim  9
      right_tcp.x, right_tcp.y, right_tcp.z, right_tcp.r1..r6,  # dims 10-18
      right_gripper.pos ]                                       # dim  19

Deploying onto a bimanual Flexiv reads the same 20D vector in the same order —
``examples/xtac_umi_bi_flexiv_rizon4_rt`` assembles it per-side-grouped from the
driver's named keys, so no dim regrouping happens anywhere in this repo.

The 6D rotation (r1-r6) is the first two COLUMNS of the rotation matrix ("On the
Continuity of Rotation Representations in Neural Networks"), matching both the
recording rig and the Flexiv driver. It is already continuous, so these
transforms pass rotations through untouched.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

# Per-side-grouped dim slices, shared with the data config's delta-action mask.
LEFT_TCP = slice(0, 9)
LEFT_GRIPPER = slice(9, 10)
RIGHT_TCP = slice(10, 19)
RIGHT_GRIPPER = slice(19, 20)
# xtac_umi_g1 recordings append the headset pose after the two hands.
HEAD_POSE = slice(20, 29)

STATE_DIM = 20
G1_STATE_DIM = STATE_DIM + 9


@dataclasses.dataclass(frozen=True)
class XtacUmiStateActions(transforms.DataTransformFn):
    """Select the two hands from a 29D G1 recording before delta encoding and normalization.

    G1 recordings append a 9D headset pose to the original 20D hand layout.
    ``use_head_pose=False`` drops it, accepting either layout so that a hand-only
    policy still serves on the robot's 20D state. ``use_head_pose=True`` keeps all
    29D and requires them.
    Works on individual samples and on batches used to compute normalization stats.
    """

    use_head_pose: bool = False

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        # Serving a hand-only policy naturally supplies 20D state, so 29D input is
        # accepted there and sliced rather than refused.
        expected_dims = (G1_STATE_DIM,) if self.use_head_pose else (STATE_DIM, G1_STATE_DIM)
        keep = G1_STATE_DIM if self.use_head_pose else STATE_DIM
        for key in ("state", "actions"):
            if key not in data:
                continue
            value = np.asarray(data[key])
            if value.ndim == 0 or value.shape[-1] not in expected_dims:
                raise ValueError(
                    f"XTac-UMI {key} must have dimensions in {expected_dims} with "
                    f"use_head_pose={self.use_head_pose}; got {value.shape}. For 29D xtac_umi_g1 "
                    "recordings, set use_head_pose=False to train the hands only, or True to keep "
                    "the headset pose."
                )
            result[key] = value[..., :keep].copy()
        return result


def _pose_slices(dim: int) -> tuple[slice, ...]:
    """The 9D pose blocks of a per-side-grouped vector: both hands, plus the head when 29D."""
    if dim == STATE_DIM:
        return (LEFT_TCP, RIGHT_TCP)
    if dim == G1_STATE_DIM:
        return (LEFT_TCP, RIGHT_TCP, HEAD_POSE)
    raise ValueError(f"Expected a {STATE_DIM}D or {G1_STATE_DIM}D XTac-UMI vector, got {dim}D")


@dataclasses.dataclass(frozen=True)
class XtacUmiFrameRelativePoses(transforms.DataTransformFn):
    """Every pose as its motion since the previous frame: ``inv(T_{t-1}) @ T_t``.

    Recordings are absolute poses in the Pico world frame (fixed when recording starts:
    z up against gravity, x forward, y left from the headset). That frame has nothing to
    do with a robot base, so the model only sees per-frame SE(3) motion, expressed in the
    moving frame itself: left TCP, right TCP and, in 29D G1 data, the head. Gripper
    openings stay absolute in both state and actions.

    - ``state`` must be ``[..., 2, D]``: the previous and the current absolute frame
      (``DataConfig.previous_state_keys`` makes the data loader supply it). The model's
      state is the current frame relative to the previous one. At an episode's first
      frame the loader clamps "previous" to the frame itself, so the motion is identity.
    - ``actions`` ``[..., H, D]`` are absolute next-frame poses (``action[t]`` is the pose
      at ``t + 1``), so each becomes relative to the one before it, the first to the
      current state: ``inv(T_t) @ A_0``, ``inv(A_0) @ A_1``, ...
    - Without actions (inference), the current absolute state is stashed under
      ``transforms.INFERENCE_CONTEXT_KEY`` for `XtacUmiFrameAbsolutePoses`.

    Works on individual samples and on batches used to compute normalization stats.
    """

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.ndim < 2 or state.shape[-2] != 2:
            raise ValueError(
                f"Frame-relative poses need the previous and current state, [..., 2, D]; got {state.shape}. "
                "Set previous_state_keys on the data config (training) or send both frames (inference)."
            )
        poses = _pose_slices(state.shape[-1])
        previous, current = state[..., 0, :], state[..., 1, :]

        result = dict(data)
        relative_state = np.array(current, copy=True)
        for pose in poses:
            relative_state[..., pose] = _relative_pose9(previous[..., pose], current[..., pose])
        result["state"] = relative_state

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != state.shape[-1]:
                raise ValueError(f"actions are {actions.shape[-1]}D but state is {state.shape[-1]}D")
            # The frame before each action: the current state, then the action before it.
            before = np.concatenate((current[..., None, :], actions[..., :-1, :]), axis=-2)
            relative_actions = np.array(actions, copy=True)
            for pose in poses:
                relative_actions[..., pose] = _relative_pose9(before[..., pose], actions[..., pose])
            result["actions"] = relative_actions
        else:
            result[transforms.INFERENCE_CONTEXT_KEY] = {"absolute_state": current}
        return result


@dataclasses.dataclass(frozen=True)
class XtacUmiFrameAbsolutePoses(transforms.DataTransformFn):
    """Chain `XtacUmiFrameRelativePoses` actions back onto the current pose.

    ``A_0 = T_t @ dT_0``, ``A_k = A_{k-1} @ dT_k``. Gripper dims pass through. Returns
    the action dims of the stashed state (20 or 29), dropping the model's padding.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        result = dict(data)
        context = result.pop(transforms.INFERENCE_CONTEXT_KEY, None)
        if context is None:
            raise ValueError("Frame-relative actions need the absolute state stashed by XtacUmiFrameRelativePoses")
        current = np.asarray(context["absolute_state"])
        dim = current.shape[-1]
        poses = _pose_slices(dim)
        relative = np.asarray(data["actions"])[..., :dim]
        actions = np.array(relative, copy=True)
        pose_before = current
        for k in range(relative.shape[-2]):
            for pose in poses:
                actions[..., k, pose] = _compose_pose9(pose_before[..., pose], relative[..., k, pose])
            pose_before = actions[..., k, :]
        result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class XtacUmiRelativeActions(transforms.DataTransformFn):
    """SE(3) relative TCP actions and a gripper-only state, the UMI policy interface.

    Each side's TCP target becomes ``inv(T_t) @ T_{t+k}``: the motion expressed in the
    current gripper's own frame, so it is the same number whether poses come from the
    Pico4 world frame (recording) or a Flexiv base frame (deployment). Gripper dims stay
    absolute. The model-facing state keeps only the two gripper openings, since absolute
    TCP poses are frame-dependent and would be out of distribution on the robot.

    Without ``actions`` (inference), the 20D absolute state is stashed under
    ``transforms.INFERENCE_CONTEXT_KEY`` for `XtacUmiAbsoluteActions`.
    Works on individual samples and on batches used to compute normalization stats.
    """

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        state = np.asarray(data["state"])
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}D absolute state, got {state.shape}")

        if "actions" in data:
            actions = np.array(data["actions"], copy=True)
            for tcp in (LEFT_TCP, RIGHT_TCP):
                actions[..., tcp] = _relative_pose9(state[..., None, tcp], actions[..., tcp])
            result["actions"] = actions
        else:
            result[transforms.INFERENCE_CONTEXT_KEY] = {"absolute_state": state}

        result["state"] = np.concatenate((state[..., LEFT_GRIPPER], state[..., RIGHT_GRIPPER]), axis=-1)
        return result


@dataclasses.dataclass(frozen=True)
class XtacUmiAbsoluteActions(transforms.DataTransformFn):
    """Compose `XtacUmiRelativeActions` outputs onto the current pose: ``T_t @ dT_k``."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        result = dict(data)
        context = result.pop(transforms.INFERENCE_CONTEXT_KEY, None)
        if context is None:
            raise ValueError("Relative actions need the absolute state stashed by XtacUmiRelativeActions")
        state = np.asarray(context["absolute_state"])
        actions = np.array(data["actions"], copy=True)
        for tcp in (LEFT_TCP, RIGHT_TCP):
            actions[..., tcp] = _compose_pose9(state[..., None, tcp], actions[..., tcp])
        result["actions"] = actions
        return result


def _rotation_from_6d(r6: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns -> 3x3 matrix, re-orthonormalized with Gram-Schmidt."""
    first = r6[..., 0:3] / np.linalg.norm(r6[..., 0:3], axis=-1, keepdims=True)
    second = r6[..., 3:6] - np.sum(first * r6[..., 3:6], axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def _relative_pose9(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``inv(T_ref) @ T_target`` for ``[x, y, z, r1..r6]`` poses (broadcasting)."""
    rotation = _rotation_from_6d(reference[..., 3:9])
    rotation_t = np.swapaxes(rotation, -1, -2)
    position = np.einsum("...ij,...j->...i", rotation_t, target[..., :3] - reference[..., :3])
    relative = rotation_t @ _rotation_from_6d(target[..., 3:9])
    return np.concatenate((position, relative[..., :, 0], relative[..., :, 1]), axis=-1)


def _compose_pose9(reference: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """``T_ref @ dT`` for ``[x, y, z, r1..r6]`` poses (broadcasting); inverse of `_relative_pose9`."""
    rotation = _rotation_from_6d(reference[..., 3:9])
    position = reference[..., :3] + np.einsum("...ij,...j->...i", rotation, relative[..., :3])
    absolute = rotation @ _rotation_from_6d(relative[..., 3:9])
    return np.concatenate((position, absolute[..., :, 0], absolute[..., :, 1]), axis=-1)


def make_xtac_umi_example() -> dict:
    """A random input example in the XTac-UMI wrist-only observation format."""
    return {
        "state": np.ones((STATE_DIM,)),
        "images": {
            "left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class XtacUmiInputs(transforms.DataTransformFn):
    """Model inputs for XTac-UMI bimanual data.

    Expected inputs:
    - ``images``: dict[name, img], each ``[channel, height, width]``. Both wrist
      cameras are required; ``head`` is required only when ``use_head_camera``.
    - ``state``: ``[20]`` in the per-side-grouped layout described in the module
      docstring.
    - ``actions``: ``[action_horizon, 20]`` (training only).

    The handheld rig has no third-person camera, so by default the model's
    ``base_0_rgb`` slot gets a black image with ``image_mask=False`` — the model
    is told the slot is empty rather than being fed a meaningless frame. Benches
    that do have a head view can set ``use_head_camera`` to fill and unmask it.
    """

    # Both wrist cameras are always required.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist", "right_wrist")
    # Accepted for client compatibility; consumed only when use_head_camera is True.
    OPTIONAL_CAMERAS: ClassVar[tuple[str, ...]] = ("head",)

    # True: `head` is required and fills base_0_rgb with image_mask=True.
    # False (default): base_0_rgb is a black image with image_mask=False.
    use_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        data = _decode_xtac_umi(data)

        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS) - set(self.OPTIONAL_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected cameras {tuple(sorted(unexpected))}; got {tuple(in_images)}")

        missing = set(self.EXPECTED_CAMERAS) - set(in_images)
        if missing:
            raise ValueError(f"Missing required wrist cameras {tuple(sorted(missing))}; got {tuple(in_images)}")

        left_wrist = in_images["left_wrist"]
        right_wrist = in_images["right_wrist"]

        if self.use_head_camera:
            if "head" not in in_images:
                raise ValueError(f"use_head_camera=True but no 'head' image; got {tuple(in_images)}")
            base_image = in_images["head"]
            base_mask = np.True_
        else:
            base_image = np.zeros_like(left_wrist)
            base_mask = np.False_

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": data["state"],
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XtacUmiOutputs(transforms.DataTransformFn):
    """Model outputs for XTac-UMI bimanual data.

    Trims the model's padded action dim back to the rig's 20D layout. The 6D
    rotation needs no conversion.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :STATE_DIM])}


def _decode_xtac_umi(data: dict) -> dict:
    """Normalize image dtype/layout: float [0,1] -> uint8, and CHW -> HWC."""

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return einops.rearrange(img, "c h w -> h w c")

    return {
        **data,
        "state": np.asarray(data["state"]),
        "images": {name: convert_image(img) for name, img in data["images"].items()},
    }
