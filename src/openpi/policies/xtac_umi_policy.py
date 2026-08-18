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

STATE_DIM = 20


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
