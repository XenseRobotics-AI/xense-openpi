import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_bi_flexiv_example() -> dict:
    """Creates a random input example for the bi flexiv policy.

    State format (20D), both TCPs first and both grippers last:
        left_tcp.{x, y, z, r1-r6} (dims 0-8) + right_tcp.{x, y, z, r1-r6} (dims 9-17)
        + left_gripper.pos (dim 18) + right_gripper.pos (dim 19)
    """
    return {
        "state": np.ones((20,)),
        "images": {
            "head": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class BiFlexivInputs(transforms.DataTransformFn):
    """Inputs for the bi flexiv policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [20] = [left_tcp.x, left_tcp.y, left_tcp.z, left_tcp.r1..r6,
                     right_tcp.x, right_tcp.y, right_tcp.z, right_tcp.r1..r6,
                     left_gripper.pos, right_gripper.pos]
    - actions: [action_horizon, 20]

    Both TCPs come first and both grippers last - this is the order the lerobot
    driver emits (`BiFlexivRizon4RT._proprioception_ft`) and the order
    `LeRobotBiFlexivDataConfig`'s delta mask assumes, `make_bool_mask(18, -1, -1)`:
    18 TCP dims delta-encoded, the two trailing gripper dims absolute. Note this
    differs from the per-side-grouped order the XTac-UMI rig records in - see
    `xtac_umi_policy`, which keeps grippers at dims 9 and 19.

    The 6D rotation representation (r1-r6) consists of the first two columns of the rotation matrix:
    - [r1, r2, r3]: First column of rotation matrix
    - [r4, r5, r6]: Second column of rotation matrix

    This representation is continuous (no discontinuities like Euler angles at ±180°) and
    doesn't have the double-cover issue of quaternions. No special encoding/decoding is needed.
    """

    # The expected camera names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_bi_flexiv(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that head image always exists.
        head_image = in_images["head"]

        images = {
            "base_0_rgb": head_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "left_wrist",
            "right_wrist_0_rgb": "right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(head_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # No conversion needed - 6D rotation is already a continuous representation.
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BiFlexivOutputs(transforms.DataTransformFn):
    """Outputs for the bi flexiv policy.

    Model output format (20 dims), same order as the input state:
        left_tcp.{x, y, z, r1-r6} (dims 0-8) + right_tcp.{x, y, z, r1-r6} (dims 9-17)
        + left_gripper.pos (dim 18) + right_gripper.pos (dim 19)

    No conversion needed - 6D rotation is already in the correct format.
    """

    def __call__(self, data: dict) -> dict:
        # Return 20 dims (in case model outputs padded actions).
        actions = np.asarray(data["actions"][:, :20])
        return {"actions": actions}


def _decode_bi_flexiv(data: dict) -> dict:
    """Decode bi flexiv data format.

    Processing steps:
    1. Convert images from [C, H, W] to [H, W, C].

    Args:
        data: Input data dict containing 'state' and 'images'.

    Returns:
        Modified data dict with converted images.
    """
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data
