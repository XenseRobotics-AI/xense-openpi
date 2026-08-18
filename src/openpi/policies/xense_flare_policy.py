import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_xense_flare_example() -> dict:
    """Creates a random input example for the xense flare policy."""
    return {
        "state": np.ones((10,)),
        "images": {
            "observation/wrist_image_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class XenseFlareInputs(transforms.DataTransformFn):
    """Inputs for the xense flare policy.

    Expected inputs (dataset format, 10 dims with 6D rotation representation):
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: array [10] = [x, y, z, r1, r2, r3, r4, r5, r6, gripper_pos]
    - actions: array [action_horizon, 10]

    The 6D rotation representation (r1-r6) consists of the first two columns of the rotation matrix:
    - [r1, r2, r3]: First column of rotation matrix
    - [r4, r5, r6]: Second column of rotation matrix

    This representation is continuous (no discontinuities like Euler angles at ±180°)
    and doesn't have the double-cover issue of quaternions (q and -q).

    Output format (model format, same 10 dims - no conversion needed):
    - state: [10] = [x, y, z, r1, r2, r3, r4, r5, r6, gripper_pos]
    - actions: [action_horizon, 10]
    """

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("observation/wrist_image_left",)

    def __call__(self, data: dict) -> dict:
        # Decode images to model format
        data = _decode_xense_flare(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that left_wrist_0_rgb image always exists.
        wrist_image_left = in_images["observation/wrist_image_left"]

        images = {
            "left_wrist_0_rgb": wrist_image_left,
        }
        image_masks = {
            "left_wrist_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "base_0_rgb": "cam_high",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(wrist_image_left)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # No conversion needed - 6D rotation is already continuous
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XenseFlareOutputs(transforms.DataTransformFn):
    """Outputs for the Xense Flare policy.

    Model output format (10 dims, 6D rotation representation):
        [x, y, z, r1, r2, r3, r4, r5, r6, gripper_pos]

    Output format (same 10 dims - no conversion needed):
        [x, y, z, r1, r2, r3, r4, r5, r6, gripper_pos]

    The 6D rotation can be converted back to quaternion using Gram-Schmidt
    orthogonalization if needed by the robot interface.
    """

    def __call__(self, data: dict) -> dict:
        # Return 10 dims (in case model outputs padded actions)
        actions = np.asarray(data["actions"][:, :10])
        # No conversion needed - 6D rotation is already in the correct format
        return {"actions": actions}


def _decode_xense_flare(data: dict) -> dict:
    """Decode xense flare data format.

    Input/Output format (10 dims, 6D rotation representation):
        [x, y, z, r1, r2, r3, r4, r5, r6, gripper_pos]

    Processing steps:
    1. Convert images from [C, H, W] to [H, W, C]

    Args:
        data: Input data dict containing 'state' and 'images'

    Returns:
        Modified data dict with converted images
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
