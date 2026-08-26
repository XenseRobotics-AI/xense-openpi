"""Aloha policy with tactile image support.

This extends the base Aloha policy to handle tactile images in addition to
the standard camera images.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

import openpi.transforms as transforms


def make_aloha_tactile_example() -> dict:
    """Create an example input for testing."""
    return {
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "left_tactile_left": np.zeros((3, 224, 224), dtype=np.uint8),
            "right_tactile_left": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "state": np.zeros(14, dtype=np.float32),
        "actions": np.zeros((50, 14), dtype=np.float32),
        "prompt": "pick up cubes",
    }


@dataclasses.dataclass(frozen=True)
class AlohaTactileInputs(transforms.DataTransformFn):
    """Inputs for the Aloha Tactile policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]
      - Visual cameras: cam_high, cam_left_wrist, cam_right_wrist
      - Tactile sensors: left_tactile_left, right_tactile_left
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The expected camera names (visual + tactile)
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        # Visual cameras (standard Aloha)
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
        # Tactile sensors. Not `cam_`-prefixed like the RGB names above: those
        # are Aloha's, while a tactile stream is named by the arm and the jaw its
        # pad sits on (`<arm>_tactile_<finger>`, serial parity) — the key lerobot
        # hands over, carried through unchanged so nothing has to be renamed.
        "left_tactile_left",
        "right_tactile_left",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_tactile_aloha(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain subset of {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Base image must exist
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add visual wrist cameras
        visual_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }

        for dest, source in visual_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # Add tactile sensors
        tactile_image_names = {
            "left_tactile_left_rgb": "left_tactile_left",
            "right_tactile_left_rgb": "right_tactile_left",
        }

        for dest, source in tactile_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                # Assume tactile images are always present.
                raise ValueError(f"Tactile image {source} not found in input data")

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaTactileOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha Tactile policy."""

    # Do Nothing

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}


def _decode_tactile_aloha(data: dict) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
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
