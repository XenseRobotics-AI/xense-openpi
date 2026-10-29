"""Policy transforms for the bimanual Elite CS66 RT robot (``bi_elite_cs66_rt``).

State/action layout (20D) follows the lerobot driver's feature order verbatim
(``BiEliteCS66RT.observation_features``, which emits per side in turn)::

    [ left_tcp.x, left_tcp.y, left_tcp.z, left_tcp.r1..r6,      # dims 0-8
      left_gripper.pos,                                         # dim  9
      right_tcp.x, right_tcp.y, right_tcp.z, right_tcp.r1..r6,  # dims 10-18
      right_gripper.pos ]                                       # dim  19

Note this is NOT the BiFlexiv order, which groups both TCPs first and both
grippers last (dims 18 and 19). Nothing regroups dims: the deployment example
(``examples/bi_elite_cs66_rt``) assembles its state vector in this order from
the driver's named keys.

TCP poses are in the driver's gravity-aligned world frame (each arm's tilted
base frame is lifted by the recipe's mount rotation), so a checkpoint only
transfers to a bench whose recipe reproduces the recording bench's world frame.

The 6D rotation (r1-r6) is the first two COLUMNS of the rotation matrix. It is
already continuous, so these transforms pass rotations through untouched.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

# Per-side-grouped dim slices.
LEFT_TCP = slice(0, 9)
LEFT_GRIPPER = slice(9, 10)
RIGHT_TCP = slice(10, 19)
RIGHT_GRIPPER = slice(19, 20)

STATE_DIM = 20

# TCP dims (0-8, 10-18) are delta-encoded w.r.t. the current state; the two
# gripper dims (9, 19) stay absolute.
DELTA_ACTION_MASK = transforms.make_bool_mask(9, -1, 9, -1)


def make_bi_elite_example() -> dict:
    """Creates a random input example for the bi elite policy."""
    return {
        "state": np.ones((STATE_DIM,)),
        "images": {
            "head": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class BiEliteInputs(transforms.DataTransformFn):
    """Inputs for the bi elite policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
      head is required; a missing wrist camera is replaced with a black image and image_mask=False.
    - state: [20], per-side-grouped (see module docstring).
    - actions: [action_horizon, 20], same order as state.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = {name: _convert_image(img) for name, img in data["images"].items()}
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")
        if "head" not in in_images:
            raise ValueError(f"The head camera is required, got {tuple(in_images)}")

        head_image = in_images["head"]
        images = {"base_0_rgb": head_image}
        image_masks = {"base_0_rgb": np.True_}
        for dest, source in (("left_wrist_0_rgb", "left_wrist"), ("right_wrist_0_rgb", "right_wrist")):
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(head_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BiEliteOutputs(transforms.DataTransformFn):
    """Outputs for the bi elite policy: the first 20 dims, in the same order as the state."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :STATE_DIM])}


def _convert_image(img) -> np.ndarray:
    img = np.asarray(img)
    # Convert to uint8 if using float images.
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # Convert from [channel, height, width] to [height, width, channel].
    return einops.rearrange(img, "c h w -> h w c")
