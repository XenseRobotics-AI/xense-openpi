"""Policy transforms for the single-arm Dobot Nova5 DH robot.

The LeRobot dataset and driver expose the same Cartesian 10D layout::

    [tcp.x, tcp.y, tcp.z, tcp.r1, tcp.r2, tcp.r3,
     tcp.r4, tcp.r5, tcp.r6, gripper.pos]

``r1-r6`` are the first two columns of the rotation matrix, so they are already
a continuous representation and do not need an additional conversion.

The robot has a head camera and one camera named ``wrist`` on the right arm.
OpenPI still presents three image slots to the model: the unused left-wrist
slot is a black image with ``image_mask=False``.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


TCP = slice(0, 9)
GRIPPER = slice(9, 10)
STATE_DIM = 10


def make_dobot_nova5_example() -> dict:
    """Create a policy input with the same keys and shapes as the Nova5 rig."""
    return {
        "state": np.ones((STATE_DIM,), dtype=np.float32),
        "images": {
            "head": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class DobotNova5Inputs(transforms.DataTransformFn):
    """Convert Nova5 LeRobot observations to OpenPI model inputs.

    Expected images are channel-first ``head`` and ``wrist`` frames. The state
    and optional action chunk use the 10D layout documented above.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_dobot_nova5(data)

        images = data["images"]
        unexpected = set(images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected cameras {tuple(sorted(unexpected))}; got {tuple(images)}")
        missing = set(self.EXPECTED_CAMERAS) - set(images)
        if missing:
            raise ValueError(f"Missing required Nova5 cameras {tuple(sorted(missing))}; got {tuple(images)}")

        head = images["head"]
        wrist = images["wrist"]
        result = {
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": np.zeros_like(wrist),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": data["state"],
        }

        if "actions" in data:
            result["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class DobotNova5Outputs(transforms.DataTransformFn):
    """Trim the model's padded action vector back to the robot's native 10D."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :STATE_DIM])}


def _decode_dobot_nova5(data: dict) -> dict:
    """Normalize LeRobot images from CHW to HWC and floats to uint8."""

    def convert_image(image):
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        return einops.rearrange(image, "c h w -> h w c")

    return {
        **data,
        "state": np.asarray(data["state"]),
        "images": {name: convert_image(image) for name, image in data["images"].items()},
    }
