"""Tianji dual-arm Cartesian control with two 20-DoF Wuji hands.

State/action layout (58D):
    left_tcp.{x, y, z, r1-r6} (dims 0-8) + right_tcp.{x, y, z, r1-r6} (dims 9-17)
    + left fingers (dims 18-37) + right fingers (dims 38-57)

Finger order per hand is index, middle, pinky, ring, thumb, four joints each.
The recorded `observation.state` is 86D: the 58D prefix above followed by 28 arm
joint positions/velocities, which are dropped by `transforms.TruncateState`
before these transforms run (see `LeRobotBiTianjiWujiDataConfig`).
"""

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms
from openpi.policies import bi_flexiv_policy

ACTION_DIM = 58


@dataclasses.dataclass(frozen=True)
class BiTianjiWujiInputs(bi_flexiv_policy.BiFlexivInputs):
    """Reuses the BiFlexiv camera mapping (head, left_wrist, right_wrist) with the 58D state/action.

    Depth streams recorded alongside the RGB views are not model inputs.
    """

    ACTION_DIM: ClassVar[int] = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"BiTianjiWuji expects a ({self.ACTION_DIM},) state after TruncateState, got {state.shape}"
            )
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != self.ACTION_DIM:
                raise ValueError(
                    f"BiTianjiWuji expects actions of shape (horizon, {self.ACTION_DIM}), got {actions.shape}"
                )
        return super().__call__(data)


@dataclasses.dataclass(frozen=True)
class BiTianjiWujiOutputs(transforms.DataTransformFn):
    """Returns the 58 TCP/finger action dims, dropping any model padding."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
