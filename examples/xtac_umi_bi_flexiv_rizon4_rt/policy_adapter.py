"""Websocket-boundary adapter between Flexiv execution and XTac-UMI policy space.

Dim order is already the policy's on both sides (``real_env.py`` assembles it
per-side-grouped), so the only conversion left is the gripper end-frame change of
basis — see ``gripper_frame.py``. With ``align_gripper_frames=False`` this adapter
is a pass-through for the numbers and does nothing but shape the request and trim
the response.
"""

from collections.abc import Mapping
from typing import Any, override

import numpy as np
from xense_client import base_policy as _base_policy

from examples.xtac_umi_bi_flexiv_rizon4_rt import gripper_frame

_WRIST_CAMERAS = ("left_wrist", "right_wrist")


class XtacUmiPolicyAdapter(_base_policy.BasePolicy):
    """Convert gripper end frames at the websocket boundary, nothing else.

    Everything below this adapter — brokers, action queues, the robot driver —
    stays in the Flexiv gripper frame; only websocket requests and responses are
    in the XTac-UMI gripper frame the training data uses.

    That boundary also covers RTC's ``prev_chunk_left_over``: those are Flexiv-frame
    actions held by the client-side queue, and the server re-bases them through the
    training input pipeline, so they need the same conversion as the state.
    """

    def __init__(
        self,
        inner: _base_policy.BasePolicy,
        *,
        align_gripper_frames: bool = True,
        prompt: str | None = None,
    ) -> None:
        self._inner = inner
        self._align = align_gripper_frames
        self._prompt = prompt

    def _convert(self, vector: np.ndarray) -> np.ndarray:
        """Flexiv <-> XTac-UMI gripper frame. Self-inverse, so one call both ways."""
        return gripper_frame.align_gripper_frames(vector) if self._align else np.asarray(vector)

    @override
    def infer(self, obs: dict, **kwargs) -> dict:
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an 'images' mapping")
        missing = set(_WRIST_CAMERAS) - set(images)
        if missing:
            raise ValueError(f"Missing required wrist cameras: {tuple(sorted(missing))}")

        server_obs: dict[str, Any] = {
            "state": self._convert(np.asarray(obs["state"])),
            "images": {camera: np.asarray(images[camera]) for camera in _WRIST_CAMERAS},
        }
        prompt = obs.get("prompt", self._prompt)
        if prompt is not None:
            server_obs["prompt"] = prompt

        server_kwargs = dict(kwargs)
        previous = server_kwargs.get("prev_chunk_left_over")
        if previous is not None:
            server_kwargs["prev_chunk_left_over"] = self._convert(np.asarray(previous))

        result = self._inner.infer(server_obs, **server_kwargs)
        if "actions" not in result:
            raise ValueError(f"Policy response is missing 'actions'; got keys {tuple(result)}")

        # Only action chunks and scalar/dict timing metadata should reach the action
        # broker. In particular drop model-space `actions_original` and the
        # array-valued server `state`, which are not executable actions.
        converted = {"actions": self._convert(np.asarray(result["actions"]))}
        for key in ("server_timing", "policy_timing"):
            if key in result:
                converted[key] = result[key]
        return converted

    @override
    def reset(self) -> None:
        self._inner.reset()

    @override
    def warmup(self, obs: dict) -> None:
        # Brokers in this repository warm up through infer(). Keeping this a no-op
        # avoids a second, subtly different conversion path.
        return None
