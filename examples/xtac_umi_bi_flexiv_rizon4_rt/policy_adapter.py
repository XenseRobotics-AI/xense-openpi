"""Websocket-boundary adapter between Flexiv execution and the XTac-UMI policy server.

No numbers are converted here. Dim order is already the policy's on both sides
(``real_env.py`` assembles it per-side-grouped), and checkpoints are trained on
re-based data (``rebase_umi_dataset.py``), whose state and actions are Flexiv
base-frame poses in the Flexiv TCP convention. The adapter only shapes the request
and trims the response.
"""

from collections.abc import Mapping
from typing import Any, override

import numpy as np
from xense_client import base_policy as _base_policy

_WRIST_CAMERAS = ("left_wrist", "right_wrist")


class XtacUmiPolicyAdapter(_base_policy.BasePolicy):
    """Send the policy server only what an XTac-UMI checkpoint consumes.

    Requests carry the 20D state, the two wrist cameras and the prompt; responses are
    trimmed to the action chunk and timing metadata. RTC's ``prev_chunk_left_over``
    passes through unchanged, like the state it is re-based against on the server.
    """

    def __init__(self, inner: _base_policy.BasePolicy, *, prompt: str | None = None) -> None:
        self._inner = inner
        self._prompt = prompt

    @override
    def infer(self, obs: dict, **kwargs) -> dict:
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an 'images' mapping")
        missing = set(_WRIST_CAMERAS) - set(images)
        if missing:
            raise ValueError(f"Missing required wrist cameras: {tuple(sorted(missing))}")

        server_obs: dict[str, Any] = {
            "state": np.asarray(obs["state"]),
            "images": {camera: np.asarray(images[camera]) for camera in _WRIST_CAMERAS},
        }
        prompt = obs.get("prompt", self._prompt)
        if prompt is not None:
            server_obs["prompt"] = prompt

        server_kwargs = dict(kwargs)
        previous = server_kwargs.get("prev_chunk_left_over")
        if previous is not None:
            server_kwargs["prev_chunk_left_over"] = np.asarray(previous)

        result = self._inner.infer(server_obs, **server_kwargs)
        if "actions" not in result:
            raise ValueError(f"Policy response is missing 'actions'; got keys {tuple(result)}")

        # Only action chunks and scalar/dict timing metadata should reach the action
        # broker. In particular drop model-space `actions_original` and the
        # array-valued server `state`, which are not executable actions.
        trimmed = {"actions": np.asarray(result["actions"])}
        for key in ("server_timing", "policy_timing"):
            if key in result:
                trimmed[key] = result[key]
        return trimmed

    @override
    def reset(self) -> None:
        self._inner.reset()

    @override
    def warmup(self, obs: dict) -> None:
        # Brokers in this repository warm up through infer(). Keeping this a no-op
        # avoids a second, subtly different request path.
        return None
