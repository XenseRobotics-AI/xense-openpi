"""The adapter passes state and actions through unchanged.

Re-based checkpoints already speak the Flexiv TCP convention, so any conversion at
this boundary would silently command wrong orientations on the arms.
"""

import numpy as np
import pytest
from xense_client import base_policy as _base_policy

from examples.xtac_umi_bi_flexiv_rizon4_rt.policy_adapter import XtacUmiPolicyAdapter


class _EchoPolicy(_base_policy.BasePolicy):
    """Returns the state it was sent as a 3-step action chunk and records the request."""

    def __init__(self) -> None:
        self.request: dict | None = None
        self.kwargs: dict | None = None

    def infer(self, obs: dict, **kwargs) -> dict:
        self.request, self.kwargs = obs, kwargs
        return {"actions": np.tile(obs["state"], (3, 1)), "state": obs["state"], "server_timing": {"infer_ms": 1.0}}


def _observation() -> dict:
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    return {
        "state": np.random.default_rng(0).normal(size=20),
        "images": {"left_wrist": image, "right_wrist": image, "head": image},
    }


def test_state_and_actions_pass_through_unchanged():
    inner = _EchoPolicy()
    observation = _observation()
    previous = np.tile(observation["state"], (2, 1))

    result = XtacUmiPolicyAdapter(inner, prompt="pick").infer(observation, prev_chunk_left_over=previous)

    np.testing.assert_array_equal(inner.request["state"], observation["state"])
    np.testing.assert_array_equal(inner.kwargs["prev_chunk_left_over"], previous)
    np.testing.assert_array_equal(result["actions"], np.tile(observation["state"], (3, 1)))
    # Only the wrist cameras and the prompt go to the server.
    assert set(inner.request["images"]) == {"left_wrist", "right_wrist"}
    assert inner.request["prompt"] == "pick"
    # The array-valued server state is not an executable action and is dropped.
    assert set(result) == {"actions", "server_timing"}


def test_missing_wrist_camera_is_rejected():
    observation = _observation()
    del observation["images"]["right_wrist"]
    with pytest.raises(ValueError, match="right_wrist"):
        XtacUmiPolicyAdapter(_EchoPolicy()).infer(observation)
