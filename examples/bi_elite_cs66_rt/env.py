"""OpenPI Environment wrapper for the bimanual Elite CS66 RT robot.

State/action format (20D, per-side-grouped — see ``real_env.py``)::

    [left_tcp.x/y/z/r1-r6 (0-8), left_gripper.pos (9),
     right_tcp.x/y/z/r1-r6 (10-18), right_gripper.pos (19)]
"""

from typing import override

import einops
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import image_tools
from xense_client.runtime import environment as _environment

from examples.bi_elite_cs66_rt.real_env import POLICY_CAMERAS
from examples.bi_elite_cs66_rt.real_env import STATE_KEYS
from examples.bi_elite_cs66_rt.real_env import BiEliteCS66RTRealEnv

logger = get_logger("BiEliteCS66RTEnv")


class BiEliteCS66RTEnvironment(_environment.Environment):
    """OpenPI environment exposing 20D per-side-grouped obs/actions.

    Same obs/action decoupling as examples/bi_flexiv_rizon4_rt: get_observation()
    reads cameras + robot state fresh, and apply_action() only submits a target
    pose — no observation read. The outer runtime loop owns obs scheduling, so
    DecoupledRuntime can drive each on its own schedule.

    Camera name mapping (real -> policy): head, left_wrist, right_wrist pass
    through unchanged; tactile cameras are not sent.
    """

    def __init__(
        self,
        robot_config: BiEliteCS66RTConfig,
        *,
        go_to_start: bool = True,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        self._env = BiEliteCS66RTRealEnv(robot_config, go_to_start=go_to_start, setup_robot=setup_robot)
        self._render_height = render_height
        self._render_width = render_width

    @override
    def reset(self) -> None:
        self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        raw_obs = self._env.get_observation()
        images = {}
        for camera in POLICY_CAMERAS:
            # The checkpoint was trained on exactly these three views; running
            # without one feeds the policy a silently wrong observation.
            if camera not in raw_obs["images"]:
                raise RuntimeError(f"Required camera {camera!r} is missing; got {tuple(raw_obs['images'])}")
            resized = image_tools.resize_with_pad(
                np.expand_dims(raw_obs["images"][camera], axis=0), self._render_height, self._render_width
            )[0]
            # (H, W, C) -> (C, H, W) for OpenPI policy input.
            images[camera] = einops.rearrange(resized, "h w c -> c h w")

        # Fail fast on a malformed state vector: a bad read would otherwise reach
        # the driver as a large, hard-to-debug arm motion.
        state = raw_obs["qpos"]
        if state.shape != (len(STATE_KEYS),) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"Invalid state: shape={state.shape}, finite={np.all(np.isfinite(state))}")
        return {"state": state, "images": images}

    @override
    def apply_action(self, action: dict) -> None:
        # Validate before sending: the servo loop executes this verbatim.
        actions = np.asarray(action["actions"], dtype=np.float32)
        if actions.shape != (len(STATE_KEYS),):
            raise ValueError(f"Expected one 20D action, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Refusing to execute an action containing NaN or Inf")
        logger.debug(f"action: {dict(zip(STATE_KEYS, actions.tolist(), strict=True))}")
        # Pure send — no observation read. The outer loop owns obs scheduling.
        self._env.send_action(actions)

    def disconnect(self) -> None:
        self._env.disconnect()
