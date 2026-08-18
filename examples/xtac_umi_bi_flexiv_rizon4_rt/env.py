"""OpenPI environment for XTac-UMI inference on a BiFlexiv Rizon4 RT robot.

State/action format (20D, XTac-UMI per-side-grouped order — see ``real_env.py``)::

    [left_tcp.x/y/z/r1-r6 (0-8), left_gripper.pos (9),
     right_tcp.x/y/z/r1-r6 (10-18), right_gripper.pos (19)]

This layer is in the same dim order as the policy, so it does no conversion at
all. The only remaining conversion — the gripper end-frame change of basis —
lives in ``policy_adapter.XtacUmiPolicyAdapter``, right at the websocket
boundary.
"""

from typing import override

import einops
from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import image_tools
from xense_client.runtime import environment as _environment

from examples.xtac_umi_bi_flexiv_rizon4_rt.real_env import STATE_KEYS
from examples.xtac_umi_bi_flexiv_rizon4_rt.real_env import XtacUmiBiFlexivRizon4RTRealEnv

logger = get_logger("XtacUmiBiFlexivRizon4RTEnv")

# Policy-facing camera names. XTac-UMI data has no third-person view, so only the
# two wrist cameras are connected and sent; the model's base_0_rgb slot is filled
# with a black image and masked out server-side (see xtac_umi_policy.XtacUmiInputs).
_WRIST_CAMERAS = ("left_wrist", "right_wrist")


class XtacUmiBiFlexivRizon4RTEnvironment(_environment.Environment):
    """OpenPI environment exposing 20D XTac-UMI-ordered obs/actions.

    Same obs/action decoupling as examples/bi_flexiv_rizon4_rt: get_observation()
    reads cameras + robot state fresh, and apply_action() only sends a target pose
    — no observation read. The outer runtime loop owns obs scheduling.

    Camera name mapping (real -> policy):
        left_wrist  -> left_wrist
        right_wrist -> right_wrist
        (no head camera; XTac-UMI checkpoints mask the base_0_rgb slot out)
    """

    def __init__(
        self,
        robot_config: BiFlexivRizon4RTConfig,
        *,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        self._env = XtacUmiBiFlexivRizon4RTRealEnv(robot_config, setup_robot=setup_robot)
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
        for camera in _WRIST_CAMERAS:
            # Unlike bi_flexiv_rizon4_rt (which silently skips a missing camera),
            # this is fatal: the checkpoint was trained on exactly these two views,
            # so running without one feeds the policy a silently wrong observation.
            if camera not in raw_obs["images"]:
                raise RuntimeError(f"Required camera {camera!r} is missing; got {tuple(raw_obs['images'])}")
            resized = image_tools.resize_with_pad(
                np.expand_dims(raw_obs["images"][camera], axis=0), self._render_height, self._render_width
            )[0]
            # (H, W, C) -> (C, H, W) for OpenPI policy input.
            images[camera] = einops.rearrange(resized, "h w c -> c h w")

        # Fail fast on a malformed state vector: a bad read would otherwise reach
        # the driver as a large, hard-to-debug arm motion.
        state = np.asarray(raw_obs["qpos"], dtype=np.float32)
        if state.shape != (len(STATE_KEYS),) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"Invalid state: shape={state.shape}, finite={np.all(np.isfinite(state))}")
        return {"state": state, "images": images}

    @override
    def apply_action(self, action: dict) -> None:
        # Validate before sending: the RT driver executes this verbatim.
        actions = np.asarray(action["actions"], dtype=np.float32)
        if actions.shape != (len(STATE_KEYS),):
            raise ValueError(f"Expected one 20D action, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Refusing to execute an action containing NaN or Inf")
        logger.debug(f"action: {dict(zip(STATE_KEYS, actions, strict=True))}")
        # Pure send — no observation read. The outer loop owns obs scheduling.
        self._env.send_action(actions)

    def disconnect(self) -> None:
        self._env.disconnect()
