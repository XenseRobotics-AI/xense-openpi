"""OpenPI Environment wrapper for Flexiv Rizon4 RT robot."""

from typing import ClassVar

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client import image_tools
from xense_client.runtime import environment as _environment

import examples.flexiv_rizon4_rt.real_env as _real_env

logger = get_logger("FlexivRizon4RTEnv")


class FlexivRizon4RTEnvironment(_environment.Environment):
    """An environment for Flexiv Rizon4 RT robot on real hardware.

    Uses the RT driver (flexiv_rt, 1 kHz C++ RT thread) for deterministic
    streaming Cartesian motion force control.

    Camera name mapping:
        Real env:   wrist_cam            -> policy: observation/wrist_image_left
        Real env:   <external_cam_key>   -> policy: <external_cam_key>
    """

    # Camera name mapping from real environment to policy expected names
    CAMERA_NAME_MAP: ClassVar[dict] = {
        "wrist_cam": "observation/wrist_image_left",
    }

    def __init__(
        self,
        robot_recipe: str,
        use_force: bool | None = None,
        go_to_start: bool | None = None,
        log_level: str | None = None,
        stiffness_ratio: float | None = None,
        start_position_degree: list[float] | None = None,
        zero_ft_sensor_on_connect: bool | None = None,
        inner_control_hz: int | None = None,
        interpolate_cmds: bool | None = None,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        self._env = _real_env.FlexivRizon4RTRealEnv(
            robot_recipe=robot_recipe,
            use_force=use_force,
            go_to_start=go_to_start,
            log_level=log_level,
            stiffness_ratio=stiffness_ratio,
            start_position_degree=start_position_degree,
            zero_ft_sensor_on_connect=zero_ft_sensor_on_connect,
            inner_control_hz=inner_control_hz,
            interpolate_cmds=interpolate_cmds,
            setup_robot=setup_robot,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        processed_images = {}

        for cam_name in obs["images"]:
            if "_depth" in cam_name:
                continue

            # Map known camera names; pass through external cameras as-is
            policy_cam_name = self.CAMERA_NAME_MAP.get(cam_name, cam_name)

            single_img = obs["images"][cam_name]
            logger.debug(f"Camera {cam_name}: shape={single_img.shape}, dtype={single_img.dtype}")

            batch_img = np.expand_dims(single_img, axis=0)
            resized_batch = image_tools.resize_with_pad(batch_img, self._render_height, self._render_width)
            resized_img = resized_batch[0]

            # (H, W, C) -> (C, H, W) for OpenPI
            processed_images[policy_cam_name] = einops.rearrange(resized_img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": processed_images,
            # prompt is injected by the policy server's InjectDefaultPrompt
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        self._env.disconnect()
