from typing import override

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import image_tools
from xense_client.runtime import environment as _environment

import examples.bi_arx5_real.real_env as _real_env

logger = get_logger("BiARX5Env")


class BiARX5RealEnvironment(_environment.Environment):
    """An environment for BiARX5 robot on real hardware based on lerobot implementation."""

    def __init__(
        self,
        left_arm_port: str = "can1",
        right_arm_port: str = "can3",
        log_level: str = "INFO",
        use_multithreading: bool = True,
        enable_tactile_sensors: bool = False,
        reset_position: list[float] | None = None,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,  # whether to connect robot hardware immediately
        controller_dt: float = 0.002,  # low-level control frequency (seconds)
        preview_time: float = 0.02,  # preview time (seconds)
        control_mode: str = "teach_mode",  # control mode for robot
    ) -> None:
        self._env = _real_env.BiARX5RealEnv(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            log_level=log_level,
            use_multithreading=use_multithreading,
            enable_tactile_sensors=enable_tactile_sensors,
            reset_position=reset_position,
            setup_robot=setup_robot,
            controller_dt=controller_dt,
            preview_time=preview_time,
            control_mode=control_mode,
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

        # create new image dict to avoid modifying original data
        processed_images = {}

        # process image data - remove depth images and resize
        for cam_name in obs["images"]:
            if "_depth" in cam_name:
                continue  # skip depth images

            # original image is already uint8 format (480, 640, 3)
            single_img = obs["images"][cam_name]

            # debug: print original image info
            logger.debug(f"Camera {cam_name}: original shape={single_img.shape}, dtype={single_img.dtype}")

            # expand single image to batch format [1, H, W, C] for resize_with_pad
            batch_img = np.expand_dims(single_img, axis=0)  # [H, W, C] -> [1, H, W, C]
            logger.debug(f"Camera {cam_name}: batch shape={batch_img.shape}")

            # resize image to specified resolution
            resized_batch = image_tools.resize_with_pad(batch_img, self._render_height, self._render_width)
            logger.debug(f"Camera {cam_name}: resized batch shape={resized_batch.shape}")

            # extract first image from batch [1, H, W, C] -> [H, W, C]
            resized_img = resized_batch[0]  # already uint8 format
            logger.debug(f"Camera {cam_name}: resized shape={resized_img.shape}")

            # convert to OpenPI expected format (H,W,C) -> (C,H,W)
            processed_images[cam_name] = einops.rearrange(resized_img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": processed_images,  # use newly created dict
            # Raw images (original resolution HWC) for recording, depth/tactile excluded.
            "images_raw": {
                cam: img for cam, img in obs["images"].items() if "_depth" not in cam and "tactile" not in cam
            },
            # prompt is injected by policy server's InjectDefaultPrompt, no need to add here
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
