"""OpenPI Environment wrapper for BiFlexiv Rizon4 RT dual-arm robot."""

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client import image_tools
from xense_client.runtime import environment as _environment

import examples.bi_flexiv_rizon4_rt.real_env as _real_env

logger = get_logger("BiFlexivRizon4RTEnv")

# Action dimension labels for debug logging (20D Cartesian)
_ACTION_LABELS = [
    "L.x",
    "L.y",
    "L.z",
    "L.r1",
    "L.r2",
    "L.r3",
    "L.r4",
    "L.r5",
    "L.r6",
    "R.x",
    "R.y",
    "R.z",
    "R.r1",
    "R.r2",
    "R.r3",
    "R.r4",
    "R.r5",
    "R.r6",
    "L.grip",
    "R.grip",
]


class BiFlexivRizon4RTEnvironment(_environment.Environment):
    """OpenPI environment for BiFlexiv Rizon4 RT dual-arm robot.

    Obs and action I/O are decoupled at this layer: get_observation() always
    reads the cameras + robot state fresh (~33 ms), and apply_action() only
    sends a target pose to the SHM (~0.2 ms). This lets a multi-threaded
    runtime drive each on its own schedule — necessary for DecoupledRuntime,
    where obs runs at camera FPS (~30 Hz) and action runs at action_hz
    (e.g. 60 Hz). For the synchronous Runtime the only change vs. the old
    dm_env-style coupling is that obs paired with each action is now the
    "obs before this action" rather than "obs after the previous action" —
    matching the lerobot recorder convention used to train this stack.

    Camera name mapping (real → policy):
        head        -> head
        left_wrist  -> left_wrist
        right_wrist -> right_wrist
        (tactile cameras are silently ignored by the policy)
    """

    def __init__(
        self,
        bi_mount_type: str = "forward",
        use_force: bool = False,
        go_to_start: bool = True,
        stiffness_ratio: float = 0.2,
        inner_control_hz: int = 1000,
        interpolate_cmds: bool = True,
        enable_tactile_sensors: bool = True,
        log_level: str = "INFO",
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        self._env = _real_env.BiFlexivRizon4RTRealEnv(
            bi_mount_type=bi_mount_type,
            use_force=use_force,
            go_to_start=go_to_start,
            stiffness_ratio=stiffness_ratio,
            inner_control_hz=inner_control_hz,
            interpolate_cmds=interpolate_cmds,
            enable_tactile_sensors=enable_tactile_sensors,
            log_level=log_level,
            setup_robot=setup_robot,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._step_count = 0

    @override
    def reset(self) -> None:
        self._env.reset()
        self._step_count = 0

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        # Reads cameras + robot state fresh. ~33 ms on this stack (camera-
        # bound). Returns the obs the policy should see for THIS step's
        # action — not a one-step-stale cache populated by the previous
        # apply_action.
        raw_obs = self._env.get_observation()

        processed_images = {}
        for cam_name, img in raw_obs["images"].items():
            if "_depth" in cam_name or "tactile" in cam_name:
                continue

            batch = np.expand_dims(img, axis=0)
            resized = image_tools.resize_with_pad(batch, self._render_height, self._render_width)[0]
            # (H, W, C) -> (C, H, W) for OpenPI policy input
            processed_images[cam_name] = einops.rearrange(resized, "h w c -> c h w")

        # Raw images (original resolution HWC) passed through for recording.
        # Policy cameras only — tactile sensors excluded.
        raw_images = {
            cam: img for cam, img in raw_obs["images"].items() if "_depth" not in cam and "tactile" not in cam
        }

        return {
            "state": raw_obs["qpos"],
            "images": processed_images,
            "images_raw": raw_images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1
        actions = action.get("actions")
        if actions is not None:
            parts = " | ".join(f"{lbl}={v:+.4f}" for lbl, v in zip(_ACTION_LABELS, actions))
            logger.debug(f"Step {self._step_count}: {parts}")
        # Pure send — no observation read. The outer loop owns obs scheduling.
        self._env.send_action(action["actions"])

    def disconnect(self) -> None:
        self._env.disconnect()
