"""OpenPI Environment wrapper for BiFlexiv Rizon4 RT dual-arm robot."""

from typing import override

import einops
from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import image_tools
from xense_client.runtime import environment as _environment

import examples.bi_flexiv_rizon4_rt.real_env as _real_env
import openpi.transforms as _transforms

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
        head                 -> head
        left_wrist           -> left_wrist
        right_wrist          -> right_wrist
        left_tactile_left    -> left_tactile_top
        left_tactile_right   -> left_tactile_bottom
        right_tactile_left   -> right_tactile_top
        right_tactile_right  -> right_tactile_bottom

    Tactile streams are passed through untouched; real_env already renamed them
    from lerobot's "<arm>_tactile_<finger>" keys. Whether the policy consumes
    them is decided server-side by the train config: a tactile model routes them
    to the FastViT branch, a plain pi05 policy drops the extra keys. Depth
    cameras are the only streams filtered out here.
    """

    def __init__(
        self,
        robot_config: BiFlexivRizon4RTConfig,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
        enable_tactile_sensors: bool = True,
        tactile_camera_mapping: dict[str, str] | None = None,
        capture_tactile_reference: bool = False,
        tactile_resize_mode: str = "center_crop",
    ) -> None:
        self._env = _real_env.BiFlexivRizon4RTRealEnv(
            robot_config=robot_config,
            setup_robot=setup_robot,
            enable_tactile_sensors=enable_tactile_sensors,
            tactile_camera_mapping=tactile_camera_mapping,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._step_count = 0
        self._capture_tactile_reference = capture_tactile_reference
        self._tactile_resize_mode = tactile_resize_mode
        self._tactile_reference: dict = {}

    def _refresh_tactile_reference(self) -> None:
        """Snapshot the undeformed tactile frames for this episode.

        A `..._diff` policy is trained on `frame - episode_reference`, where the
        reference is frame 0 of the episode: gripper open, gel undeformed. This is
        the inference-side counterpart, so it must be taken with the hand empty --
        right after reset, before anything is grasped. Taking it while holding
        something bakes that deformation into the zero point and every subsequent
        difference is wrong.
        """
        if not self._capture_tactile_reference:
            return
        images = self._env.get_observation()["images"]
        self._tactile_reference = {
            f"{cam}_ref": img for cam, img in images.items() if "tactile" in cam and "_depth" not in cam
        }
        if self._tactile_reference:
            logger.info(f"Captured tactile reference frames: {sorted(self._tactile_reference)}")
        else:
            logger.warn("capture_tactile_reference is on but the robot exposed no tactile cameras")

    @override
    def reset(self) -> None:
        self._env.reset()
        self._step_count = 0
        self._refresh_tactile_reference()

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

        if self._capture_tactile_reference and not self._tactile_reference:
            # reset() normally fills this. Falling back here keeps a runtime that
            # skipped reset from silently sending no reference at all -- which the
            # server rejects rather than papers over.
            logger.warn("No tactile reference captured yet; taking one now (was reset() called?)")
            self._refresh_tactile_reference()

        images = {**raw_obs["images"], **self._tactile_reference}
        processed_images = {}
        for cam_name, img in images.items():
            if "_depth" in cam_name:
                continue

            if "tactile" in cam_name:
                # Tactile geometry must match the training transform exactly, so it
                # goes through the same helper rather than a copy of the logic here.
                # `openpi.transforms.TactileDifference` re-applies fit_square
                # server-side and it is idempotent, so a mismatch shows up as a
                # wrong picture rather than a double resize -- keep
                # --tactile_resize_mode in step with the train config's
                # `tactile_resize_mode`.
                resized = _transforms.fit_square(
                    np.asarray(img), self._render_height, self._tactile_resize_mode
                )
            else:
                batch = np.expand_dims(img, axis=0)
                resized = image_tools.resize_with_pad(batch, self._render_height, self._render_width)[0]
            # (H, W, C) -> (C, H, W) for OpenPI policy input
            processed_images[cam_name] = einops.rearrange(resized, "h w c -> c h w")

        # Raw images (original resolution HWC) passed through for recording and
        # streaming. Both consumers pick cameras by name (recorder._POLICY_CAMERAS,
        # subscriber's --cameras), so carrying the tactile streams here costs a dict
        # entry by reference and lets a recorder opt into them later.
        raw_images = {cam: img for cam, img in raw_obs["images"].items() if "_depth" not in cam}

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
