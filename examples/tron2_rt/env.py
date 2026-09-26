"""OpenPI Environment wrapper for the LimX TRON2 dual-arm robot (Tron2RT driver).

Matches the layout the policy was trained on (``LeRobotTron2DataConfig``,
e.g. ``Xense/tron2rt-pnp-0918``):

State/action (20D):
    [left_tcp.x/y/z/r1-r6 (0-8), right_tcp.x/y/z/r1-r6 (9-17),
     left_gripper.pos (18), right_gripper.pos (19)]

Cameras (driver name -> policy name):
    top         -> head
    left_wrist  -> left_wrist
    right_wrist -> right_wrist
    (tactile streams are ignored by the policy)

Frames: ``Tron2RT.get_observation`` reports TCP poses in the tool-calibrated TCP
frame, and the recorded dataset actions are in that same frame, so the policy
outputs TCP-frame targets. ``Tron2RT.send_action`` however consumes seventh-axis
flange poses (the frame of the native IK). ``apply_action`` therefore maps each
predicted TCP pose back to the flange with ``T_flange = T_tcp @ inv(T_flange_tcp)``
before sending. Without a calibration file the transform is identity and this
is a no-op.
"""

import time
from typing import override

import einops
from lerobot.robots.tron2_rt import Tron2RT
from lerobot.robots.tron2_rt import tron2_rt as _tron2_rt_driver
from lerobot.robots.tron2_rt.config_tron2_rt import Tron2RTConfig
from lerobot.robots.tron2_rt.config_tron2_rt import Tron2RTControlMode
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import image_tools
from xense_client.runtime import environment as _environment

logger = get_logger("Tron2RTEnv")

_SIDES = ("left", "right")
_TCP_KEYS = {side: _tron2_rt_driver._tcp_keys(side) for side in _SIDES}
_GRIPPER_KEYS = ("left_gripper.pos", "right_gripper.pos")

# Driver camera name -> policy camera name (must match BiFlexivInputs.EXPECTED_CAMERAS).
_CAMERA_MAP = {
    "top": "head",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}

# Action dimension labels for debug logging (20D Cartesian)
_ACTION_LABELS = [
    *(f"L.{k.split('.')[1]}" for k in _TCP_KEYS["left"]),
    *(f"R.{k.split('.')[1]}" for k in _TCP_KEYS["right"]),
    "L.grip",
    "R.grip",
]


class Tron2RTEnvironment(_environment.Environment):
    """OpenPI environment for the TRON2 robot in Cartesian RT mode.

    get_observation() reads state + cameras fresh; apply_action() only submits
    a waypoint to the native 300 Hz publisher (which interpolates between
    waypoints using their measured send times), so obs and action I/O stay
    decoupled like the BiFlexiv example.
    """

    def __init__(
        self,
        robot_config: Tron2RTConfig,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        if Tron2RT is None:
            raise RuntimeError(
                "lerobot.robots.tron2_rt.Tron2RT is unavailable: the libtron2rt / tron2-ik runtime "
                "could not be imported in this environment."
            )
        if robot_config.control_mode != Tron2RTControlMode.CARTESIAN or not robot_config.observe_tcp:
            raise ValueError(
                "The TRON2 policy uses Cartesian TCP state/actions: need control_mode=cartesian and observe_tcp=True."
            )
        # The 20D vectors end in left/right_gripper.pos, so both sides need a gripper.
        missing = [
            side
            for side, cfg in (("left", robot_config.left_gripper), ("right", robot_config.right_gripper))
            if cfg is None
        ]
        if missing:
            raise ValueError(
                f"No gripper configured on: {', '.join(missing)}. The policy's state and action "
                "vectors are 20D ending in left/right_gripper.pos, so both sides need one."
            )

        self.config = robot_config
        self.robot = Tron2RT(robot_config)
        self._render_height = render_height
        self._render_width = render_width
        self._step_count = 0
        self._episode_count = 0
        # T_flange_tcp per side; filled in on connect.
        self._inv_tool = {side: np.eye(4) for side in _SIDES}

        if setup_robot:
            self.setup_robot()

    def setup_robot(self) -> None:
        logger.info(f"Connecting to TRON2 RT at {self.config.robot_ip} (go_to_start={self.config.go_to_start})...")
        self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
        # Use the exact transforms the driver applies to its TCP observations, so
        # the TCP -> flange inverse is guaranteed to match.
        self.robot._load_tool_transforms()
        for side in _SIDES:
            tool = np.asarray(self.robot._tool_transforms[side], dtype=np.float64)
            self._inv_tool[side] = np.linalg.inv(tool)
            if not np.allclose(tool, np.eye(4), atol=1e-12):
                logger.info(f"{side} TCP calibration active: t_flange_tcp={tool[:3, 3].round(4).tolist()}")
        logger.info("TRON2 RT connected and ready")

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        # connect() already moved to the start pose for the first episode.
        if self._episode_count > 1 and self.config.go_to_start:
            logger.info("Resetting TRON2 RT to start pose...")
            self.robot.reset_to_initial_position()
            logger.info("TRON2 RT reset completed")

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        raw_obs = self.robot.get_observation()

        state = np.array(
            [raw_obs[k] for side in _SIDES for k in _TCP_KEYS[side]] + [raw_obs[k] for k in _GRIPPER_KEYS],
            dtype=np.float32,
        )

        raw_images = {}
        processed_images = {}
        for src, dst in _CAMERA_MAP.items():
            img = raw_obs.get(src)
            if img is None:
                logger.debug(f"Camera {src} not found in observation")
                continue
            raw_images[dst] = img
            resized = image_tools.resize_with_pad(img[None], self._render_height, self._render_width)[0]
            # (H, W, C) -> (C, H, W) for OpenPI policy input
            processed_images[dst] = einops.rearrange(resized, "h w c -> c h w")

        return {
            "state": state,
            "images": processed_images,
            "images_raw": raw_images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1
        actions = action.get("actions")
        if actions is None:
            return
        parts = " | ".join(f"{lbl}={v:+.4f}" for lbl, v in zip(_ACTION_LABELS, actions))
        logger.debug(f"Step {self._step_count}: {parts}")
        self.robot.send_action(self._build_action_dict(np.asarray(actions)))

    def _build_action_dict(self, action: np.ndarray) -> dict[str, float]:
        """20D TCP-frame policy action -> Tron2RT flange-frame action dict."""
        action_dict: dict[str, float] = {}
        for side, offset in (("left", 0), ("right", 9)):
            tcp = _tron2_rt_driver._pose9d_to_matrix(action[offset : offset + 9], side)
            flange = tcp @ self._inv_tool[side]
            pose9d = _tron2_rt_driver._matrix_to_pose9d(flange, side)
            action_dict.update(zip(_TCP_KEYS[side], pose9d, strict=True))
        action_dict["left_gripper.pos"] = float(np.clip(action[18], 0.0, 1.0))
        action_dict["right_gripper.pos"] = float(np.clip(action[19], 0.0, 1.0))
        return action_dict

    def disconnect(self) -> None:
        # Not gated on robot.is_connected: that aggregate is False as soon as a
        # camera or gripper is unhealthy, which is exactly when the arm still
        # needs its safe reset and RT shutdown. Tron2RT.disconnect() is a no-op
        # when nothing is connected. reset_on_disconnect (default True) returns
        # the robot to the start pose.
        logger.info("Disconnecting TRON2 RT...")
        try:
            self.robot.disconnect()
            time.sleep(0.5)
            logger.info("TRON2 RT disconnected")
        except Exception as e:
            logger.warn(f"Error during disconnect: {e}")
