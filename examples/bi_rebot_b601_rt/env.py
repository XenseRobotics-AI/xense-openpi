"""Rebot B601 RT observations and absolute actions in the recording's 20D order."""

import threading
import time
from typing import override

from lerobot.robots.bi_rebot_b601_rt.config_bi_rebot_b601_rt import BiRebotB601RTConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import image_tools
from xense_client.runtime.environment import Environment

from examples.bi_rebot_b601_rt.recipe import validate_robot_config

logger = get_logger("BiRebotB601RTEnv")
STATE_KEYS = tuple(
    f"{side}_{key}"
    for side in ("left", "right")
    for key in ("tcp.x", "tcp.y", "tcp.z", *(f"tcp.r{i}" for i in range(1, 7)), "gripper.pos")
)
DEFAULT_CAMERA_MAP = {"head": "head", "left_wrist": "left_left_wrist", "right_wrist": "right_right_wrist"}


class BiRebotB601RTEnvironment(Environment):
    """Read cameras only in get_observation; apply_action only submits a target.

    TCP positions/rotations stay in each arm's own base frame. Rotation is the
    first two matrix columns, matching LeRobotBiRebotB601DataConfig. There is
    no additional delta conversion here; the policy server returns absolutes.
    """

    def __init__(
        self,
        robot_config: BiRebotB601RTConfig,
        *,
        go_to_start: bool = True,
        render_height: int = 224,
        render_width: int = 224,
        camera_map: dict[str, str] | None = None,
        prompt: str | None = None,
        dry_run: bool = False,
        setup_robot: bool = True,
    ):
        validate_robot_config(robot_config, go_to_start=go_to_start)
        self.config = robot_config
        self._go_to_start = go_to_start
        self._render_height = render_height
        self._render_width = render_width
        self._camera_map = dict(DEFAULT_CAMERA_MAP if camera_map is None else camera_map)
        if set(self._camera_map) != set(DEFAULT_CAMERA_MAP):
            raise ValueError("camera_map must map head, left_wrist, and right_wrist to driver camera names")
        self._prompt = prompt
        self._dry_run = dry_run
        self._first_reset = True
        self._stop_requested = threading.Event()
        self._failure: Exception | None = None
        self._last_rtc_seq = -1
        self.robot = make_robot_from_config(robot_config)
        if setup_robot:
            try:
                self.robot.connect(calibrate=False, go_to_start=go_to_start)
            except BaseException:
                # Also release peripherals after a partially completed connect.
                self.robot.disconnect(reset=False)
                raise

    @override
    def reset(self) -> None:
        self._last_rtc_seq = -1
        # connect() already waited for the initial reset to finish.
        if self._first_reset:
            self._first_reset = False
            return
        if not self._go_to_start or self._stop_requested.is_set():
            return
        self.robot.reset_to_initial_position()
        # Reading rt_moving refreshes both reset watchdogs and raises on faults
        # or timeouts. Never continue policy execution after a failed reset.
        while self.robot.rt_moving:
            if self._stop_requested.wait(0.01):
                return

    def request_stop(self) -> None:
        self._stop_requested.set()

    @override
    def is_episode_complete(self) -> bool:
        return self._stop_requested.is_set()

    def raise_if_failed(self) -> None:
        """Propagate worker failures that DecoupledRuntime otherwise only logs."""
        if self._failure is not None:
            raise self._failure

    @override
    def get_observation(self) -> dict:
        try:
            return self._get_observation()
        except Exception as error:
            self._failure = error
            self.request_stop()
            raise

    def _get_observation(self) -> dict:
        raw = self.robot.get_observation()
        state = np.array([raw[key] for key in STATE_KEYS], dtype=np.float32)
        if not np.all(np.isfinite(state)):
            raise ValueError("Robot state contains NaN or Inf")
        images = {}
        for policy_name, driver_name in self._camera_map.items():
            if driver_name not in raw:
                raise RuntimeError(f"Required camera {driver_name!r} is missing from robot observation")
            image = np.asarray(raw[driver_name])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB from {driver_name}, got {image.shape}")
            image = image_tools.resize_with_pad(image[None], self._render_height, self._render_width)[0]
            images[policy_name] = image.transpose(2, 0, 1)
        observation = {"state": state, "images": images}
        if self._prompt is not None:
            observation["prompt"] = self._prompt
        return observation

    def build_action_dict(self, actions: np.ndarray) -> dict[str, float]:
        actions = np.asarray(actions, dtype=np.float64).copy()
        if actions.shape != (20,) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Expected a finite 20D action, got shape {actions.shape}")
        # Validate BOTH rotations before the facade sends to the left arm.
        # Otherwise an invalid right pose would fail after moving the left arm.
        for side, offset in (("left", 0), ("right", 10)):
            first, second = actions[offset + 3 : offset + 6], actions[offset + 6 : offset + 9]
            norm = np.linalg.norm(first)
            if norm < 1e-8:
                raise ValueError(f"{side}: TCP rotation column 1 is zero")
            first = first / norm
            if np.linalg.norm(second - first * np.dot(first, second)) < 1e-8:
                raise ValueError(f"{side}: TCP rotation columns are parallel")
            # External TacCap/serial grippers use [0, 1]; a native seventh
            # motor uses radians and must not be clamped to that interval.
            if getattr(self.config, f"{side}_arm_config").gripper is not None:
                actions[offset + 9] = np.clip(actions[offset + 9], 0.0, 1.0)
        return dict(zip(STATE_KEYS, actions.tolist(), strict=True))

    @override
    def apply_action(self, action: dict) -> None:
        if self._stop_requested.is_set():
            return
        try:
            command = self.build_action_dict(action["actions"])
            if self._dry_run:
                logger.info(f"DRY RUN action: {command}")
                metrics = action.get("rtc_metrics")
                if metrics is not None and metrics["inference_seq"] != self._last_rtc_seq:
                    self._last_rtc_seq = metrics["inference_seq"]
                    logger.info(f"RTC: {metrics}")
                return
            t0 = time.monotonic()
            self.robot.send_action(command)
            logger.debug(f"send_action: {(time.monotonic() - t0) * 1000:.2f}ms")
        except Exception as error:
            self._failure = error
            self.request_stop()
            raise

    def disconnect(self) -> None:
        self.request_stop()
        # Always call the facade: is_connected is false if only one arm is
        # still connected, but that arm and its peripherals need cleanup too.
        self.robot.disconnect()
