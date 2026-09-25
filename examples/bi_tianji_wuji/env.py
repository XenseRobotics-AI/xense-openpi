"""Hardware lifecycle and RGB/state conversion for Tianji/Wuji deployment."""

import logging
import time

import numpy as np
from xense_client import image_tools

from examples.bi_tianji_wuji.schema import ACTION_KEYS
from examples.bi_tianji_wuji.schema import CAMERAS
from examples.bi_tianji_wuji.schema import action_to_dict
from examples.bi_tianji_wuji.schema import state_from_observation

logger = logging.getLogger(__name__)


class TianjiWujiEnvironment:
    def __init__(self, robot, *, dry_run: bool, render_size: int = 224, reset_timeout_s: float = 15.0):
        self.robot = robot
        self.dry_run = dry_run
        self.render_size = render_size
        self.reset_timeout_s = reset_timeout_s

    def connect(self) -> None:
        if set(self.robot.action_features) != set(ACTION_KEYS):
            raise ValueError("Robot action schema does not match the trained 58D Tianji/Wuji policy")
        self.robot.connect(calibrate=False)

    def reset(self) -> None:
        if self.dry_run:
            return
        self.robot.start_reset_to_initial_position()
        deadline = time.monotonic() + self.reset_timeout_s
        while not self.robot.wait_for_reset_completion(0.05):
            if time.monotonic() >= deadline:
                raise TimeoutError("Tianji/Wuji reset did not complete; refusing policy execution")

    def hold(self) -> None:
        if not self.dry_run:
            # Native cancel holds the last arm command and ends streaming.
            # Wuji's own worker continues holding its last finger target.
            self.robot.cancel()

    def get_observation(self, prompt: str) -> dict:
        raw = self.robot.get_observation()
        images = {}
        for name in CAMERAS:
            frame = np.asarray(raw[name])
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"Camera {name} must return uint8 HWC RGB, got {frame.shape}/{frame.dtype}")
            resized = image_tools.resize_with_pad(frame[None], self.render_size, self.render_size)[0]
            images[name] = np.transpose(resized, (2, 0, 1))
        return {"state": state_from_observation(raw), "images": images, "prompt": prompt}

    def apply_action(self, action: np.ndarray) -> None:
        command = action_to_dict(action)
        if not self.dry_run:
            self.robot.send_action(command)

    def disconnect(self) -> None:
        # Also release partially-connected devices; do not gate on the
        # composite is_connected property (one failed hand makes it false).
        self.robot.disconnect()
