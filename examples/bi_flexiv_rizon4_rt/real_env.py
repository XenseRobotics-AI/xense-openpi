"""Real environment for BiFlexiv Rizon4 RT dual-arm robot.

Wraps lerobot BiFlexivRizon4RT for use with OpenPI inference.

State/action format (20D):
    [left_tcp.x/y/z/r1-r6 (0-8), right_tcp.x/y/z/r1-r6 (9-17),
     left_gripper.pos (18), right_gripper.pos (19)]
"""

import collections
import time

import dm_env
from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import emergency_stop_flexiv_rt_robot
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("BiFlexivRizon4RTRealEnv")

# Policy-facing camera names (must match BiFlexivInputs.EXPECTED_CAMERAS)
_POLICY_CAMERAS = ("head", "left_wrist", "right_wrist")


class BiFlexivRizon4RTRealEnv:
    """Environment for BiFlexiv Rizon4 RT dual-arm robot.

    State/action space (20D):
        left_tcp.{x,y,z,r1-r6} (9D) + right_tcp.{x,y,z,r1-r6} (9D)
        + left_gripper.pos (1D) + right_gripper.pos (1D)

    Observation:
        {"qpos": np.ndarray (20D), "images": {cam_name: (H,W,3)}}
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
        setup_robot: bool = True,
    ):
        self.config = BiFlexivRizon4RTConfig(
            bi_mount_type=bi_mount_type,
            use_force=use_force,
            go_to_start=go_to_start,
            stiffness_ratio=stiffness_ratio,
            inner_control_hz=inner_control_hz,
            interpolate_cmds=interpolate_cmds,
            enable_tactile_sensors=enable_tactile_sensors,
            log_level=log_level,
        )
        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self) -> None:
        """Connect and initialize both arms."""
        logger.info("Connecting to BiFlexiv Rizon4 RT robot...")
        try:
            self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
            logger.info("BiFlexiv Rizon4 RT connected and ready")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    def get_qpos(self, obs: dict) -> np.ndarray:
        """Build 20D state vector from observation dict.

        Ordering matches the dataset feature order:
            [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]
        """
        left_tcp = [obs["left_tcp.x"], obs["left_tcp.y"], obs["left_tcp.z"]]
        left_tcp += [obs[f"left_tcp.r{i}"] for i in range(1, 7)]
        right_tcp = [obs["right_tcp.x"], obs["right_tcp.y"], obs["right_tcp.z"]]
        right_tcp += [obs[f"right_tcp.r{i}"] for i in range(1, 7)]
        return np.array(left_tcp + right_tcp + [obs["left_gripper.pos"], obs["right_gripper.pos"]], dtype=np.float32)

    def get_images(self, obs: dict) -> dict:
        """Extract camera images from observation dict."""
        images = {}
        for cam_name in _POLICY_CAMERAS:
            if cam_name in obs:
                images[cam_name] = obs[cam_name]
            else:
                logger.debug(f"Camera {cam_name} not found in observation")
        return images

    def get_observation(self) -> dict:
        """Get observation compatible with OpenPI format."""
        raw_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(raw_obs)
        obs["images"] = self.get_images(raw_obs)
        return obs

    def get_reward(self) -> float:
        return 0.0

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """Reset both arms to start positions and wait for completion."""
        if not fake:
            logger.info("Resetting BiFlexiv Rizon4 RT to start positions...")
            try:
                self.robot.reset_to_initial_position()
                # Block until the non-blocking RT trajectory actually finishes.
                # Phase 1: wait for is_moving to become True (RT thread picks up request).
                # Phase 2: wait for is_moving to become False (trajectory complete).
                t0 = time.time()
                while not self.robot.rt_moving:
                    if time.time() - t0 > 1.0:
                        logger.warning("RT trajectory never started, proceeding anyway")
                        break
                    time.sleep(0.001)
                while self.robot.rt_moving:
                    if time.time() - t0 > 15.0:
                        logger.warning("Reset trajectory timeout, proceeding anyway")
                        break
                    time.sleep(0.05)
                logger.info("BiFlexiv Rizon4 RT reset completed")
            except Exception as e:
                logger.error(f"Failed to reset: {e}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Execute 20D action on both arms (dm_env step semantics).

        Sends the action and returns a TimeStep including the resulting
        observation. The wrapper in env.py no longer uses this path —
        it calls send_action() and get_observation() independently so
        an outer loop (e.g. DecoupledRuntime's obs/action threads) can
        run them on different schedules. Kept for backward compatibility
        with code that expects dm_env semantics.

        Args:
            action: [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]
        """
        self.send_action(action)
        obs = self.get_observation()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def send_action(self, action: np.ndarray) -> None:
        """Send a 20D action to the robot. Does NOT read observations.

        Use this when an outer loop reads observations independently. By
        keeping send + read separate, the action thread isn't pinned to the
        camera frame rate — critical for DecoupledRuntime where the action
        thread must run at action_hz (e.g. 60 Hz) while observations run at
        camera FPS (~30 Hz).

        Args:
            action: [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]
        """
        import time as _time

        action_dict = self._build_action_dict(action)
        t0 = _time.time()
        try:
            self.robot.send_action(action_dict)
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise
        logger.debug(f"send_action: {(_time.time() - t0) * 1000:.2f}ms")

    def _build_action_dict(self, action: np.ndarray) -> dict:
        """Build the per-key action dict that BiFlexivRizon4RT.send_action expects."""
        action_dict = {}
        # Left TCP (0-8)
        action_dict["left_tcp.x"] = float(action[0])
        action_dict["left_tcp.y"] = float(action[1])
        action_dict["left_tcp.z"] = float(action[2])
        for i in range(6):
            action_dict[f"left_tcp.r{i + 1}"] = float(action[3 + i])
        # Right TCP (9-17)
        action_dict["right_tcp.x"] = float(action[9])
        action_dict["right_tcp.y"] = float(action[10])
        action_dict["right_tcp.z"] = float(action[11])
        for i in range(6):
            action_dict[f"right_tcp.r{i + 1}"] = float(action[12 + i])
        # Grippers (18, 19)
        action_dict["left_gripper.pos"] = float(np.clip(action[18], 0.0, 1.0))
        action_dict["right_gripper.pos"] = float(np.clip(action[19], 0.0, 1.0))
        return action_dict

    def disconnect(self) -> None:
        """Disconnect both arms, grippers, and cameras."""
        if self.robot.is_connected:
            logger.info("Disconnecting BiFlexiv Rizon4 RT...")
            try:
                self.robot.disconnect()
                time.sleep(1)
                logger.info("BiFlexiv Rizon4 RT disconnected")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
                if emergency_stop_flexiv_rt_robot(self.robot, logger):
                    logger.warning("Emergency stop fallback completed for BiFlexiv Rizon4 RT")
