"""Real environment for Flexiv Rizon4 RT robot.

This module wraps the lerobot FlexivRizon4RT (real-time) implementation for use with OpenPI.

Key differences from flexiv_rizon4_real:
- Uses flexiv_rt backend (C++ RT thread at 1 kHz) instead of NRT flexivrdk
- Only supports RT_CARTESIAN_MOTION_FORCE mode (no joint impedance)
- Action space: always 10D [x, y, z, r1-r6, gripper]
- reset_to_initial_position() is non-blocking (RT trajectory)
"""

import collections
import time

import dm_env
from lerobot.robots.flexiv_rizon4_rt.config_flexiv_rizon4_rt import FlexivRizon4RTConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("FlexivRizon4RTRealEnv")

# Constants for Flexiv Rizon4 RT
CARTESIAN_STATE_DIM = 10  # [x, y, z, r1-r6, gripper] = 10D


class FlexivRizon4RTRealEnv:
    """Environment for real Flexiv Rizon4 robot with RT Cartesian control.

    Uses FlexivRizon4RT (1 kHz C++ RT thread via flexiv_rt) for deterministic
    streaming Cartesian motion force control.

    Action/State space (always 10D):
        [tcp.x, tcp.y, tcp.z, tcp.r1, tcp.r2, tcp.r3, tcp.r4, tcp.r5, tcp.r6, gripper.pos]

        Where tcp.r1-tcp.r6 is the 6D rotation representation (first two columns of rotation matrix):
        - [tcp.r1, tcp.r2, tcp.r3]: First column of rotation matrix
        - [tcp.r4, tcp.r5, tcp.r6]: Second column of rotation matrix

    Observation space:
        {"qpos": np.ndarray (10D),
         "images": {"wrist_cam": (H,W,3), "left_tactile": (H,W,3), "right_tactile": (H,W,3), ...}}
    """

    def __init__(
        self,
        robot_sn: str = "Rizon4-063423",
        use_gripper: bool = True,
        use_force: bool = False,
        go_to_start: bool = True,
        log_level: str = "INFO",
        setup_robot: bool = True,
        # Gripper settings
        gripper_type: str = "flare_gripper",
        gripper_mac_addr: str = "e2b26adbb104",
        gripper_cam_size: tuple[int, int] = (640, 480),
        gripper_rectify_size: tuple[int, int] = (400, 700),
        gripper_max_pos: float = 85.0,
        # RT-specific settings
        stiffness_ratio: float = 0.2,
        start_position_degree: list[float] | None = None,
        zero_ft_sensor_on_connect: bool = True,
        inner_control_hz: int = 1000,
        interpolate_cmds: bool = True,
        # External cameras (scene cameras)
        cameras: dict | None = None,
    ):
        config_kwargs = {
            "robot_sn": robot_sn,
            "use_gripper": use_gripper,
            "use_force": use_force,
            "go_to_start": go_to_start,
            "log_level": log_level,
            "gripper_type": gripper_type,
            "gripper_mac_addr": gripper_mac_addr,
            "gripper_cam_size": gripper_cam_size,
            "gripper_rectify_size": gripper_rectify_size,
            "gripper_max_pos": gripper_max_pos,
            "stiffness_ratio": stiffness_ratio,
            "zero_ft_sensor_on_connect": zero_ft_sensor_on_connect,
            "inner_control_hz": inner_control_hz,
            "interpolate_cmds": interpolate_cmds,
            "cameras": cameras or {},
        }
        if start_position_degree is not None:
            config_kwargs["start_position_degree"] = start_position_degree

        self.config = FlexivRizon4RTConfig(**config_kwargs)

        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self):
        """Connect and initialize robot."""
        logger.info("Connecting to Flexiv Rizon4 RT robot...")
        try:
            self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
            logger.info("Flexiv Rizon4 RT robot connected and ready for inference")
        except Exception as e:
            logger.error(f"Failed to connect Flexiv Rizon4 RT robot: {e}")
            raise

    def get_qpos(self, obs: dict) -> np.ndarray:
        """Get Cartesian state from observation.

        Returns:
            [tcp.x, tcp.y, tcp.z, tcp.r1, ..., tcp.r6, gripper.pos] (10D)
        """
        position = [obs["tcp.x"], obs["tcp.y"], obs["tcp.z"]]
        rotation = [obs[f"tcp.r{i + 1}"] for i in range(6)]
        gripper = [obs["gripper.pos"]]
        return np.array(position + rotation + gripper, dtype=np.float32)

    def get_images(self, obs: dict) -> dict:
        """Get camera images from observation."""
        images = {}
        camera_names = ["wrist_cam", "left_tactile", "right_tactile"]
        camera_names.extend(self.config.cameras.keys())

        for cam_name in camera_names:
            if cam_name in obs:
                images[cam_name] = obs[cam_name]
            else:
                logger.debug(f"Camera {cam_name} not found in observation")

        return images

    def get_observation(self) -> dict:
        """Get complete observation compatible with OpenPI format."""
        current_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(current_obs)
        obs["images"] = self.get_images(current_obs)
        return obs

    def get_reward(self) -> float:
        return 0.0

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """Reset robot to initial position.

        The RT driver's reset_to_initial_position() starts a non-blocking RT
        trajectory. We wait until the trajectory is complete before returning.
        """
        if not fake:
            logger.info("Resetting Flexiv Rizon4 RT to start position...")
            try:
                self.robot.reset_to_initial_position()
                # Wait for the RT trajectory to complete
                timeout = 10.0
                start_time = time.time()
                while self.robot.rt_moving:
                    if time.time() - start_time > timeout:
                        logger.warning("Reset trajectory timeout, proceeding anyway")
                        break
                    time.sleep(0.05)
                logger.info("Flexiv Rizon4 RT reset completed")
            except Exception as e:
                logger.error(f"Failed to reset Flexiv Rizon4 RT: {e}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Execute Cartesian action on the RT robot.

        Args:
            action: [tcp.x, tcp.y, tcp.z, tcp.r1, ..., tcp.r6, gripper.pos] (10D)
        """
        action_dict = {
            "tcp.x": float(action[0]),
            "tcp.y": float(action[1]),
            "tcp.z": float(action[2]),
        }
        for i in range(6):
            action_dict[f"tcp.r{i + 1}"] = float(action[3 + i])

        gripper_pos = float(action[9])
        action_dict["gripper.pos"] = float(np.clip(gripper_pos, 0.0, 1.0))

        try:
            self.robot.send_action(action_dict)
        except Exception as e:
            logger.error(f"Failed to send action to Flexiv Rizon4 RT: {e}")
            raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def disconnect(self):
        """Disconnect robot connection."""
        if self.robot.is_connected:
            logger.info("Disconnecting Flexiv Rizon4 RT robot...")
            try:
                self.robot.disconnect()
                time.sleep(1)
                logger.info("Flexiv Rizon4 RT robot disconnected")
            except Exception as e:
                logger.warning(f"Error during Flexiv Rizon4 RT disconnect: {e}")
