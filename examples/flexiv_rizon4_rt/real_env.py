"""Real environment for Flexiv Rizon4 RT robot.

This module wraps the lerobot FlexivRizon4RT (real-time) implementation for use with OpenPI.

This is the only Flexiv path left: lerobot removed the NRT driver (flexivrdk)
and its `flexiv_rizon4` package at d6d02f88, and the `flexiv_rizon4_real`
example that wrapped it went with it.

- flexiv_rt backend: C++ RT thread at 1 kHz
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
        robot_config: FlexivRizon4RTConfig,
        setup_robot: bool = True,
    ):
        """Wrap an already-decoded robot config.

        Args:
            robot_config: Built by ``recipe.load_robot_config`` — the recipe
                supplies the arm SN, start pose, cameras and the typed
                ``gripper:`` block (the flat ``gripper_*`` fields and the
                ``flare_gripper`` backend no longer exist), with the CLI's run
                tuning merged on top.
            setup_robot: Connect immediately.
        """
        self.config = robot_config

        # The 10D state/action vector this example speaks ends in gripper.pos,
        # and lerobot only emits that key when a gripper is configured — so a
        # recipe with no `gripper:` block would KeyError on the first
        # observation instead of degrading to 9D. Say so up front.
        if self.config.gripper is None:
            raise ValueError(
                "Recipe configures no gripper, but this example's state and action "
                "vectors are 10D ending in gripper.pos. Add a `gripper:` block."
            )

        # Every image the policy sees is a configured camera now: neither
        # gripper backend carries one, and this driver does no USB-hub
        # auto-discovery, so an empty `cameras:` means a state-only observation.
        if not self.config.cameras:
            logger.warn(
                "Recipe configures no cameras — the policy will receive state only. "
                "Pin at least `wrist_cam` in the recipe's `cameras:` block."
            )

        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self):
        """Connect and initialize robot."""
        logger.info(
            f"Connecting to Flexiv Rizon4 RT robot "
            f"(sn={self.config.robot_sn}, "
            f"gripper={self.config.gripper.type if self.config.gripper else None})..."
        )
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
        """Get camera images from observation.

        Every camera is a configured one now: neither gripper backend carries a
        camera, so the wrist and tactile feeds the Flare gripper used to inject
        are ordinary `cameras:` entries in the recipe.
        """
        images = {}
        camera_names = list(self.config.cameras.keys())

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
                        logger.warn("Reset trajectory timeout, proceeding anyway")
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
                logger.warn(f"Error during Flexiv Rizon4 RT disconnect: {e}")
