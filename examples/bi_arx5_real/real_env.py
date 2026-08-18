import collections
import time

import dm_env

# import lerobot bi_arx5 config and make_robot_from_config
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5ControlMode
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("BiARX5RealEnv")

# default reset position (joint angles + gripper position) - same as lerobot config
DEFAULT_RESET_POSITION = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.002]

# safety limits (0.9 times the raw limits as detection threshold)
_JOINT_POS_MIN_RAW = np.array([-3.14, -0.05, -0.2, -1.6, -1.57, -2.0])
_JOINT_POS_MAX_RAW = np.array([2.618, 3.5, 3.2, 1.55, 1.57, 2.0])
_SAFETY_FACTOR = 0.9

# calculate safety limits (using 0.9 times the range)
JOINT_POS_MIN = _JOINT_POS_MIN_RAW * _SAFETY_FACTOR
JOINT_POS_MAX = _JOINT_POS_MAX_RAW * _SAFETY_FACTOR
GRIPPER_MIN = -0.1
GRIPPER_MAX = 1.8


class BiARX5RealEnv:
    """
    Environment for real BiARX5 robot bi-manual manipulation
    Based on lerobot BiARX5 implementation (No ROS, direct ARX5 SDK)

    Action space:      [left_joint_1.pos, ..., left_joint_6.pos, left_gripper.pos,
                        right_joint_1.pos, ..., right_joint_6.pos, right_gripper.pos]  # 14 dimensions

    Observation space: {"qpos": Concat[left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)],
                        "qvel": similar structure (zeros for BiARX5),
                        "images": {"cam_high": (H,W,3), "cam_left_wrist": (H,W,3), "cam_right_wrist": (H,W,3)}}
    """

    def __init__(
        self,
        left_arm_port: str = "can1",
        right_arm_port: str = "can3",
        log_level: str = "INFO",
        use_multithreading: bool = True,
        enable_tactile_sensors: bool = False,
        reset_position: list[float] | None = None,
        setup_robot: bool = True,
        controller_dt: float = 0.002,  # low-level control frequency (seconds)
        preview_time: float = 0.02,  # preview time (seconds)
        control_mode: str = "teach_mode",
    ):
        self._reset_position = reset_position or DEFAULT_RESET_POSITION

        # convert control_mode string to enum
        control_mode_enum = BiARX5ControlMode(control_mode)

        # create BiARX5 config - directly use lerobot config
        self.config = BiARX5Config(
            id="bi_arx5_real_inference",
            left_arm_model="X5",
            left_arm_port=left_arm_port,
            right_arm_model="X5",
            right_arm_port=right_arm_port,
            log_level=log_level,
            use_multithreading=use_multithreading,
            enable_tactile_sensors=enable_tactile_sensors,
            inference_mode=True,  # inference mode, set appropriate preview_time
            controller_dt=controller_dt,  # pass in controller_dt from parameters
            preview_time=preview_time,  # pass in preview_time from parameters
            control_mode=control_mode_enum,
        )

        # create robot instance - use existing BiARX5 class
        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self):
        """connect and initialize robot (no ROS, direct ARX5 SDK)"""
        logger.info("Connecting to BiARX5 robot via ARX5 SDK...")
        try:
            self.robot.connect(calibrate=False, go_to_start=True)
            logger.info("BiARX5 robot connected and ready for inference")
        except Exception as e:
            logger.error(f"Failed to connect BiARX5 robot: {e}")
            raise

    def get_qpos(self, obs):
        """get joint positions - directly from BiARX5.get_observation()"""
        # organize as [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        left_joints = [obs[f"left_joint_{i+1}.pos"] for i in range(6)]
        left_gripper = [obs["left_gripper.pos"]]

        right_joints = [obs[f"right_joint_{i+1}.pos"] for i in range(6)]
        right_gripper = [obs["right_gripper.pos"]]

        return np.concatenate([left_joints, left_gripper, right_joints, right_gripper])

    def get_qvel(self, obs):
        """get joint velocities - BiARX5 SDK does not provide velocity feedback, return zero vector"""
        return np.zeros(14)  # 6+1 + 6+1 = 14

    def get_images(self, obs):
        """get camera images - directly from BiARX5.get_observation()"""
        images = {}

        # map your camera names to aloha_real expected names
        camera_mapping = {
            "head": "cam_high",  # head camera -> cam_high
            "left_wrist": "cam_left_wrist",  # left wrist camera -> cam_left_wrist
            "right_wrist": "cam_right_wrist",  # right wrist camera -> cam_right_wrist
        }
        if self.config.enable_tactile_sensors:
            camera_mapping["left_tactile_0"] = "left_tactile_0"
            camera_mapping["right_tactile_0"] = "right_tactile_0"

        for lerobot_name, openpi_name in camera_mapping.items():
            if lerobot_name in obs:
                images[openpi_name] = obs[lerobot_name]
            else:
                logger.warn(f"Camera {lerobot_name} not found in observation")

        return images

    def get_observation(self):
        """get complete observation - compatible with aloha_real format"""
        current_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(current_obs)
        obs["qvel"] = self.get_qvel(current_obs)
        obs["images"] = self.get_images(current_obs)
        # print(obs["images"].keys()) # display the keys of the robot observation images
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        """reset robot to initial position - use your smooth_go_start()"""
        if not fake:
            logger.info("Resetting BiARX5 to start position...")
            try:
                # use existing smooth movement method
                self.robot.smooth_go_start(duration=2.0)
                logger.info("BiARX5 reset completed")
            except Exception as e:
                logger.error(f"Failed to reset BiARX5: {e}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def _check_action_safety(self, action: np.ndarray) -> tuple[bool, str]:
        """check if action is within safety limits

        Args:
            action: 14-dimensional action array [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]

        Returns:
            (is_safe, error_message): whether action is safe, and error message
        """
        action = np.asarray(action)

        # check left arm joints (index 0-5)
        left_joints = action[0:6]
        for i, (val, min_val, max_val) in enumerate(zip(left_joints, JOINT_POS_MIN, JOINT_POS_MAX)):
            if val < min_val or val > max_val:
                return (
                    False,
                    f"Left joint {i+1} out of range: {val:.4f} not in [{min_val:.4f}, {max_val:.4f}]",
                )

        # check left gripper (index 6)
        left_gripper = action[6]
        if left_gripper < GRIPPER_MIN or left_gripper > GRIPPER_MAX:
            return (
                False,
                f"Left gripper out of range: {left_gripper:.4f} not in [{GRIPPER_MIN}, {GRIPPER_MAX}]",
            )

        # check right arm joints (index 7-12)
        right_joints = action[7:13]
        for i, (val, min_val, max_val) in enumerate(zip(right_joints, JOINT_POS_MIN, JOINT_POS_MAX)):
            if val < min_val or val > max_val:
                return (
                    False,
                    f"Right joint {i+1} out of range: {val:.4f} not in [{min_val:.4f}, {max_val:.4f}]",
                )

        # check right gripper (index 13)
        right_gripper = action[13]
        if right_gripper < GRIPPER_MIN or right_gripper > GRIPPER_MAX:
            return (
                False,
                f"Right gripper out of range: {right_gripper:.4f} not in [{GRIPPER_MIN}, {GRIPPER_MAX}]",
            )

        return True, ""

    def step(self, action):
        """execute action - convert format and call your send_action()"""
        # safety check
        is_safe, error_msg = self._check_action_safety(action)
        if not is_safe:
            logger.error(f"Action safety check failed: {error_msg}")
            raise ValueError(f"Unsafe action detected: {error_msg}")

        # ensure robot is in normal position control mode (not gravity compensation mode)
        if self.robot.is_gravity_compensation_mode():
            logger.info("Switching from gravity compensation to normal position control for action execution")
            self.robot.set_to_normal_position_control()
        # else:
        #     logger.info("Robot is already in normal position control mode")

        # convert OpenPI action array to your BiARX5 action dictionary format
        action_dict = {}

        # left arm action (first 7: 6 joints + gripper)
        for i in range(6):
            action_dict[f"left_joint_{i+1}.pos"] = float(action[i])
        action_dict["left_gripper.pos"] = float(action[6])

        # right arm action (last 7: 6 joints + gripper)
        for i in range(6):
            action_dict[f"right_joint_{i+1}.pos"] = float(action[7 + i])
        action_dict["right_gripper.pos"] = float(action[13])
        # for better gripper control to catch the cubes
        # if (
        #     action_dict["left_gripper.pos"] < 0.48
        #     and action_dict["left_gripper.pos"] > 0.38
        # ):
        #     action_dict["left_gripper.pos"] = 0.4
        # if (
        #     action_dict["right_gripper.pos"] < 0.45
        #     and action_dict["right_gripper.pos"] > 0.35
        # ):
        #     action_dict["right_gripper.pos"] = 0.4

        # print(
        #     "gripper_action",
        #     action_dict["left_gripper.pos"],
        #     action_dict["right_gripper.pos"],
        # )

        # send action to your BiARX5 robot
        try:
            self.robot.send_action(action_dict)
        except Exception as e:
            logger.error(f"Failed to send action to BiARX5: {e}")
            raise

        # wait for a control cycle (your config has controller_dt = 0.01, i.e. 100Hz)
        # time.sleep(self.config.controller_dt)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def disconnect(self):
        """disconnect robot connection"""
        if self.robot.is_connected:
            logger.info("Disconnecting BiARX5 robot...")
            try:
                self.robot.disconnect()
                time.sleep(1)
                logger.info("BiARX5 robot disconnected")
            except Exception as e:
                logger.warn(f"Error during BiARX5 disconnect: {e}")
