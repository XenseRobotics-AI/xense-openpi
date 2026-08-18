"""BiFlexiv Rizon4 RT real environment for XTac-UMI inference, wrist cameras only.

Wraps the lerobot ``BiFlexivRizon4RT`` driver. The bench comes from a recipe YAML
— this example reuses ``examples/bi_flexiv_rizon4_rt/recipes/`` (``forward-04`` is
the taccap-gripper bench), so there is one set of bench definitions in the repo.

State/action layout (20D, XTac-UMI per-side-grouped order)::

    [left_tcp.x/y/z/r1-r6 (0-8), left_gripper.pos (9),
     right_tcp.x/y/z/r1-r6 (10-18), right_gripper.pos (19)]

This is the recording rig's feature order (``BiTaccapGripper.observation_features``
in xense-taccap-lerobot), NOT the ``bi_flexiv_rizon4_rt`` example's order, which
groups both TCPs first and both grippers last. The driver hands us a dict of
named keys, so the order is ours to choose and we choose the policy's — the
alternative, regrouping dims somewhere downstream, is a silent off-by-nine
waiting to happen.

Differences from ``examples/bi_flexiv_rizon4_rt/real_env.py``:
    - Only the two wrist cameras are connected. XTac-UMI data has no third-person
      view, so the head camera the recipe pins is dropped before the robot is
      constructed.
    - Force/wrench readings are off: XTac-UMI checkpoints consume the 20D
      pose/gripper space only, and enabling force would change the state layout.
"""

import collections
import time

from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("XtacUmiBiFlexivRizon4RTRealEnv")

_WRIST_CAMERAS = ("left_wrist", "right_wrist")

# The 20D vector, in order, as driver observation/action keys. One list drives
# both the state read and the action write so the two can't drift apart.
STATE_KEYS: tuple[str, ...] = (
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    *(f"left_tcp.r{i}" for i in range(1, 7)),
    "left_gripper.pos",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    *(f"right_tcp.r{i}" for i in range(1, 7)),
    "right_gripper.pos",
)
_GRIPPER_INDICES = (STATE_KEYS.index("left_gripper.pos"), STATE_KEYS.index("right_gripper.pos"))


class XtacUmiBiFlexivRizon4RTRealEnv:
    """Native 20D XTac-UMI-ordered access to a BiFlexiv Rizon4 RT, wrist cameras only.

    This class intentionally does not inherit the bi_flexiv example environment:
    that one emits the BiFlexiv dim order and carries a dm_env ``step()`` this
    example has no use for.
    """

    def __init__(
        self,
        robot_config: BiFlexivRizon4RTConfig,
        setup_robot: bool = True,
    ) -> None:
        """Wrap an already-decoded robot config.

        Args:
            robot_config: Built by ``examples.bi_flexiv_rizon4_rt.recipe.load_robot_config``
                — the recipe supplies bench hardware (arm SNs, start/home poses,
                gripper block), with the CLI's run tuning merged on top.
            setup_robot: Connect immediately.
        """
        if robot_config.use_force:
            raise ValueError("XTac-UMI inference uses the 20D pose/gripper space; use_force must be False")

        missing = [
            side
            for side, cfg in (("left", robot_config.left_gripper), ("right", robot_config.right_gripper))
            if cfg is None
        ]
        if missing:
            raise ValueError(
                f"Recipe configures no gripper on: {', '.join(missing)}. The 20D state and action "
                "vectors end each side with its gripper.pos, so both sides need one."
            )

        self.config = robot_config
        # The recipe pins a head RealSense for the teleop/record benches. Drop it
        # before the robot object constructs or connects its cameras — XTac-UMI
        # checkpoints mask the base_0_rgb slot out, so the frames would be
        # captured, resized and thrown away.
        self.config.cameras.pop("head", None)
        # taccap grippers auto-discover their own wrist + tactile cameras at
        # connect, so a recipe that lists no wrist camera is normal and fine; one
        # that lists something else is a bench mismatch worth stopping for.
        extra = set(self.config.cameras) - set(_WRIST_CAMERAS)
        if extra:
            raise RuntimeError(
                f"Expected only wrist cameras after dropping the head camera, got {tuple(sorted(extra))}"
            )

        self.robot = make_robot_from_config(self.config)
        if setup_robot:
            self.setup_robot()

    def setup_robot(self) -> None:
        logger.info(
            f"Connecting BiFlexiv Rizon4 RT (left={self.config.left_robot_sn}, "
            f"right={self.config.right_robot_sn}, "
            f"gripper={self.config.gripper.type if self.config.gripper else None}), "
            "wrist cameras only"
        )
        self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
        logger.info("BiFlexiv Rizon4 RT connected; head camera is disabled")

    @staticmethod
    def get_qpos(obs: dict) -> np.ndarray:
        """Build the 20D state vector in XTac-UMI per-side-grouped order."""
        return np.asarray([obs[key] for key in STATE_KEYS], dtype=np.float32)

    @staticmethod
    def get_images(obs: dict) -> dict:
        missing = set(_WRIST_CAMERAS) - set(obs)
        if missing:
            raise RuntimeError(f"Missing wrist camera observations: {tuple(sorted(missing))}")
        return {camera: obs[camera] for camera in _WRIST_CAMERAS}

    def get_observation(self) -> dict:
        raw_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(raw_obs)
        obs["images"] = self.get_images(raw_obs)
        return obs

    def reset(self) -> None:
        """Reset both arms to their start pose and block until the move finishes."""
        logger.info("Resetting BiFlexiv Rizon4 RT to its configured start pose")
        self.robot.reset_to_initial_position()

        # reset_to_initial_position() drives a non-blocking RT trajectory. Wait for
        # it to start, then to finish. Unlike bi_flexiv_rizon4_rt (which logs and
        # proceeds on a slow reset), a 15 s overrun raises: starting inference from
        # an unknown mid-reset pose is more dangerous than aborting the episode.
        start = time.monotonic()
        while not self.robot.rt_moving:
            if time.monotonic() - start > 1.0:
                logger.warning("RT reset trajectory did not report a start within 1 second")
                return
            time.sleep(0.001)
        while self.robot.rt_moving:
            if time.monotonic() - start > 15.0:
                raise TimeoutError("BiFlexiv RT reset trajectory exceeded 15 seconds")
            time.sleep(0.05)

    @staticmethod
    def build_action_dict(action: np.ndarray) -> dict[str, float]:
        """Build the per-key action dict ``BiFlexivRizon4RT.send_action`` expects.

        Grippers are clipped to [0, 1]; TCP pose values pass through verbatim
        (same convention as bi_flexiv_rizon4_rt).
        """
        values = np.asarray(action)
        if values.shape != (len(STATE_KEYS),):
            raise ValueError(f"Expected a 20D action, got {values.shape}")
        return {
            key: float(np.clip(value, 0.0, 1.0)) if index in _GRIPPER_INDICES else float(value)
            for index, (key, value) in enumerate(zip(STATE_KEYS, values, strict=True))
        }

    def send_action(self, action: np.ndarray) -> None:
        """Send a 20D action to the robot. Does NOT read observations.

        Kept separate from get_observation() so the outer runtime loop owns obs
        scheduling — same contract as bi_flexiv_rizon4_rt.
        """
        self.robot.send_action(self.build_action_dict(action))

    def disconnect(self) -> None:
        """Disconnect both arms, grippers, and cameras.

        The driver's own disconnect() performs the safe RT-thread stop, home
        motion and resource cleanup, so no emergency-stop fallback is needed.
        """
        if self.robot.is_connected:
            logger.info("Disconnecting BiFlexiv Rizon4 RT")
            self.robot.disconnect()
