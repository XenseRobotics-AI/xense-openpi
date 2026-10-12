"""Real environment for the bimanual Elite CS66 RT robot.

Wraps lerobot ``BiEliteCS66RT`` for use with OpenPI inference. The bench it
drives comes from a recipe YAML (see ``recipe.py`` and ``recipes/README.md``);
run tuning comes from the CLI.

State/action format (20D, per-side-grouped — the driver's feature order and the
order ``openpi.policies.bi_elite_policy`` trains on)::

    [left_tcp.x/y/z/r1-r6 (0-8), left_gripper.pos (9),
     right_tcp.x/y/z/r1-r6 (10-18), right_gripper.pos (19)]

This is NOT the ``bi_flexiv_rizon4_rt`` example's order, which groups both TCPs
first and both grippers last. TCP poses are in the driver's gravity-aligned
world frame, both when read and when sent.
"""

import time

from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTControlMode
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("BiEliteCS66RTRealEnv")

# Policy-facing camera names (must match bi_elite_policy.BiEliteInputs.EXPECTED_CAMERAS)
POLICY_CAMERAS = ("head", "left_wrist", "right_wrist")

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

# Upper bound on how long a reset to the start pose may take.
_RESET_TIMEOUT_S = 15.0


class BiEliteCS66RTRealEnv:
    """Native 20D access to a BiElite CS66 RT: state read, action write, reset."""

    def __init__(
        self,
        robot_config: BiEliteCS66RTConfig,
        *,
        go_to_start: bool = True,
        setup_robot: bool = True,
    ):
        """Wrap an already-decoded robot config.

        Args:
            robot_config: Built by ``recipe.load_robot_config`` — the recipe
                supplies the bench hardware (controller IPs, start/home poses,
                mount rotations, head camera, gripper block), with the CLI's run
                tuning merged on top.
            go_to_start: MoveJ both arms to the recipe's start pose on connect.
            setup_robot: Connect immediately.
        """
        self.config = robot_config
        self._go_to_start = go_to_start

        # The policy was trained on Cartesian TCP + gripper only; any other
        # schema changes the 20D vector out from under it.
        if self.config.control_mode != BiEliteCS66RTControlMode.CARTESIAN_SERVO:
            raise ValueError(f"control_mode must be cartesian_servo, got {self.config.control_mode}")
        if not self.config.observe_tcp:
            raise ValueError("observe_tcp must be true: the 20D state is built from TCP poses")
        # lerobot only emits {side}_gripper.pos for a side that has a gripper, so
        # dropping one would KeyError on the first observation. Say so up front.
        missing = [
            side
            for side, cfg in (("left", self.config.left_gripper), ("right", self.config.right_gripper))
            if cfg is None
        ]
        if missing:
            raise ValueError(
                f"Recipe configures no gripper on: {', '.join(missing)}. This example's state and "
                "action vectors include left/right_gripper.pos, so both sides need one."
            )

        self.robot = make_robot_from_config(self.config)
        if setup_robot:
            self.setup_robot()

    def setup_robot(self) -> None:
        """Connect and initialize both arms."""
        logger.info(
            f"Connecting to BiElite CS66 RT robot "
            f"(left={self.config.left_robot_ip}, right={self.config.right_robot_ip}, "
            f"gripper={self.config.gripper.type if self.config.gripper else None})..."
        )
        try:
            self.robot.connect(calibrate=False, go_to_start=self._go_to_start)
            logger.info("BiElite CS66 RT connected and ready")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    def get_observation(self) -> dict:
        """Read both arms, both grippers and the policy cameras.

        Returns:
            {"qpos": np.ndarray (20D), "images": {cam_name: (H, W, 3)}}
        """
        raw_obs = self.robot.get_observation()
        qpos = np.array([raw_obs[key] for key in STATE_KEYS], dtype=np.float32)
        images = {cam: raw_obs[cam] for cam in POLICY_CAMERAS if cam in raw_obs}
        return {"qpos": qpos, "images": images}

    def reset(self) -> None:
        """Return both arms to the start pose and wait for the motion to finish."""
        logger.info("Resetting BiElite CS66 RT to start positions...")
        self.robot.reset_to_initial_position()
        # With the background servo loop the reset is asynchronous: wait for the
        # reset to register, then for it to complete.
        t0 = time.time()
        while not self.robot.rt_moving:
            if time.time() - t0 > 1.0:
                logger.warn("Reset motion never started, proceeding anyway")
                break
            time.sleep(0.001)
        while self.robot.rt_moving:
            if time.time() - t0 > _RESET_TIMEOUT_S:
                logger.warn("Reset motion timeout, proceeding anyway")
                break
            time.sleep(0.05)
        logger.info("BiElite CS66 RT reset completed")

    def send_action(self, action: np.ndarray) -> None:
        """Send a 20D per-side-grouped action. Does NOT read observations."""
        action = np.asarray(action, dtype=np.float64).copy()
        action[list(_GRIPPER_INDICES)] = np.clip(action[list(_GRIPPER_INDICES)], 0.0, 1.0)
        t0 = time.time()
        self.robot.send_action({key: float(value) for key, value in zip(STATE_KEYS, action, strict=True)})
        logger.debug(f"send_action: {(time.time() - t0) * 1000:.2f}ms")

    def disconnect(self) -> None:
        """Disconnect both arms, grippers, and cameras."""
        if self.robot.is_connected:
            logger.info("Disconnecting BiElite CS66 RT...")
            try:
                self.robot.disconnect()
                logger.info("BiElite CS66 RT disconnected")
            except Exception as e:
                logger.warn(f"Error during disconnect: {e}")
