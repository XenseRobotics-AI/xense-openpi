"""Run a Tianji/Wuji checkpoint on the robot host, without RTC.

Ported from TacXense's real-robot-verified ``examples/bi_tianji_wuji`` client; only
the expected training config differs. See README.md for setup.
"""

# Hardware imports stay lazy so --help works offline.

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time

import numpy as np

from examples import run_config
from examples.bi_tianji_wuji.schema import ACTION_KEYS
from examples.bi_tianji_wuji.schema import validate_actions

logger = logging.getLogger(__name__)
RUNS_DIR = Path(__file__).parent / "runs"
TRAIN_CONFIG = "pi05_base_bi_tianji_wuji_block_sort_0925_h100"


@dataclass
class Args:
    run: str | None = None
    robot_recipe: str | None = None
    host: str = "localhost"
    port: int = 8000
    expected_config: str = TRAIN_CONFIG
    task: str = "Sort and place the blocks on the tabletop."
    dry_run: bool = False
    runtime_hz: float = 30.0
    action_horizon: int = 50
    num_episodes: int = 1
    max_episode_steps: int = 1000000
    reset_on_start: bool = True
    reset_timeout_s: float = 15.0
    inference_timeout_s: float = 10.0
    render_size: int = 224
    log_file: str = "logs/bi_tianji_wuji.log"

    def validate(self) -> None:
        if not self.robot_recipe:
            raise ValueError("Pass --args.robot-recipe or a run YAML containing robot_recipe")
        for name in ("runtime_hz", "reset_timeout_s", "inference_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("action_horizon", "num_episodes", "max_episode_steps", "render_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_horizon > 50:
            raise ValueError("This checkpoint predicts 50 steps; action_horizon must be <= 50")


def run_episode(environment, policy, args: Args) -> None:
    """Synchronous chunk execution; never burst actions to catch up after a stall."""
    if args.reset_on_start:
        environment.reset()
    policy.reset()
    step = 0
    period = 1.0 / args.runtime_hz
    while step < args.max_episode_steps:
        environment.hold()
        observation = environment.get_observation(args.task)
        started = time.monotonic()
        result = policy.infer(observation)
        actions = validate_actions(result["actions"])
        if len(actions) < args.action_horizon:
            raise ValueError(f"Server returned {len(actions)} steps, requested {args.action_horizon}")
        logger.info("Chunk: inference_ms=%.1f shape=%s", (time.monotonic() - started) * 1000, actions.shape)
        count = min(args.action_horizon, args.max_episode_steps - step)
        for index in range(count):
            started = time.monotonic()
            if index:
                observation = environment.get_observation(args.task)
            environment.apply_action(actions[index])
            step += 1
            logger.info(
                "Step %d obs=%s action=%s dry_run=%s",
                step,
                np.array2string(observation["state"], separator=",", max_line_width=10000),
                np.array2string(actions[index], separator=",", max_line_width=10000),
                args.dry_run,
            )
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    environment.hold()


def main(args: Args) -> None:
    """Use --args.run dry-run to check the real camera/state-to-policy path."""
    args.validate()
    log_path = Path(args.log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    logger.info(run_config.describe(args, Args, RUNS_DIR))
    logger.info("Settings: %s", args)
    logger.info("State/action field order: %s", ACTION_KEYS)

    # Lazy imports keep --help and schema tests usable without hardware SDKs.
    from lerobot.robots.tianji_arm_wuji import TianjiArmWuji
    from xense_client.websocket_client_policy import WebsocketClientPolicy

    from examples.bi_tianji_wuji.env import TianjiWujiEnvironment
    from examples.bi_tianji_wuji.recipe import load_robot_config

    config = load_robot_config(args.robot_recipe)
    if 1.0 / args.runtime_hz >= config.command_timeout_s:
        raise ValueError("runtime_hz is too low for the robot command_timeout_s")
    policy = WebsocketClientPolicy(host=args.host, port=args.port, request_timeout_s=args.inference_timeout_s)
    environment = None
    try:
        metadata = policy.get_server_metadata()
        logger.info("Server metadata: %s", metadata)
        if metadata.get("config") != args.expected_config:
            raise ValueError(
                f"Wrong policy server: expected config {args.expected_config!r}, got {metadata.get('config')!r}"
            )
        environment = TianjiWujiEnvironment(
            TianjiArmWuji(config),
            dry_run=args.dry_run,
            render_size=args.render_size,
            reset_timeout_s=args.reset_timeout_s,
        )
        if args.dry_run:
            logger.info(
                "Dry-run: suppress policy/reset actions and shutdown homing; driver connect still enables position holding"
            )
        environment.connect()
        for episode in range(args.num_episodes):
            logger.info("Episode %d", episode + 1)
            run_episode(environment, policy, args)
    except KeyboardInterrupt:
        logger.info("Interrupted; stopping without automatic homing")
    except Exception:
        logger.exception("Deployment failed; stopping without automatic homing")
        raise
    finally:
        if environment is not None:
            try:
                environment.hold()
            except Exception:
                logger.exception("Failed to hold arms during shutdown")
            try:
                environment.disconnect()
            except Exception:
                logger.exception("Failed to disconnect robot")
        policy.disconnect()


if __name__ == "__main__":
    main(run_config.cli(main, Args, RUNS_DIR))
