#!/usr/bin/env python
"""Main script for Flexiv Rizon4 RT robot inference with OpenPI.

Uses the real-time (RT) driver (flexiv_rt) which runs a C++ RT thread at 1 kHz
for deterministic streaming Cartesian motion force control.

Only RT_CARTESIAN_MOTION_FORCE mode is supported (no joint impedance).
Action space: 10D [tcp.x, tcp.y, tcp.z, tcp.r1-r6, gripper.pos]

--robot_recipe picks the bench: a name resolves against
examples/flexiv_rizon4_rt/recipes/, a path loads any recipe YAML. It carries the
arm SN, start pose, cameras and the typed gripper block. See recipes/README.md —
the flat gripper_* knobs and the flare_gripper backend are gone upstream, and
this driver does no camera auto-discovery, so the wrist camera is pinned there.

Example usage:
    # Basic inference (non-RTC mode)
    python -m examples.flexiv_rizon4_rt.main \\
        --robot_recipe default \\
        --host 192.168.2.215 \\
        --port 8000

    # With RTC enabled
    python -m examples.flexiv_rizon4_rt.main \\
        --robot_recipe default \\
        --host 192.168.2.215 \\
        --port 8000 \\
        --rtc_enabled

    # Dry run (robot connected but actions not executed)
    python -m examples.flexiv_rizon4_rt.main \\
        --robot_recipe default \\
        --host 192.168.2.215 \\
        --port 8000 \\
        --dry_run
"""

from dataclasses import dataclass
import signal
import sys

from lerobot.utils.robot_utils import get_logger
from typing_extensions import override
import tyro
from xense_client import action_chunk_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.flexiv_rizon4_rt.env as _env
import examples.flexiv_rizon4_rt.recipe as _recipe

logger = get_logger("FlexivRizon4RTMain")


class DryRunEnvironmentWrapper(_environment.Environment):
    """Wrapper: intercept and print policy action, but not actually execute."""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 Episode {self._episode_count} - environment reset (dry run mode)")
        logger.info(f"{'='*80}\n")
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1

        actions = action.get("actions")
        if actions is not None:
            logger.info(f"\n{'─'*80}")
            logger.info(f"🎯 Step {self._step_count} - policy output action (10D Cartesian):")
            logger.info(f"{'─'*80}")

            labels = [
                "tcp.x",
                "tcp.y",
                "tcp.z",
                "tcp.r1",
                "tcp.r2",
                "tcp.r3",
                "tcp.r4",
                "tcp.r5",
                "tcp.r6",
                "gripper.pos",
            ]
            for i, (label, value) in enumerate(zip(labels, actions)):
                logger.info(f"  [{i:2d}] {label:12s}: {value:+.6f}")

            logger.info(f"{'─'*80}")
            logger.info("⚠️  DRY RUN mode: action intercepted, NOT executed on robot")
            logger.info(f"{'─'*80}\n")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


@dataclass
class Args:
    """Arguments for Flexiv Rizon4 RT inference.

    The bench comes from --robot_recipe; everything else here is run tuning,
    applied on top of the decoded recipe, so a flag always wins:
    dataclass default < recipe YAML < --args.* flag.
    """

    # Which physical bench to drive. A name resolves against
    # examples/flexiv_rizon4_rt/recipes/; a path loads any recipe YAML. The
    # recipe carries the arm SN, start pose, cameras and the typed gripper
    # block. Required: connecting to the wrong bench is not a safe default.
    robot_recipe: str

    # Policy server connection
    host: str = "localhost"
    port: int = 8000

    # Robot run tuning
    use_force: bool = False
    go_to_start: bool = False
    log_level: str = "INFO"

    # RT-specific settings
    stiffness_ratio: float = 0.2
    # None = use the recipe's start_position_degree.
    start_position_degree: list[float] | None = None
    zero_ft_sensor_on_connect: bool = True
    # inner_control_hz: how often the C++ RT callback (1 kHz) consumes a new
    #   Python command. Range [1, 1000]. Default=1000 (every 1 ms cycle).
    #   e.g. 500 → consume every 2 ms; 100 → every 10 ms.
    inner_control_hz: int = 1000
    # interpolate_cmds: smooth motion between sparse Python commands via linear interpolation.
    #   Only effective when inner_control_hz < 1000.
    interpolate_cmds: bool = True

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings
    action_horizon: int = 50
    runtime_hz: float = 20.0
    num_episodes: int = 1
    max_episode_steps: int = 100000

    # Dry run mode (robot connected but actions not executed)
    dry_run: bool = False

    # RTC config
    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 20
    execution_horizon: int = 50
    blend_steps: int = 3
    default_delay: int = 2


def main(args: Args) -> None:
    # Resolve the bench before connecting: WebsocketClientPolicy blocks until the
    # policy server answers, and a typo'd recipe name should not cost that wait.
    recipe_path = _recipe.resolve_recipe_path(args.robot_recipe)
    logger.info(f"Robot recipe: {recipe_path}")

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    metadata = ws_client_policy.get_server_metadata()
    logger.info(f"Server metadata: {metadata}")

    base_environment = _env.FlexivRizon4RTEnvironment(
        robot_recipe=args.robot_recipe,
        use_force=args.use_force,
        go_to_start=args.go_to_start,
        log_level=args.log_level,
        stiffness_ratio=args.stiffness_ratio,
        start_position_degree=args.start_position_degree,
        zero_ft_sensor_on_connect=args.zero_ft_sensor_on_connect,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
    )

    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 DRY RUN mode enabled")
        logger.info("   - Policy action output will be printed")
        logger.info("   - Action will NOT be sent to robot")
        logger.info("   - Robot will stay in initial position")
        logger.info("=" * 80 + "\n")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        logger.info("✅ Normal mode: actions will be executed on robot (RT Cartesian)")
        environment = base_environment

    if args.rtc_enabled:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=rtc_action_chunk_broker.RTCActionChunkBroker(
                    policy=ws_client_policy,
                    frequency_hz=args.runtime_hz,
                    action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
                    rtc_enabled=args.rtc_enabled,
                    execution_horizon=args.execution_horizon,
                    blend_steps=args.blend_steps,
                    default_delay=args.default_delay,
                )
            ),
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    def safe_disconnect():
        """Safe disconnect robot."""
        try:
            actual_env = environment
            if isinstance(environment, DryRunEnvironmentWrapper):
                actual_env = environment._wrapped_env

            if hasattr(actual_env, "_env") and hasattr(actual_env._env, "robot"):
                if actual_env._env.robot.is_connected:
                    logger.info("Safe disconnect RT robot...")
                    actual_env._env.disconnect()
                    logger.info("✓ RT robot disconnected")
                else:
                    logger.info("Robot not connected, no need to disconnect")
        except Exception as e:
            logger.warning(f"Error disconnecting robot: {e}")

    def signal_handler(sig, frame):
        logger.info("\n⚠️ Detected user interrupt (Ctrl+C)")
        safe_disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Detected user interrupt (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        safe_disconnect()


if __name__ == "__main__":
    tyro.cli(main)
