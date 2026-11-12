#!/usr/bin/env python
"""BiRebot B601 RT inference with optional real-time chunking (RTC).

python -m examples.bi_rebot_b601_rt.main --args.run pick-and-place-rtc --args.host <server>
python -m examples.bi_rebot_b601_rt.main --args.run dry-run --args.host <server>
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import threading

from lerobot.utils.robot_utils import get_logger
from xense_client import action_chunk_broker
from xense_client import paced_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy
from xense_client.runtime import decoupled_runtime
from xense_client.runtime import runtime as runtime_lib
from xense_client.runtime.agents.policy_agent import PolicyAgent

from examples.bi_rebot_b601_rt import recipe
from examples.bi_rebot_b601_rt.env import BiRebotB601RTEnvironment
import examples.run_config as run_config

logger = get_logger("BiRebotB601RTMain")
RUNS_DIR = Path(__file__).parent / "runs"


@dataclass
class Args:
    # Run YAML name or path; CLI flags override its settings.
    run: str | None = None
    # Bench name from recipes/, or an upstream teleop/record YAML path.
    robot_recipe: str | None = None
    # Policy server hostname/IP.
    host: str = "localhost"
    # Policy server WebSocket port.
    port: int = 8000
    # None uses the checkpoint config's default prompt.
    task: str | None = None
    # Move to the recorded start joint poses on connect and between episodes.
    go_to_start: bool = True
    # None preserves the recipe's return-to-zero setting on disconnect.
    reset_on_exit: bool | None = None
    # Enable tactile camera capture on both external grippers.
    enable_tactile: bool = False
    # Override both arms' Cartesian speed caps (m/s); None preserves the recipe.
    max_lin_speed: float | None = None
    # Override both arms' angular speed caps (rad/s); None preserves the recipe.
    max_ang_speed: float | None = None
    # Driver observation key for the head image.
    head_camera: str = "head"
    # Driver observation key for the left wrist image.
    left_wrist_camera: str = "left_left_wrist"
    # Driver observation key for the right wrist image.
    right_wrist_camera: str = "right_right_wrist"
    # Policy input image height in pixels.
    render_height: int = 224
    # Policy input image width in pixels.
    render_width: int = 224
    # Dataset recording rate is 30 Hz; the native controller interpolates at 100 Hz.
    runtime_hz: float = 30.0
    # Number of episodes to execute.
    num_episodes: int = 1
    # Per-episode step limit; 0 runs until interrupted.
    max_episode_steps: int = 1000000
    # Connects the robot and honors startup/exit resets, but sends no policy actions.
    dry_run: bool = False
    # Number of actions to execute per non-RTC inference call.
    action_horizon: int = 50
    # Enable asynchronous RTC inference and prefix guidance.
    rtc_enabled: bool = False
    # Request a new RTC chunk when the queue falls to this many actions.
    action_queue_size_to_get_new_actions: int = 20
    # RTC execution horizon sent to the policy server, in steps.
    execution_horizon: int = 50
    # Queue transition blending steps; 0 disables linear blending.
    blend_steps: int = 0
    # Initial RTC inference delay estimate, in steps.
    default_delay: int = 4
    # Extra steps added to the measured delay estimate.
    delay_margin: int = 2
    # 0 uses the synchronous runtime; positive values enable independent action pacing.
    action_hz: float = 0.0
    # Additional action buffer in decoupled mode; keep small for RTC.
    paced_queue_size: int = 1


def validate_args(args: Args) -> None:
    if not math.isfinite(args.runtime_hz) or args.runtime_hz <= 0:
        raise ValueError("runtime_hz must be finite and positive")
    if not math.isfinite(args.action_hz) or args.action_hz < 0:
        raise ValueError("action_hz must be finite and nonnegative")
    for name in (
        "num_episodes",
        "action_horizon",
        "execution_horizon",
        "paced_queue_size",
        "render_height",
        "render_width",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "max_episode_steps",
        "action_queue_size_to_get_new_actions",
        "blend_steps",
        "default_delay",
        "delay_margin",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")


def make_broker(args: Args, policy):
    """The RTC queue contains absolute 20D actions, including both grippers."""
    if args.rtc_enabled:
        return rtc_action_chunk_broker.RTCActionChunkBroker(
            policy=policy,
            frequency_hz=args.action_hz or args.runtime_hz,
            action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
            rtc_enabled=True,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            default_delay=args.default_delay,
            delay_margin=args.delay_margin,
            dry_run=args.dry_run,
            # Server applies the Rebot delta mask (9 TCP, gripper, 9 TCP, gripper)
            # and normalization to the absolute prefix using the current state.
        )
    return action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=args.action_horizon)


def main(args: Args) -> None:
    validate_args(args)
    logger.info(run_config.describe(args, Args, RUNS_DIR))
    if args.robot_recipe is None:
        raise SystemExit("Select a bench with --args.robot-recipe <name/path> or --args.run <name>.")
    config = recipe.load_robot_config(
        args.robot_recipe,
        enable_tactile=args.enable_tactile,
        max_lin_speed=args.max_lin_speed,
        max_ang_speed=args.max_ang_speed,
        reset_on_exit=args.reset_on_exit,
    )
    recipe.validate_robot_config(config, go_to_start=args.go_to_start)
    logger.info(f"Robot recipe: {recipe.resolve_recipe_path(args.robot_recipe)}")
    ws = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    environment = None
    broker = None
    paced = None
    previous_handler = None
    try:
        logger.info(f"Server metadata: {ws.get_server_metadata()}")
        environment = BiRebotB601RTEnvironment(
            config,
            go_to_start=args.go_to_start,
            render_height=args.render_height,
            render_width=args.render_width,
            camera_map={
                "head": args.head_camera,
                "left_wrist": args.left_wrist_camera,
                "right_wrist": args.right_wrist_camera,
            },
            prompt=args.task,
            dry_run=args.dry_run,
        )
        runtime = None
        stopping = threading.Event()

        def signal_handler(sig, frame):
            if stopping.is_set():
                logger.warning("Second Ctrl+C: forcing exit before cleanup completes")
                os._exit(1)
            stopping.set()
            environment.request_stop()
            if runtime is not None:
                runtime.request_stop()
            logger.info("Stopping inference; press Ctrl+C again to force exit")

        previous_handler = signal.signal(signal.SIGINT, signal_handler)
        for episode in range(args.num_episodes):
            if stopping.is_set():
                break
            broker = make_broker(args, ws)
            if args.action_hz > 0:
                paced = paced_broker.PacedBroker(
                    inner=broker, queue_size=args.paced_queue_size, target_hz=args.action_hz
                )
                runtime = decoupled_runtime.DecoupledRuntime(
                    environment=environment,
                    broker=paced,
                    subscribers=[],
                    obs_hz=args.runtime_hz,
                    action_hz=args.action_hz,
                    num_episodes=1,
                    max_episode_steps=args.max_episode_steps,
                )
            else:
                runtime = runtime_lib.Runtime(
                    environment=environment,
                    agent=PolicyAgent(broker),
                    subscribers=[],
                    max_hz=args.runtime_hz,
                    num_episodes=1,
                    max_episode_steps=args.max_episode_steps,
                )
            runtime.run()
            environment.raise_if_failed()
            if stopping.is_set():
                break
            if episode + 1 < args.num_episodes:
                # Join the old inference worker before creating the next queue:
                # an in-flight reply must never repopulate a reset RTC broker.
                if paced is not None:
                    paced.stop()
                if args.rtc_enabled:
                    broker.stop()
                broker = None
                paced = None
    finally:
        try:
            if environment is not None:
                environment.request_stop()
            # The websocket client has no public close API. Closing the socket
            # releases any blocked recv before joining RTC's inference thread.
            try:
                ws._ws.close()
            finally:
                if paced is not None:
                    paced.stop()
                if args.rtc_enabled and broker is not None:
                    broker.stop()
        finally:
            try:
                if environment is not None:
                    environment.disconnect()
            finally:
                if previous_handler is not None:
                    signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main(run_config.cli(main, Args, RUNS_DIR))
