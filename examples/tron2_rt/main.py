#!/usr/bin/env python
"""Main script for LimX TRON2 dual-arm robot inference with OpenPI.

Drives the lerobot ``Tron2RT`` driver (native 300 Hz joint publisher with
Cartesian waypoint streaming) against a policy served from e.g.
``configs/_examples/pi05_base_tron2rt_pnp_0918.yaml``. The bench is described by
``Tron2RTConfig`` defaults (robot IP, Bridge camera host, start joints, TacCap
grippers); the flags below override the few knobs that change between runs.

--args.run picks a run YAML from runs/, which presets any of the flags below.
Flags still override the file. See examples/run_config.py.

Example usage:
    # Serve the policy (GPU machine)
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config=pi05_base_tron2rt_pnp_0918 --policy.dir=<checkpoint_dir>

    # Basic inference
    python -m examples.tron2_rt.main --args.host 192.168.2.100 --args.port 8000

    # Dry run (robot connected but actions not sent)
    python -m examples.tron2_rt.main --args.run dry-run --args.host 192.168.2.100

    # Override the prompt sent to the server
    python -m examples.tron2_rt.main --args.host 192.168.2.100 --args.prompt "Put the block into the box"
"""

from dataclasses import dataclass
import os
import pathlib
import signal
import threading
from typing import override

from lerobot.grippers import TaccapFollowerConfig
from lerobot.robots.tron2_rt.config_tron2_rt import Tron2RTConfig
from lerobot.utils.robot_utils import get_logger
from xense_client import action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.run_config as _run_config
import examples.tron2_rt.env as _env

logger = get_logger("Tron2RTMain")

# Run YAMLs shipped with this example; --args.run resolves bare names here.
RUNS_DIR = pathlib.Path(__file__).parent / "runs"


class DryRunEnvironmentWrapper(_environment.Environment):
    """Intercepts policy actions and prints them without executing on robot."""

    def __init__(self, wrapped_env: _env.Tron2RTEnvironment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Episode {self._episode_count} - reset (dry run)")
        logger.info(f"{'=' * 80}\n")
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
        if actions is None:
            return
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Step {self._step_count} - policy action (20D Cartesian, TCP frame):")
        logger.info(f"{'─' * 80}")
        for i, (label, value) in enumerate(zip(_env._ACTION_LABELS, actions)):
            logger.info(f"  [{i:2d}] {label:8s}: {value:+.6f}")
        logger.info("DRY RUN: action NOT sent to robot")
        logger.info(f"{'─' * 80}\n")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


class PromptEnvironmentWrapper(_environment.Environment):
    """Attaches a fixed language prompt to every observation."""

    def __init__(self, wrapped_env: _environment.Environment, prompt: str):
        self._wrapped_env = wrapped_env
        self._prompt = prompt

    @override
    def reset(self) -> None:
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return {**self._wrapped_env.get_observation(), "prompt": self._prompt}

    @override
    def apply_action(self, action: dict) -> None:
        self._wrapped_env.apply_action(action)


@dataclass
class Args:
    """Arguments for TRON2 RT inference.

    Any of these can be preset in a run YAML under runs/ and selected with
    --args.run. Flags still win over the file; see examples/run_config.py.
    """

    # Which run YAML to take the settings below from. A name resolves against
    # examples/tron2_rt/runs/; a path loads any YAML.
    run: str | None = None

    # Policy server
    host: str = "localhost"
    port: int = 8000
    # Language prompt sent with each observation. None = use the server's
    # default_prompt (the train config sets "Put the block into the box").
    prompt: str | None = None

    # Robot (overrides on top of Tron2RTConfig defaults)
    robot_ip: str = "10.192.1.2"
    camera_host: str = "10.192.1.4"
    go_to_start: bool = True
    # Return to the start pose on exit (Ctrl+C / end of run).
    reset_on_disconnect: bool = True
    # The policy does not consume tactile streams; leave them off to save USB/CPU.
    enable_tactile: bool = False
    # Must match the TCP calibration used when recording the training data:
    # observations and dataset actions are expressed in the calibrated TCP frame.
    use_tool_calibration: bool = True
    # Explicit tool_tcp.yaml; None = the driver's default calibration dir.
    tool_calibration_path: str | None = None
    # Tron2RT driver log level.
    log_level: str = "INFO"

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings (dataset is recorded at 30 fps)
    runtime_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 1000000

    # Dry run mode
    dry_run: bool = False

    # Action chunking: number of actions executed from each predicted chunk
    # before re-querying the server (model action_horizon is 50).
    action_horizon: int = 50


def make_robot_config(args: Args) -> Tron2RTConfig:
    return Tron2RTConfig(
        robot_ip=args.robot_ip,
        camera_host=args.camera_host,
        go_to_start=args.go_to_start,
        reset_on_disconnect=args.reset_on_disconnect,
        gripper=TaccapFollowerConfig(enable_tactile=args.enable_tactile),
        use_tool_calibration=args.use_tool_calibration,
        tool_calibration_path=pathlib.Path(args.tool_calibration_path) if args.tool_calibration_path else None,
        log_level=args.log_level,
    )


def main(args: Args) -> None:
    logger.info(_run_config.describe(args, Args, RUNS_DIR))

    # Build (and validate) the robot config before connecting: the websocket
    # client blocks until the policy server answers.
    robot_config = make_robot_config(args)
    logger.info(
        f"TRON2 RT: robot_ip={robot_config.robot_ip}, camera_host={robot_config.camera_host}, "
        f"gripper={robot_config.gripper.type if robot_config.gripper else None}, "
        f"tactile={args.enable_tactile}"
    )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    base_environment = _env.Tron2RTEnvironment(
        robot_config=robot_config,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
    )

    environment: _environment.Environment = base_environment
    if args.dry_run:
        logger.info("DRY RUN mode: actions will be printed, not executed")
        environment = DryRunEnvironmentWrapper(base_environment)
    if args.prompt is not None:
        environment = PromptEnvironmentWrapper(environment, args.prompt)

    policy = action_chunk_broker.ActionChunkBroker(
        policy=ws_client_policy,
        action_horizon=args.action_horizon,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=policy),
        subscribers=[],
        max_hz=args.runtime_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # SIGINT handling: first press asks the runtime to stop cleanly; `finally`
    # then disconnects, which (reset_on_disconnect) returns the robot to the
    # start pose. A second Ctrl+C escapes via os._exit.
    _shutdown_in_progress = threading.Event()

    def signal_handler(sig, frame):
        if _shutdown_in_progress.is_set():
            logger.warn("Second Ctrl+C — forcing exit. Robot may not return to start pose.")
            os._exit(1)
        _shutdown_in_progress.set()
        logger.info("Ctrl+C — stopping runtime gracefully (press Ctrl+C again to force exit)")
        runtime.request_stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        try:
            base_environment.disconnect()
        except Exception as e:
            logger.warn(f"Error disconnecting: {e}")


if __name__ == "__main__":
    main(_run_config.cli(main, Args, RUNS_DIR))
