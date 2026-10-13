#!/usr/bin/env python
"""Main script for bimanual Elite CS66 RT inference with OpenPI.

Two files describe a launch. --args.robot-recipe picks the physical bench: a
name resolves against examples/bi_elite_cs66_rt/recipes/, a path loads any
recipe YAML. It carries the controller IPs, start/home poses, mount rotations,
head camera and gripper block. See recipes/README.md.

--args.run picks a run YAML from runs/, which presets any of the flags below
(including the recipe), so a full launch is one line. Flags still override the
file. See examples/run_config.py.

State/action are 20D, per-side-grouped (grippers at dims 9 and 19) — the order
LeRobotBiEliteDataConfig trains on, NOT the BiFlexiv order.

Example usage:
    # Dry run (robot connected, actions printed but not sent)
    python -m examples.bi_elite_cs66_rt.main --args.run dry-run --args.host <server>

    # Basic inference, all on the CLI
    python -m examples.bi_elite_cs66_rt.main \\
        --args.robot-recipe diagonal-08-taccap --args.host 192.168.2.100 --args.port 8000

    # With RTC enabled
    python -m examples.bi_elite_cs66_rt.main \\
        --args.robot-recipe diagonal-08-taccap --args.host 192.168.2.100 --args.port 8000 --args.rtc-enabled
"""

from dataclasses import dataclass
import os
import pathlib
import signal
import threading
from typing import override

from lerobot.utils.robot_utils import get_logger
from xense_client import action_chunk_broker
from xense_client import paced_broker as _paced_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import decoupled_runtime as _decoupled_runtime
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.bi_elite_cs66_rt.env as _env
from examples.bi_elite_cs66_rt.real_env import STATE_KEYS
import examples.bi_elite_cs66_rt.recipe as _recipe
import examples.run_config as _run_config

logger = get_logger("BiEliteCS66RTMain")

# Run YAMLs shipped with this example; --args.run resolves bare names here.
RUNS_DIR = pathlib.Path(__file__).parent / "runs"


class DryRunEnvironmentWrapper(_environment.Environment):
    """Intercepts policy actions and prints them without executing on robot."""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0
        self._last_rtc_inference_seq = -1

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        self._last_rtc_inference_seq = -1
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
        if actions is not None:
            logger.info(f"\n{'─' * 80}")
            logger.info(f"Step {self._step_count} - policy action (20D Cartesian):")
            logger.info(f"{'─' * 80}")
            for i, (label, value) in enumerate(zip(STATE_KEYS, actions, strict=True)):
                logger.info(f"  [{i:2d}] {label:18s}: {value:+.6f}")
            logger.info(f"{'─' * 80}")

            rtc = action.get("rtc_metrics")
            if rtc is not None:
                seq = int(rtc.get("inference_seq", 0))
                if seq != self._last_rtc_inference_seq:
                    self._last_rtc_inference_seq = seq
                    logger.info(
                        f"RTC [dry run] new chunk #{seq}: "
                        f"infer+server RTT={rtc['infer_round_trip_ms']:.1f} ms; "
                        f"delay est={rtc['estimated_delay_steps']} steps, "
                        f"real={rtc['real_delay_steps']} steps; "
                        f"queue_after={rtc['queue_size_after_merge']}; "
                        f"infer RTT p95={rtc['latency_p95_ms']:.1f} ms"
                    )

            logger.info("DRY RUN: action NOT sent to robot")
            logger.info(f"{'─' * 80}\n")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


@dataclass
class Args:
    """Arguments for bimanual Elite CS66 RT inference.

    The bench comes from --args.robot-recipe. The robot tuning flags below are
    applied on top of the decoded recipe; the loader logs each recipe value it
    overrides. A flag left at None keeps the recipe's value.
    """

    # Which run YAML to take the settings below from. A name resolves against
    # examples/bi_elite_cs66_rt/runs/; a path loads any YAML.
    run: str | None = None
    # Which physical bench to drive. A name resolves against
    # examples/bi_elite_cs66_rt/recipes/ (diagonal-08-taccap); a path loads any
    # recipe YAML, including one from the lerobot-xense tree. Required (here or
    # in the run YAML): connecting to the wrong bench is not a safe default.
    robot_recipe: str | None = None

    # Policy server
    host: str = "localhost"
    port: int = 8000

    # Robot run tuning
    go_to_start: bool = True
    # Routed onto the recipe's `gripper:` block. The policy does not consume
    # tactile images, so leaving the sensors off saves USB bandwidth.
    enable_tactile: bool = False
    # Per-tick manipulability log; the teleop recipe turns it on for diagnosis.
    log_manipulability: bool = False
    # Per-arm Cartesian speed caps of the servo loop (m/s, rad/s). None keeps the recipe's.
    max_lin_speed: float | None = None
    max_ang_speed: float | None = None

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings
    runtime_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 1000000

    # Dry run mode
    dry_run: bool = False

    # Non-RTC action chunking
    action_horizon: int = 50

    # RTC config
    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 20
    execution_horizon: int = 50
    blend_steps: int = 0
    default_delay: int = 4

    # Decoupled mode: action thread emits at action_hz independently of the
    # obs loop (which is pinned to camera FPS). 0 = disabled (single-threaded
    # Runtime). When enabled, RTC's frequency_hz tracks action_hz so its delay
    # estimation stays consistent with reality.
    action_hz: float = 0.0
    paced_queue_size: int = 50


def main(args: Args) -> None:
    logger.info(_run_config.describe(args, Args, RUNS_DIR))

    if args.robot_recipe is None:
        raise SystemExit(
            "No bench selected. Pass --args.robot-recipe <name>, or --args.run <name> "
            f"for a run file that sets it. Recipes: {', '.join(_recipe.available_recipes())}."
        )

    # Build the robot config before connecting: WebsocketClientPolicy blocks
    # until the policy server answers, and a bad recipe should not cost that wait.
    recipe_path = _recipe.resolve_recipe_path(args.robot_recipe)
    robot_config = _recipe.load_robot_config(
        recipe_path,
        enable_tactile=args.enable_tactile,
        log_manipulability=args.log_manipulability,
        max_lin_speed=args.max_lin_speed,
        max_ang_speed=args.max_ang_speed,
    )
    logger.info(
        f"Robot recipe: {recipe_path} "
        f"(left={robot_config.left_robot_ip}, right={robot_config.right_robot_ip}, "
        f"gripper={robot_config.gripper.type if robot_config.gripper else None}, "
        f"max_lin_speed={robot_config.max_lin_speed}, max_ang_speed={robot_config.max_ang_speed})"
    )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    base_environment = _env.BiEliteCS66RTEnvironment(
        robot_config,
        go_to_start=args.go_to_start,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
    )

    if args.dry_run:
        logger.info("DRY RUN mode: actions will be printed, not executed")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        environment = base_environment

    decoupled_mode = args.action_hz > 0
    # In decoupled mode the broker is popped at action_hz, not runtime_hz —
    # RTC's delay/blend math reads frequency_hz to estimate per-step elapsed
    # time, so it must match the actual pop rate.
    effective_broker_hz = args.action_hz if decoupled_mode else args.runtime_hz

    if args.rtc_enabled:
        # The RTC prefix is re-based into the current state's delta frame on the
        # policy server (training input pipeline), so no delta_state_dim here.
        policy = rtc_action_chunk_broker.RTCActionChunkBroker(
            policy=ws_client_policy,
            frequency_hz=effective_broker_hz,
            action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
            rtc_enabled=args.rtc_enabled,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            default_delay=args.default_delay,
            dry_run=args.dry_run,
        )
    else:
        policy = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
        )

    if decoupled_mode:
        # target_hz=action_hz throttles the producer to the consumer rate, so it
        # does not drain the broker's queue faster than inference can refill it.
        policy = _paced_broker.PacedBroker(
            inner=policy,
            queue_size=args.paced_queue_size,
            target_hz=args.action_hz,
        )
        logger.info(f"Decoupled runtime: obs at ~{args.runtime_hz} Hz (camera-bound), action at {args.action_hz} Hz")
        runtime = _decoupled_runtime.DecoupledRuntime(
            environment=environment,
            broker=policy,
            subscribers=[],
            obs_hz=args.runtime_hz,
            action_hz=args.action_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(policy=policy),
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    # SIGINT handling: first press asks the runtime to wind down threads
    # cleanly; main's `finally` then disconnects the robot. A second Ctrl+C
    # while shutdown is in progress escapes to os._exit, accepting that the
    # arms may not return home.
    _shutdown_in_progress = threading.Event()

    def signal_handler(sig, frame):
        if _shutdown_in_progress.is_set():
            logger.warn("Second Ctrl+C — forcing exit. Arms may not return home cleanly.")
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
