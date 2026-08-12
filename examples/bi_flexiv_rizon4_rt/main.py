#!/usr/bin/env python
"""Main script for BiFlexiv Rizon4 RT dual-arm robot inference with OpenPI.

Example usage:
    # Basic inference
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000

    # With RTC enabled
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --rtc_enabled

    # Side-mount configuration
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --bi_mount_type side

    # Dry run (robot connected but actions not sent)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --dry_run

    # Inference + simultaneous recording in LeRobot format
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 \\
        --record \\
        --record_repo_id Xense/my_new_dataset \\
        --task "pack 6 cosmetic bottles into the carton"

    # Keyboard-delimited episodes (lerobot style): right arrow starts / ends
    # and saves an episode; left arrow discards and re-records; ESC exits.
    # Add --confirm_success to mark each episode with Enter (success) or
    # Backspace (failure) at frame level (observation.is_success).
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 \\
        --record \\
        --record_repo_id Xense/my_new_dataset \\
        --task "pack 6 cosmetic bottles into the carton" \\
        --keyboard_control --confirm_success

    # Inference with Pico4 human intervention (both grips held → teleop takes over)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --pico4_intervention

    # Inference with Pico4 intervention + simultaneous recording. Recording
    # under --pico4_intervention automatically adds a frame-level
    # observation.is_intervention flag (1 = human takeover frame, 0 = policy),
    # so the recorded dataset can identify which frames were teleoperated.
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 \\
        --pico4_intervention \\
        --record \\
        --record_repo_id Xense/my_new_dataset \\
        --task "pack 6 cosmetic bottles into the carton"
"""

from dataclasses import dataclass
import os
import signal
import threading

from lerobot.teleoperators.bi_pico4 import BiPico4
from lerobot.teleoperators.bi_pico4.config_bi_pico4 import BiPico4Config
from lerobot.utils.robot_utils import get_logger
from typing_extensions import override
import tyro
from xense_client import action_chunk_broker
from xense_client import paced_broker as _paced_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import decoupled_runtime as _decoupled_runtime
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.bi_flexiv_rizon4_rt.env as _env
import examples.bi_flexiv_rizon4_rt.intervention as _intervention
import examples.bi_flexiv_rizon4_rt.keyboard_control as _keyboard_control
import examples.bi_flexiv_rizon4_rt.recorder as _recorder

logger = get_logger("BiFlexivRizon4RTMain")

# Action dimension labels for dry-run logging
_ACTION_LABELS = [
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "left_gripper.pos",
    "right_gripper.pos",
]


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
            for i, (label, value) in enumerate(zip(_ACTION_LABELS, actions)):
                logger.info(f"  [{i:2d}] {label:18s}: {value:+.6f}")
            logger.info(f"{'─' * 80}")
            rtc = action.get("rtc_metrics")
            if rtc is not None:
                seq = int(rtc.get("inference_seq", 0))
                if seq != self._last_rtc_inference_seq:
                    self._last_rtc_inference_seq = seq
                    next_d = rtc["delay_for_next_infer_steps"]
                    logger.info(
                        f"RTC [dry run] new chunk #{seq}: "
                        f"infer+server RTT={rtc['infer_round_trip_ms']:.1f} ms "
                        f"(model + WebSocket round-trip); "
                        f"delay est={rtc['estimated_delay_steps']} steps, "
                        f"real={rtc['real_delay_steps']} steps, "
                        f"time-based={rtc['inference_delay_steps']} steps; "
                        f"delay_for_next_infer={next_d}; "
                        f"merge={rtc['merge_ms']:.2f} ms; "
                        f"queue_after={rtc['queue_size_after_merge']}; "
                        f"infer RTT p95={rtc['latency_p95_ms']:.1f} ms"
                    )
            logger.info("DRY RUN: action NOT sent to robot")
            logger.info(f"{'─' * 80}\n")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


@dataclass
class Args:
    """Arguments for BiFlexiv Rizon4 RT inference."""

    # Policy server
    host: str = "localhost"
    port: int = 8000

    # Robot configuration
    bi_mount_type: str = "side"  # "forward" or "side"
    use_force: bool = False
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    enable_tactile_sensors: bool = False
    log_level: str = "DEBUG"

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
    action_queue_size_to_get_new_actions: int = 30
    execution_horizon: int = 50
    blend_steps: int = 0
    default_delay: int = 4

    # Decoupled mode: action thread emits at action_hz independently of the
    # obs loop (which is pinned to camera FPS). 0 = disabled (legacy single-
    # threaded Runtime). When enabled, RTC's frequency_hz tracks action_hz
    # so its delay estimation stays consistent with reality.
    action_hz: float = 0.0
    paced_queue_size: int = 50

    # Recording (LeRobot format, raw 640x480 images + absolute actions)
    record: bool = False
    record_repo_id: str = "Xense/recorded_dataset"
    record_root: str | None = None  # local save path, defaults to ~/.cache/huggingface/lerobot
    task: str = "pack 6 cosmetic bottles into the carton"
    # Record a frame-level observation.is_intervention flag when pico4
    # intervention is active (auto-enabled with --pico4_intervention; pass
    # explicitly to override).
    record_intervention_flag: bool | None = None

    # Keyboard-controlled episode delimiting (lerobot style):
    #   Right arrow: start episode / end + save
    #   Left arrow : discard current episode and re-record
    #   Enter      : end + save with is_success=True (needs --confirm_success)
    #   Backspace  : end + save with is_success=False (needs --confirm_success)
    #   ESC        : discard current episode and exit
    keyboard_control: bool = False
    confirm_success: bool = False

    # Pico4 human-in-the-loop intervention (hold both grips to take over)
    pico4_intervention: bool = False
    pico4_pos_sensitivity: float = 1.0
    pico4_ori_sensitivity: float = 1.0


def main(args: Args) -> None:
    if args.keyboard_control and args.action_hz > 0:
        # KeyboardControlledEnvironmentWrapper blocks in get_observation,
        # which deadlocks the decoupled runtime's action thread. Refuse the
        # combo rather than hanging the robot.
        raise SystemExit(
            "--keyboard_control is not supported with --action_hz > 0 in this release. "
            "Run with --action_hz 0 (synchronous runtime), or disable keyboard control."
        )

    if args.pico4_intervention and args.rtc_enabled:
        # RTCActionChunkBroker owns an execution queue + blending; its reset
        # semantics differ from ActionChunkBroker. A correct RTC handoff needs
        # a separate design pass — refuse the combo rather than silently doing
        # the wrong thing.
        raise SystemExit(
            "--pico4_intervention is not supported with --rtc_enabled in this release. "
            "Run without --rtc_enabled, or disable intervention."
        )

    decoupled_mode = args.action_hz > 0
    if decoupled_mode and args.pico4_intervention:
        # DecoupledRuntime spawns an action thread that pops from a queue the
        # producer keeps filling. Switching control to teleop mid-stream would
        # require draining the queue race-free across three threads — defer.
        raise SystemExit(
            "--pico4_intervention is not supported with --action_hz > 0 in this release. "
            "Run with --action_hz 0 (synchronous runtime) when intervention is needed."
        )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    base_environment = _env.BiFlexivRizon4RTEnvironment(
        bi_mount_type=args.bi_mount_type,
        use_force=args.use_force,
        go_to_start=args.go_to_start,
        stiffness_ratio=args.stiffness_ratio,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        enable_tactile_sensors=args.enable_tactile_sensors,
        log_level=args.log_level,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
    )
    environment = base_environment

    if args.dry_run:
        logger.info("DRY RUN mode: actions will be printed, not executed")
        environment = DryRunEnvironmentWrapper(environment)

    keyboard_controller: _keyboard_control.KeyboardEpisodeController | None = None
    if args.keyboard_control:
        keyboard_controller = _keyboard_control.KeyboardEpisodeController(
            confirm_success=args.confirm_success
        )
        keyboard_controller.start()
        logger.info(
            "Keyboard episode control enabled. "
            "Right arrow toggles recording; left arrow re-records; "
            "ESC exits."
        )
        # Episodes are delimited by keyboard input, not by a fixed count or
        # step budget: run a practically unbounded number of episodes and let
        # the operator end each one.
        args.num_episodes = 10_000
        args.max_episode_steps = 0

    subscribers = []
    if args.record:
        if args.dry_run:
            logger.warning(
                "Recording is enabled in dry-run mode — state/action data will be from policy output only (no real robot motion)"
            )
        record_intervention = args.record_intervention_flag
        if record_intervention is None:
            record_intervention = args.pico4_intervention
        if args.pico4_intervention and record_intervention:
            logger.info("Recording frame-level observation.is_intervention flag (pico4 intervention).")
        recorder = _recorder.make_recorder_subscriber(
            repo_id=args.record_repo_id,
            task=args.task,
            fps=int(args.runtime_hz),
            root=args.record_root,
            controller=keyboard_controller,
            record_intervention=record_intervention,
            confirm_success=args.confirm_success,
        )
        subscribers.append(recorder)
        logger.info(f"Recording enabled: repo_id={args.record_repo_id}, task='{args.task}'")

    # In decoupled mode the broker is popped at action_hz, not runtime_hz —
    # RTC's internal delay/blend math reads frequency_hz to estimate
    # per-step elapsed time, so it must match the actual pop rate.
    effective_broker_hz = args.action_hz if decoupled_mode else args.runtime_hz

    if args.rtc_enabled:
        policy = rtc_action_chunk_broker.RTCActionChunkBroker(
            policy=ws_client_policy,
            frequency_hz=effective_broker_hz,
            action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
            rtc_enabled=args.rtc_enabled,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            default_delay=args.default_delay,
            dry_run=args.dry_run,
            # Training used DeltaActions(mask=[True]*18 + [False]*2): the first
            # 18 action dims (bi-arm TCP XYZ + 6D rot) are deltas relative to
            # obs.state; the last 2 (grippers) are absolute. Telling the broker
            # this lets it re-base prev_chunk_left_over from the previous
            # inference's state into the current obs.state before sending, so
            # the model sees a prefix consistent with its training distribution
            # and the postfix joins smoothly at merge.
            delta_state_dim=18,
        )
    else:
        policy = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
        )

    if decoupled_mode:
        # Wrap whichever broker we just built; the PacedBroker presents the
        # same BasePolicy contract to anything else but adds submit_obs /
        # pop_action / start / stop for DecoupledRuntime.
        #
        # target_hz=action_hz throttles the producer to the consumer rate.
        # Without it, the producer drains an internal broker queue (RTC has
        # one) faster than its background inference can refill, generating
        # "Action queue exhausted" warning spam during startup. For non-RTC
        # the queue.put back-pressure already paces things, but matching the
        # consumer rate is still the right default.
        policy = _paced_broker.PacedBroker(
            inner=policy,
            queue_size=args.paced_queue_size,
            target_hz=args.action_hz,
        )

    intervention_controller: _intervention.Pico4InterventionController | None = None
    if args.pico4_intervention:
        teleop = BiPico4(
            BiPico4Config(
                pos_sensitivity=args.pico4_pos_sensitivity,
                ori_sensitivity=args.pico4_ori_sensitivity,
            )
        )
        intervention_controller = _intervention.Pico4InterventionController(teleop, base_environment)
        intervention_controller.start()
        environment = _intervention.InterventionEnvironmentWrapper(environment, intervention_controller)
        agent = _intervention.InterventionPolicyAgent(
            inner_agent=_policy_agent.PolicyAgent(policy=policy),
            controller=intervention_controller,
            broker=policy,
        )
    else:
        agent = _policy_agent.PolicyAgent(policy=policy)

    if keyboard_controller is not None:
        environment = _env.KeyboardControlledEnvironmentWrapper(environment, keyboard_controller)

    if decoupled_mode:
        logger.info(
            f"Decoupled runtime: obs at ~{args.runtime_hz} Hz (camera-bound), " f"action at {args.action_hz} Hz"
        )
        runtime = _decoupled_runtime.DecoupledRuntime(
            environment=environment,
            broker=policy,  # PacedBroker
            subscribers=subscribers,
            obs_hz=args.runtime_hz,
            action_hz=args.action_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=agent,
            subscribers=subscribers,
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    def safe_disconnect() -> None:
        try:
            if intervention_controller is not None:
                intervention_controller.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting Pico4: {e}")
        try:
            actual_env = environment
            if isinstance(actual_env, _intervention.InterventionEnvironmentWrapper):
                actual_env = actual_env._wrapped_env
            if isinstance(actual_env, DryRunEnvironmentWrapper):
                actual_env = actual_env._wrapped_env
            actual_env.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    # SIGINT handling: first press asks the runtime to wind down threads
    # cleanly (DecoupledRuntime joins its action + obs threads here, ~0.5 s);
    # main's `finally` then runs safe_disconnect, which homes the arms via
    # MoveJ before releasing the SDK — same end-state as the original
    # synchronous runtime. A second Ctrl+C while shutdown is in progress
    # escapes to os._exit, accepting that the arms may not return home.
    _shutdown_in_progress = threading.Event()

    def signal_handler(sig, frame):
        if _shutdown_in_progress.is_set():
            logger.warning("Second Ctrl+C — forcing exit. Arms may not return home cleanly.")
            os._exit(1)
        _shutdown_in_progress.set()
        logger.info("Ctrl+C — stopping runtime gracefully " "(press Ctrl+C again to force exit)")
        runtime.request_stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except _keyboard_control.KeyboardExit:
        logger.info("Keyboard exit requested — shutting down cleanly")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        safe_disconnect()


if __name__ == "__main__":
    tyro.cli(main)
