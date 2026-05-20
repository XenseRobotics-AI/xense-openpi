#!/usr/bin/env python
"""Main script for BiFlexiv Rizon4 RT dual-arm robot inference with OpenPI.

Two files describe a launch. --args.robot-recipe picks the physical bench: a
name resolves against examples/bi_flexiv_rizon4_rt/recipes/, a path loads any
recipe YAML. It carries the arm SNs, start/home poses, head camera and gripper
block — the lerobot config dataclass stopped carrying bench hardware when
`stations/` was folded into recipes upstream. See recipes/README.md.

--args.run picks a run YAML from runs/, which presets any of the flags below
(including the recipe), so a full launch is one line instead of ten. Flags still
override the file. See runs/README.md.

Example usage:
    # A run file — everything preset
    python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole

    # ...with a one-off change on top
    python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole --args.dry-run

    # Basic inference, all on the CLI
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-05 --args.host 192.168.2.100 --args.port 8000

    # With RTC enabled
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-05 --args.host 192.168.2.100 --args.port 8000 --args.rtc-enabled

    # A different bench (taccap grippers)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-04 --args.host 192.168.2.100 --args.port 8000

    # A recipe from the lerobot-xense tree
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe ~/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-04.yaml \\
        --args.host 192.168.2.100 --args.port 8000

    # Dry run (robot connected but actions not sent)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-05 --args.host 192.168.2.100 --args.port 8000 --args.dry-run

    # Inference + simultaneous recording in LeRobot format
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-05 --args.host 192.168.2.100 --args.port 8000 \\
        --args.record \\
        --args.record-repo-id Xense/my_new_dataset \\
        --args.task "pack 6 cosmetic bottles into the carton"

    # Inference + stream head camera & state to the video-playback laptop at 10 Hz
    # (off-laptop detection + seamless video switching; never blocks control)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-05 --args.host 192.168.2.100 --args.port 8000 \\
        --args.subscribe --args.subscribe-url ws://192.168.2.50:9100 --args.subscribe-hz 10

    # Inference with Pico4 human intervention (both grips held → teleop takes over)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-05 --args.host 192.168.2.100 --args.port 8000 --args.pico4-intervention
"""

from dataclasses import dataclass
import os
import pathlib
import signal
import threading
from typing import override

from lerobot.teleoperators.bi_pico4 import BiPico4
from lerobot.teleoperators.bi_pico4.config_bi_pico4 import BiPico4Config
from lerobot.utils.robot_utils import get_logger
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
import examples.bi_flexiv_rizon4_rt.recipe as _recipe
import examples.bi_flexiv_rizon4_rt.recorder as _recorder
import examples.bi_flexiv_rizon4_rt.subscribe as _subscribe
import examples.run_config as _run_config

logger = get_logger("BiFlexivRizon4RTMain")

# Run YAMLs shipped with this example; --args.run resolves bare names here.
RUNS_DIR = pathlib.Path(__file__).parent / "runs"

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
    """Arguments for BiFlexiv Rizon4 RT inference.

    The bench comes from --args.robot-recipe. Everything else here is run
    tuning, which the CLI owns outright: every tuning flag has a concrete
    default, so it is always applied on top of the decoded recipe. A tuning key
    written into a recipe — or already present in an upstream lerobot
    teleop/record recipe — loses to the flag; the loader logs each one it
    overrides so the swap is visible rather than silent.

    Any of these can be preset in a run YAML under runs/ and selected with
    --args.run, which is how a full launch fits on one line. Flags still win over
    the file; see examples/run_config.py.
    """

    # Which run YAML to take the settings below from. A name resolves against
    # examples/bi_flexiv_rizon4_rt/runs/; a path loads any YAML. Optional — with
    # no run file this is the same CLI it has always been.
    run: str | None = None

    # Which physical bench to drive. A name resolves against
    # examples/bi_flexiv_rizon4_rt/recipes/ (forward-01, forward-04, forward-05,
    # forward-06, diagonal-02); a path loads any recipe YAML, including one from
    # the lerobot-xense tree. The recipe carries the arm SNs, start/home poses,
    # head camera and gripper block — the lerobot config dataclass no longer
    # does. Required (here or in the run YAML): connecting to the wrong bench is
    # not a safe default.
    robot_recipe: str | None = None

    # Policy server
    host: str = "localhost"
    port: int = 8000

    # Robot run tuning
    use_force: bool = False
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    enable_tactile_sensors: bool = True
    log_level: str = "DEBUG"

    # Tactile camera mapping (lerobot camera name -> policy-side name).
    # The four values below correspond to BiFlexivTactileInputs.EXPECTED_CAMERAS;
    # only edit them if your robot exposes the tactile cameras under different keys.
    left_tactile_top_cam: str = "left_tactile_top"
    left_tactile_bottom_cam: str = "left_tactile_bottom"
    right_tactile_top_cam: str = "right_tactile_top"
    right_tactile_bottom_cam: str = "right_tactile_bottom"

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
    # obs loop (which is pinned to camera FPS). 0 = disabled (legacy single-
    # threaded Runtime). When enabled, RTC's frequency_hz tracks action_hz
    # so its delay estimation stays consistent with reality.
    action_hz: float = 0.0
    paced_queue_size: int = 50

    # Enable the obs subscriber that streams observations to the downstream
    # video-playback laptop (③) for off-laptop detection + seamless video switching.
    # One-way ws push on a daemon thread; never blocks the 30 Hz control loop.
    # (Inference is on the separate 5090 server, set via --host/--port.)
    # NB: distinct from --args.robot-recipe — this is the detection-data stream.
    subscribe: bool = False
    # obs ws URL of the video-playback laptop's app (its --obs_port, default 9100).
    subscribe_url: str = "ws://127.0.0.1:9100"
    # Which raw cameras to stream from observation["images_raw"]; the detector uses head.
    subscribe_cameras: tuple[str, ...] = ("head",)
    # Stream the 20-D robot state — the gripper detector needs it.
    subscribe_state: bool = True
    # Also stream the 20-D model action (debug/overlay only; the detector ignores it).
    subscribe_action: bool = False
    # Cap the stream rate to this many frames/sec (wall-clock throttle, independent of
    # runtime_hz). 0 = stream on every control step. e.g. 10 = stream the head at 10 Hz.
    subscribe_hz: float = 0.0
    # Stream every Nth step (integer subsample); prefer subscribe_hz to target a rate.
    subscribe_stride: int = 1
    # Startup handshake: when --subscribe is set, block until the video-playback laptop
    # is reachable before running (like the VLA client waits for the inference server),
    # so we never run inference with the screen unreachable. Give up after this many
    # seconds; 0 = wait forever (retry), matching the VLA client.
    subscribe_handshake_timeout: float = 0.0

    # Recording (LeRobot format, raw 640x480 images + absolute actions)
    record: bool = False
    record_repo_id: str = "Xense/recorded_dataset"
    record_root: str | None = None  # local save path, defaults to ~/.cache/huggingface/lerobot
    task: str = "pack 6 cosmetic bottles into the carton"

    # Pico4 human-in-the-loop intervention (hold both grips to take over)
    pico4_intervention: bool = False
    pico4_pos_sensitivity: float = 1.0
    pico4_ori_sensitivity: float = 1.0


def main(args: Args) -> None:
    logger.info(_run_config.describe(args, Args, RUNS_DIR))
    if args.robot_recipe is None:
        raise SystemExit(
            "No bench selected. Pass --args.robot-recipe <name>, or --args.run <name> "
            f"for a run file that sets it. Recipes: {', '.join(_recipe.available_recipes())}."
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

    # Build the robot config before connecting: WebsocketClientPolicy blocks
    # until the policy server answers, and neither a typo'd recipe name nor a
    # bad key inside one should cost that wait. Decoding here also means the
    # path logged is provably the file the arms are configured from.
    recipe_path = _recipe.resolve_recipe_path(args.robot_recipe)
    robot_config = _recipe.load_robot_config(
        recipe_path,
        use_force=args.use_force,
        go_to_start=args.go_to_start,
        stiffness_ratio=args.stiffness_ratio,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        enable_tactile_sensors=args.enable_tactile_sensors,
        log_level=args.log_level,
    )
    logger.info(
        f"Robot recipe: {recipe_path} "
        f"(left={robot_config.left_robot_sn}, right={robot_config.right_robot_sn}, "
        f"gripper={robot_config.gripper.type if robot_config.gripper else None})"
    )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    base_environment = _env.BiFlexivRizon4RTEnvironment(
        robot_config=robot_config,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
        tactile_camera_mapping={
            "left_tactile_top": args.left_tactile_top_cam,
            "left_tactile_bottom": args.left_tactile_bottom_cam,
            "right_tactile_top": args.right_tactile_top_cam,
            "right_tactile_bottom": args.right_tactile_bottom_cam,
        },
    )

    if args.dry_run:
        logger.info("DRY RUN mode: actions will be printed, not executed")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        environment = base_environment

    subscribers = []
    if args.record:
        if args.dry_run:
            logger.warn(
                "Recording is enabled in dry-run mode — state/action data will be from policy output only (no real robot motion)"
            )
        recorder = _recorder.make_recorder_subscriber(
            repo_id=args.record_repo_id,
            task=args.task,
            fps=int(args.runtime_hz),
            root=args.record_root,
        )
        subscribers.append(recorder)
        logger.info(f"Recording enabled: repo_id={args.record_repo_id}, task='{args.task}'")

    if args.subscribe:
        # require_handshake blocks here until the video-playback laptop is up and
        # greets us — so, like the VLA policy client waiting for the inference server,
        # we never proceed to inference while the screen PC is unreachable.
        obs_subscriber = _subscribe.make_obs_subscriber(
            uri=args.subscribe_url,
            cameras=tuple(args.subscribe_cameras),
            send_state=args.subscribe_state,
            send_action=args.subscribe_action,
            subscribe_hz=args.subscribe_hz,
            send_stride=args.subscribe_stride,
            require_handshake=True,
            handshake_timeout_s=args.subscribe_handshake_timeout,
        )
        subscribers.append(obs_subscriber)
        logger.info(f"Subscribing obs to detection machine: {args.subscribe_url}")

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

    if decoupled_mode:
        logger.info(f"Decoupled runtime: obs at ~{args.runtime_hz} Hz (camera-bound), action at {args.action_hz} Hz")
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
            logger.warn(f"Error disconnecting Pico4: {e}")
        try:
            actual_env = environment
            if isinstance(actual_env, _intervention.InterventionEnvironmentWrapper):
                actual_env = actual_env._wrapped_env
            if isinstance(actual_env, DryRunEnvironmentWrapper):
                actual_env = actual_env._wrapped_env
            actual_env.disconnect()
        except Exception as e:
            logger.warn(f"Error disconnecting: {e}")

    # SIGINT handling: first press asks the runtime to wind down threads
    # cleanly (DecoupledRuntime joins its action + obs threads here, ~0.5 s);
    # main's `finally` then runs safe_disconnect, which homes the arms via
    # MoveJ before releasing the SDK — same end-state as the original
    # synchronous runtime. A second Ctrl+C while shutdown is in progress
    # escapes to os._exit, accepting that the arms may not return home.
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
        safe_disconnect()


if __name__ == "__main__":
    main(_run_config.cli(main, Args, RUNS_DIR))
