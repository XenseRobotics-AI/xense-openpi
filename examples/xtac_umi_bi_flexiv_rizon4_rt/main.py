#!/usr/bin/env python
"""Run an XTac-UMI-trained OpenPI checkpoint on a BiFlexiv Rizon4 RT robot.

Pipeline (everything client-side stays in the Flexiv gripper frame)::

    env (20D XTac-UMI-ordered obs) -> broker (Flexiv-frame action queue)
        -> XtacUmiPolicyAdapter (gripper end-frame change of basis)
        -> WebsocketClientPolicy -> XTac-UMI policy server

The dim order is the policy's on both sides — ``real_env.py`` assembles the state
per-side-grouped from the driver's named keys — so nothing regroups dims anywhere.

--args.robot-recipe picks the bench, reusing
``examples/bi_flexiv_rizon4_rt/recipes/``; ``forward-04`` is the taccap-gripper
bench. The head camera a recipe pins is dropped: XTac-UMI checkpoints mask the
base_0_rgb slot out.

This is a reduced variant of examples/bi_flexiv_rizon4_rt/main.py: no recording
and no Pico4 intervention. It runs the synchronous Runtime by default and the
pace-decoupled one (obs thread + action thread) with ``--args.action-hz > 0``,
with optional RTC action chunking in either.

--args.run picks a run YAML from runs/, which presets any of the flags below
(including the recipe), so a full launch is one line instead of ten. Flags still
override the file. See runs/README.md.

Example::

    # forward-04 rebased checkpoint：只打印策略动作，按 Ctrl+C 结束
    python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-04 \\
        --args.host 192.168.204.87 \\
        --args.port 8000 \\
        --args.no-align-gripper-frames \\
        --args.runtime-hz 30 \\
        --args.dry-run

    # --args.robot-recipe：选择 forward-04 台位的机器人、起始关节和夹爪配置。
    # --args.host / --args.port：策略服务器的 IP 和 WebSocket 端口。
    # --args.no-align-gripper-frames：关闭末端坐标轴转换；rebased 数据已采用 Flexiv TCP 约定。
    # --args.runtime-hz：控制循环最高频率为 30 Hz，与训练数据帧率一致。
    # --args.dry-run：只打印策略动作；连接、reset 和退出回 home 仍可能移动机器人。

    # A run file — everything preset
    python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main --args.run dry-run

    # All on the CLI
    python -m examples.xtac_umi_bi_flexiv_rizon4_rt.main \\
        --args.robot-recipe forward-04 \\
        --args.host 192.168.142.220 --args.port 8000 \\
        --args.runtime-hz 30 --args.rtc-enabled --args.dry-run
"""

from dataclasses import dataclass
import pathlib
import signal
import sys
from typing import override

from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import action_chunk_broker
from xense_client import paced_broker as _paced_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import decoupled_runtime as _decoupled_runtime
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.bi_flexiv_rizon4_rt.recipe as _recipe
import examples.run_config as _run_config
from examples.xtac_umi_bi_flexiv_rizon4_rt.env import XtacUmiBiFlexivRizon4RTEnvironment
from examples.xtac_umi_bi_flexiv_rizon4_rt.policy_adapter import XtacUmiPolicyAdapter
from examples.xtac_umi_bi_flexiv_rizon4_rt.real_env import STATE_KEYS

logger = get_logger("XtacUmiBiFlexivRizon4RTMain")

# Run YAMLs shipped with this example; --args.run resolves bare names here.
RUNS_DIR = pathlib.Path(__file__).parent / "runs"

# The frequency XTac-UMI data is recorded at. Drifting from it hurts both the
# policy's input distribution and RTC's delay estimation, which reads frequency_hz.
_TRAINING_HZ = 30.0


class DryRunEnvironmentWrapper(_environment.Environment):
    """Intercepts policy actions and prints them without executing on the robot.

    What is printed is exactly what the robot would execute — the adapter has
    already converted it back to the Flexiv gripper frame.
    """

    def __init__(self, wrapped: XtacUmiBiFlexivRizon4RTEnvironment) -> None:
        self._wrapped_env = wrapped
        self._step = 0

    @override
    def reset(self) -> None:
        self._step = 0
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        self._step += 1
        values = np.asarray(action["actions"])
        logger.info(f"DRY RUN step {self._step}: action not executed")
        for index, (name, value) in enumerate(zip(STATE_KEYS, values, strict=True)):
            logger.info(f"  [{index:02d}] {name:<19s} {value:+0.6f}")
        rtc_metrics = action.get("rtc_metrics")
        if rtc_metrics is not None:
            logger.info(f"RTC metrics: {rtc_metrics}")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


@dataclass
class Args:
    """Arguments for XTac-UMI inference on BiFlexiv Rizon4 RT.

    The bench comes from --args.robot-recipe; everything else is run tuning that
    the CLI owns and always applies on top of the decoded recipe.

    Any of these can be preset in a run YAML under runs/ and selected with
    --args.run; flags still win over the file. See examples/run_config.py.
    """

    # Which run YAML to take the settings below from. A name resolves against
    # examples/xtac_umi_bi_flexiv_rizon4_rt/runs/; a path loads any YAML.
    run: str | None = None

    # Bench recipe: a name under examples/bi_flexiv_rizon4_rt/recipes/ (forward-04
    # is the taccap-gripper bench) or a path to any recipe YAML.
    robot_recipe: str | None = None

    # Policy server
    host: str = "localhost"
    port: int = 8000
    # Task prompt. None = fall back to the checkpoint's default_prompt (see the
    # training config's LeRobotXtacUmiDataConfig).
    prompt: str | None = None

    # Robot run tuning (applied on top of the recipe)
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    log_level: str = "INFO"

    # Gripper end-frame change of basis. On by default: the Flexiv driver reports
    # TCP orientations in the Flexiv gripper frame (z forward, y right, x up) and
    # the training data uses the XTac-UMI one (x forward, y left, z up). Disable
    # with --args.no-align-gripper-frames if a bench already reports UMI poses.
    align_gripper_frames: bool = True

    # Checkpoints trained with use_frame_relative_poses: send the previous and the
    # current state ([2, 20]) on every request. The model sees only per-frame motion,
    # which the server chains back onto the robot's current TCP, so the robot TCP
    # frames must already be axis-aligned with the XTac-UMI gripper frame:
    # align_gripper_frames has to be off.
    frame_relative_poses: bool = False
    # What the model's state is relative to, matching the checkpoint's data config:
    # "previous_frame" sends the previous control step, "episode_start" the state right
    # after the reset.
    state_anchor: str = "previous_frame"

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings
    runtime_hz: float = _TRAINING_HZ
    num_episodes: int = 1

    # Dry run mode: connect and read, but never send an action to the arms.
    dry_run: bool = False

    # Pace-decoupled runtime: the action thread pops at action_hz, independently of
    # the obs loop (pinned to camera FPS). 0 = the synchronous Runtime (legacy).
    #
    # action_hz ABOVE the recording rate speeds the robot up: a chunk of N steps is
    # played in N/action_hz seconds, so 60 Hz runs the 30 Hz demonstrations at twice
    # their speed. It does not add resolution — the chunk's geometry is unchanged.
    # Raise action_hz with the chain-relative actions in mind: every step multiplies
    # onto the previous one, so a shorter, faster chunk also drifts less.
    action_hz: float = 0.0
    # Producer/consumer buffer depth. >= action_hz x worst-case inference stall.
    paced_queue_size: int = 50

    # Non-RTC action chunking
    action_horizon: int = 50

    # RTC config
    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 30
    execution_horizon: int = 50
    blend_steps: int = 0
    default_delay: int = 4


def check_args(args: Args) -> None:
    """Refuse flag combinations that would run the policy in the wrong frame."""
    if args.state_anchor not in ("previous_frame", "episode_start"):
        raise SystemExit(f"--args.state-anchor must be previous_frame or episode_start, got {args.state_anchor!r}")
    if args.state_anchor != "previous_frame" and not args.frame_relative_poses:
        raise SystemExit("--args.state-anchor needs --args.frame-relative-poses.")
    if args.action_hz < 0:
        raise SystemExit(f"--args.action-hz must be >= 0 (0 = synchronous runtime), got {args.action_hz}")
    if args.paced_queue_size < 1:
        raise SystemExit(f"--args.paced-queue-size must be >= 1, got {args.paced_queue_size}")
    if args.frame_relative_poses and args.align_gripper_frames:
        raise SystemExit(
            "--args.frame-relative-poses needs --args.no-align-gripper-frames: the robot TCP frames must "
            "already match the XTac-UMI gripper frame, and the relative actions are chained onto them as is."
        )


def main(args: Args) -> None:
    logger.info(_run_config.describe(args, Args, RUNS_DIR))
    check_args(args)
    if args.robot_recipe is None:
        raise SystemExit(
            f"No bench selected. Pass --args.robot-recipe <name>. Recipes: {', '.join(_recipe.available_recipes())}."
        )
    if args.runtime_hz != _TRAINING_HZ:
        logger.warning(f"XTac-UMI data was recorded at {_TRAINING_HZ:g} Hz; requested {args.runtime_hz:g} Hz")

    # Decode the recipe before touching the network or the arms, so a bad key costs
    # a parse error rather than a connect timeout, and the path logged is provably
    # the file the arms are configured from. use_force is pinned False: XTac-UMI
    # checkpoints consume the 20D pose/gripper space, and force would change the
    # state layout.
    recipe_path = _recipe.resolve_recipe_path(args.robot_recipe)
    robot_config = _recipe.load_robot_config(
        recipe_path,
        use_force=False,
        go_to_start=args.go_to_start,
        stiffness_ratio=args.stiffness_ratio,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        enable_tactile=False,
        log_level=args.log_level,
    )
    logger.info(
        f"Robot recipe: {recipe_path} "
        f"(left={robot_config.left_robot_sn}, right={robot_config.right_robot_sn}, "
        f"gripper={robot_config.gripper.type if robot_config.gripper else None})"
    )

    websocket_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info(f"Server metadata: {websocket_policy.get_server_metadata()}")
    # Conversion boundary: brokers and queues below this adapter stay in the Flexiv
    # gripper frame; only websocket requests/responses are in XTac-UMI space.
    converted_policy = XtacUmiPolicyAdapter(
        websocket_policy,
        align_gripper_frames=args.align_gripper_frames,
        prompt=args.prompt,
    )

    base_environment = XtacUmiBiFlexivRizon4RTEnvironment(
        robot_config,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
        send_previous_state=args.frame_relative_poses,
        state_anchor=args.state_anchor,
    )
    environment: _environment.Environment
    if args.dry_run:
        logger.info("DRY RUN enabled: the robot is connected/read, but policy actions are not sent")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        environment = base_environment

    # The decoupled runtime pops from the broker at action_hz; the synchronous one
    # at runtime_hz.
    effective_broker_hz = args.action_hz if args.action_hz > 0 else args.runtime_hz
    if args.rtc_enabled:
        # The broker's queue holds Flexiv-frame actions (the adapter converts server
        # responses back). prev_chunk_left_over takes the opposite trip through the
        # adapter. frequency_hz must match the real consumption rate: the decoupled
        # runtime pops at action_hz, the synchronous one at runtime_hz. RTC reads it
        # to convert its per-step delay into seconds.
        broker = rtc_action_chunk_broker.RTCActionChunkBroker(
            policy=converted_policy,
            frequency_hz=effective_broker_hz,
            action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
            rtc_enabled=True,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            default_delay=args.default_delay,
            dry_run=args.dry_run,
        )
    else:
        broker = action_chunk_broker.ActionChunkBroker(
            policy=converted_policy,
            action_horizon=args.action_horizon,
        )

    decoupled_mode = args.action_hz > 0
    if decoupled_mode:
        # target_hz=action_hz throttles the producer to the consumer's rate: without
        # it an RTC broker's internal queue is drained faster than its background
        # inference refills it, which spams "Action queue exhausted" at startup.
        broker = _paced_broker.PacedBroker(
            inner=broker,
            queue_size=args.paced_queue_size,
            target_hz=args.action_hz,
        )

    runtime: _runtime.Runtime | _decoupled_runtime.DecoupledRuntime
    if decoupled_mode:
        if args.action_hz != _TRAINING_HZ:
            logger.warning(
                f"XTac-UMI data was recorded at {_TRAINING_HZ:g} Hz; action_hz {args.action_hz:g} plays the "
                f"same chunks at {args.action_hz / _TRAINING_HZ:.2f}x their recorded speed"
            )
        logger.info(
            f"Decoupled runtime: obs at ~{args.runtime_hz:g} Hz (camera-bound), action at {args.action_hz:g} Hz"
        )
        runtime = _decoupled_runtime.DecoupledRuntime(
            environment=environment,
            broker=broker,
            subscribers=[],
            obs_hz=args.runtime_hz,
            action_hz=args.action_hz,
            num_episodes=args.num_episodes,
        )
    else:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(policy=broker),
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
        )

    def disconnect() -> None:
        try:
            environment.disconnect()
        except Exception as error:
            logger.warning(f"Error while disconnecting: {error}")

    # SIGINT: ask the runtime to wind down; the `finally` then runs disconnect(),
    # which homes the arms via the driver's own shutdown. Unlike
    # bi_flexiv_rizon4_rt there is no os._exit escape on a second Ctrl+C — the
    # handler is idempotent, so repeated presses just re-request the stop (forcing
    # an exit mid-homing would leave the arms in an unknown pose).
    def signal_handler(sig, frame) -> None:
        del sig, frame
        logger.info("Ctrl+C received; stopping runtime")
        if hasattr(runtime, "request_stop"):
            runtime.request_stop()
        else:
            disconnect()
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    try:
        runtime.run()
    finally:
        disconnect()


if __name__ == "__main__":
    main(_run_config.cli(main, Args, RUNS_DIR))
