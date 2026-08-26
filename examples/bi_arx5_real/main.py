import dataclasses
import pathlib
from typing import override

from lerobot.utils.robot_utils import get_logger
from xense_client import action_chunk_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.bi_arx5_real.env as _env
import examples.bi_arx5_real.recorder as _recorder
import examples.run_config as _run_config

logger = get_logger("BiARX5Main")

# Run YAMLs shipped with this example; --args.run resolves bare names here.
RUNS_DIR = pathlib.Path(__file__).parent / "runs"


class DryRunEnvironmentWrapper(_environment.Environment):
    """Wrapper: intercept and print policy action, but not actually execute"""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🔄 Episode {self._episode_count} - environment reset (dry run mode)")
        logger.info(f"{'=' * 80}\n")
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        obs = self._wrapped_env.get_observation()

        # prompt is automatically injected by the InjectDefaultPrompt transform of the policy server
        # no need to add it in the client

        # print observation summary (simplified version)
        # if self._step_count % 10 == 0:  # print observation summary every 10 steps
        #     state = obs.get("state")
        #     images = obs.get("images", {})
        #     logging.info(f"📊 step {self._step_count} - observation summary:")
        #     if state is not None:
        #         logging.info(
        #             f"   state dimension: {state.shape}, range: [{state.min():.3f}, {state.max():.3f}]"
        #         )
        #     logging.info(f"   image count: {len(images)}")

        return obs

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1

        # print policy output action
        actions = action.get("actions")
        if actions is not None:
            logger.info(f"\n{'─' * 80}")
            logger.info(f"🎯 step {self._step_count} - policy output action:")
            logger.info(f"{'─' * 80}")

            # print detailed action information
            # logger.info(f"action dimension: {actions.shape}")
            # logger.info(f"action type: {actions.dtype}")
            # logger.info(f"action range: [{actions.min():.6f}, {actions.max():.6f}]")

            # print each joint action value
            joint_names = [
                "left_joint_1",
                "left_joint_2",
                "left_joint_3",
                "left_joint_4",
                "left_joint_5",
                "left_joint_6",
                "left_gripper",
                "right_joint_1",
                "right_joint_2",
                "right_joint_3",
                "right_joint_4",
                "right_joint_5",
                "right_joint_6",
                "right_gripper",
            ]

            logger.info("\ndetailed action values:")
            for i, (name, value) in enumerate(zip(joint_names, actions)):
                logger.info(f"  [{i:2d}] {name:12s}: {value:+.6f} rad")

            logger.info("\ngripper action:")
            logger.info(f"  left gripper (index 6):  {actions[6]:.6f}")
            logger.info(f"  right gripper (index 13): {actions[13]:.6f}")

            logger.info(f"{'─' * 80}")
            logger.info("⚠️  dry run mode: action intercepted, not actually executed to robot")
            logger.info(f"{'─' * 80}\n")


@dataclasses.dataclass
class Args:
    """Arguments for BiARX5 inference.

    Any of these can be preset in a run YAML under runs/ and selected with
    --args.run; flags still win over the file. See examples/run_config.py.
    """

    # Which run YAML to take the settings below from. A name resolves against
    # examples/bi_arx5_real/runs/; a path loads any YAML.
    run: str | None = None

    host: str = "0.0.0.0"
    port: int = 8000

    num_episodes: int = 1
    max_episode_steps: int = 10000

    # bi_arx5 specific configs
    left_arm_port: str = "can1"
    right_arm_port: str = "can3"
    log_level: str = "INFO"
    use_multithreading: bool = True

    action_horizon: int = 30  # action_horizon for actionchunkbroker

    # lower controller config
    controller_dt: float = 0.002  # lower controller frequency, unit: second (0.002s = 2ms = 500Hz)
    control_mode: str = "joint_control"
    enable_tactile: bool = False  # enable tactile sensors, default False
    preview_time: float = 0.03  # preview time = 1/runtime_hz, for smooth interpolation
    runtime_hz: int = 30  # runtime frequency, unit: Hz
    # dry run mode: only print policy output, not actually execute action
    dry_run: bool = False

    # RTC config
    rtc_enabled: bool = False
    # Threshold to request new actions, when action queue size is less than this value, new actions will be requested
    action_queue_size_to_get_new_actions: int = 20
    # Sample action with rtc horizon
    execution_horizon: int = 30  # execution_horizon for rtc_action_chunk_broker
    # Number of steps to blend between old and new actions at merge point
    # 0 = no blending (hard switch), 2-3 = smooth transition
    blend_steps: int = 5
    # Default inference_delay for warmup and fallback (in steps)
    default_delay: int = 2

    # Recording (LeRobot format, raw HWC images + absolute actions)
    record: bool = False
    record_repo_id: str = "Xense/recorded_arx5_dataset"
    record_root: str | None = None  # local save path, defaults to ~/.cache/huggingface/lerobot
    task: str = "pick and place"


def main(args: Args) -> None:
    logger.info(_run_config.describe(args, Args, RUNS_DIR))
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    metadata = ws_client_policy.get_server_metadata()
    logger.info(f"Server metadata: {metadata}")

    # create base environment
    base_environment = _env.BiARX5RealEnvironment(
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        log_level=args.log_level,
        use_multithreading=args.use_multithreading,
        enable_tactile=args.enable_tactile,
        reset_position=metadata.get("reset_pose"),
        controller_dt=args.controller_dt,  # pass in lower controller frequency
        preview_time=args.preview_time,  # pass in preview time
        control_mode=args.control_mode,
    )

    # if dry run mode, wrap the environment with the wrapper
    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 dry run mode enabled")
        logger.info("   - policy action output will be printed")
        logger.info("   - action will not be sent to robot")
        logger.info("   - robot will stay in initial position")
        logger.info("=" * 80 + "\n")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        logger.info("✅ normal mode: action will be executed to robot")
        environment = base_environment

    subscribers = []
    if args.record:
        recorder = _recorder.make_recorder_subscriber(
            repo_id=args.record_repo_id,
            task=args.task,
            fps=int(args.runtime_hz),
            root=args.record_root,
        )
        subscribers.append(recorder)
        logger.info(f"Recording enabled: repo_id={args.record_repo_id}, task='{args.task}'")

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
            subscribers=subscribers,
            max_hz=args.runtime_hz,  # runtime frequency, unit: Hz
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
            subscribers=subscribers,
            max_hz=args.runtime_hz,  # runtime frequency, unit: Hz
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    def safe_disconnect():
        """safe disconnect robot"""
        try:
            # check if the environment is a wrapper
            actual_env = environment
            if isinstance(environment, DryRunEnvironmentWrapper):
                actual_env = environment._wrapped_env

            if hasattr(actual_env, "_env") and hasattr(actual_env._env, "robot"):
                if actual_env._env.robot.is_connected:
                    logger.info("safe disconnect robot...")
                    actual_env._env.disconnect()
                    logger.info("✓ robot disconnected")
                else:
                    logger.info("robot not connected, no need to disconnect")
        except Exception as disconnect_error:
            logger.warn(f"error disconnecting robot: {disconnect_error}")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("\n⚠️  detected user interrupt (Ctrl+C)")
        logger.info("program exited safely")
    except Exception as e:
        logger.error(f"\n❌ runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # always safe disconnect robot
        safe_disconnect()


if __name__ == "__main__":
    main(_run_config.cli(main, Args, RUNS_DIR))
