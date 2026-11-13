#!/usr/bin/env python
"""Replay a handheld XTac-UMI episode on a BiFlexiv Rizon4 RT robot.

No policy server: ``UmiReplayAgent`` composes the recorded hand motion onto the
arms' start TCPs and the synchronous Runtime sends one target per step. See
``replay_agent.py`` for the mapping and the checks that run before the first target.

The episode should have been recorded with the hands starting roughly the way the
arms do at the recipe's start joints; that is what the yaw is estimated from.

Dry run is the default: the robot connects and resets to its start pose, the plan
is built and summarized from the real start TCPs, and no target is sent.

Example::

    # Build and check the plan only (connecting and resetting still moves the arms to the start pose)
    python -m examples.xtac_umi_bi_flexiv_rizon4_rt.replay --args.run replay \\
        --args.repo-id Xense/<dataset> --args.episode 0

    # Replay on the arms at quarter speed
    python -m examples.xtac_umi_bi_flexiv_rizon4_rt.replay --args.run replay \\
        --args.repo-id Xense/<dataset> --args.episode 0 \\
        --args.speed-scale 0.25 --args.no-dry-run
"""

from dataclasses import dataclass
import pathlib
import signal
import time
from typing import override

from lerobot.utils.robot_utils import get_logger
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime

import examples.bi_flexiv_rizon4_rt.recipe as _recipe
import examples.run_config as _run_config
from examples.xtac_umi_bi_flexiv_rizon4_rt import replay_agent as _replay_agent
from examples.xtac_umi_bi_flexiv_rizon4_rt.env import XtacUmiBiFlexivRizon4RTEnvironment

logger = get_logger("XtacUmiReplay")

RUNS_DIR = pathlib.Path(__file__).parent / "runs"


class ReplayEnvironment(_environment.Environment):
    """Ends the episode when the agent runs out of targets; in a dry run, sends none."""

    def __init__(
        self,
        wrapped: XtacUmiBiFlexivRizon4RTEnvironment,
        agent: _replay_agent.UmiReplayAgent,
        *,
        dry_run: bool,
    ) -> None:
        self._wrapped = wrapped
        self._agent = agent
        self._dry_run = dry_run

    @override
    def reset(self) -> None:
        self._wrapped.reset()

    @override
    def is_episode_complete(self) -> bool:
        # A dry run has nothing to step through once the plan is built and reported.
        return self._dry_run or self._agent.done

    @override
    def get_observation(self) -> dict:
        return self._wrapped.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        if not self._dry_run:
            self._wrapped.apply_action(action)

    def disconnect(self) -> None:
        self._wrapped.disconnect()


@dataclass
class Args:
    """Arguments for replaying an XTac-UMI episode on BiFlexiv Rizon4 RT.

    Any of these can be preset in a run YAML under runs/ and selected with
    --args.run; flags still win over the file. See examples/run_config.py.
    """

    # Which run YAML to take the settings below from (a name under runs/, or a path).
    run: str | None = None

    # Bench recipe: a name under examples/bi_flexiv_rizon4_rt/recipes/ or a path to any
    # recipe YAML (e.g. lerobot-xense recipes/teleop/bi_flexiv_rizon4_rt/forward-04-taccap.yaml).
    robot_recipe: str | None = None

    # Episode to replay: a LeRobot v3 dataset with 20D or 29D per-side-grouped state/action.
    repo_id: str | None = None
    # Dataset directory; defaults to $HF_LEROBOT_HOME/<repo_id>.
    dataset_root: pathlib.Path | None = None
    # Episode index within the dataset.
    episode: int = 0

    # Pico-world -> arm-World yaw in degrees. None = estimate it from the two hands'
    # start fingertip headings; giving it also skips the start-pose consistency checks.
    yaw_deg: float | None = None
    # Playback speed relative to the recording, in (0, 1].
    speed_scale: float = 1.0
    # A single source frame stepping more than this marks a tracker relocalization.
    max_step_m: float = 0.1
    # Same, for the single-frame rotation step.
    max_step_deg: float = 30.0
    # Largest allowed disagreement between the two hands' yaw estimates.
    max_yaw_spread_deg: float = 15.0
    # Largest allowed distance of any target from its arm's start TCP.
    max_displacement_m: float = 0.6

    # Robot run tuning (applied on top of the recipe)
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    log_level: str = "INFO"

    # Where each run's planned and measured trajectory is written (.npz); None disables it.
    trajectory_dir: pathlib.Path | None = pathlib.Path("replay_logs")

    # Build and report the plan without sending any target. Pass --args.no-dry-run to move.
    dry_run: bool = True


def main(args: Args) -> None:
    logger.info(_run_config.describe(args, Args, RUNS_DIR))
    if args.robot_recipe is None:
        raise SystemExit(
            f"No bench selected. Pass --args.robot-recipe <name>. Recipes: {', '.join(_recipe.available_recipes())}."
        )
    if args.repo_id is None and args.dataset_root is None:
        raise SystemExit("No episode selected. Pass --args.repo-id (or --args.dataset-root) and --args.episode.")

    # Load the episode before touching the arms, so a bad path costs nothing.
    dataset_dir = _replay_agent.resolve_dataset_dir(args.repo_id or "", args.dataset_root)
    episode = _replay_agent.load_episode(dataset_dir, args.episode)
    logger.info(
        f"Episode {args.episode} of {dataset_dir}: {len(episode.actions)} frames, "
        f"{len(episode.actions) / episode.fps:.1f} s at {episode.fps:g} Hz"
    )
    agent = _replay_agent.UmiReplayAgent(
        episode,
        yaw_deg=args.yaw_deg,
        speed_scale=args.speed_scale,
        max_step_m=args.max_step_m,
        max_step_deg=args.max_step_deg,
        max_yaw_spread_deg=args.max_yaw_spread_deg,
        max_displacement_m=args.max_displacement_m,
    )

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
    if args.dry_run:
        logger.info("DRY RUN: the robot connects and resets, the plan is reported, no target is sent")

    environment = ReplayEnvironment(
        XtacUmiBiFlexivRizon4RTEnvironment(robot_config, setup_robot=True),
        agent,
        dry_run=args.dry_run,
    )
    # The plan has one target per source frame period, so the loop runs at the recording rate.
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=episode.fps,
        num_episodes=1,
    )

    def signal_handler(sig, frame) -> None:
        del sig, frame
        logger.info("Ctrl+C received; stopping runtime")
        runtime.request_stop()

    signal.signal(signal.SIGINT, signal_handler)
    try:
        runtime.run()
        if not args.dry_run:
            logger.info(agent.tracking_summary())
    finally:
        try:
            environment.disconnect()
        except Exception as error:
            logger.warning(f"Error while disconnecting: {error}")
        if args.trajectory_dir is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            agent.save_trajectory(args.trajectory_dir / f"episode{args.episode}_{stamp}.npz")


if __name__ == "__main__":
    main(_run_config.cli(main, Args, RUNS_DIR))
