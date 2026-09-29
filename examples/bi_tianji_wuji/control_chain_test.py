#!/usr/bin/env python3
"""Replay an absolute 58D NPZ trajectory through the production policy loop.

Collect in lerobot-xensehand with examples/tianji_wuji/record_trajectory.py.
Run from xense-openpi with:
    python -m examples.bi_tianji_wuji.control_chain_test \
      --robot-recipe block-sort \
      --trajectory examples/bi_tianji_wuji/tmp/tianji_wuji_demo.npz

No teleoperator is imported. --dry-run still connects position-holding drivers.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

# Support both direct script invocation and python -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bi_tianji_wuji.schema import ACTION_KEYS, validate_actions

# Rotation-6D fields within left/right 9D TCP targets.
TCP_SLICES = (slice(3, 9), slice(12, 18))


def load_trajectory(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("format_version") != 1 or tuple(metadata.get("action_keys", ())) != ACTION_KEYS:
            raise ValueError("NPZ must use format_version=1 and the exact Tianji/Wuji action_keys order")
        actions = validate_actions(data["actions"]).copy()
    return actions, metadata


class TrajectoryPolicy:
    """Supply consecutive recorded targets using the real policy interface."""

    def __init__(self, actions: np.ndarray, horizon: int, delay_s: float = 0):
        self.actions = validate_actions(actions)
        if not 1 <= horizon <= 50 or not np.isfinite(delay_s) or delay_s < 0:
            raise ValueError("horizon must be 1..50 and delay_s must be finite and nonnegative")
        self.horizon = horizon
        self.delay_s = delay_s
        self.reset()

    def reset(self):
        self.index = 0

    def infer(self, observation):
        if self.index >= len(self.actions):
            raise RuntimeError("Trajectory exhausted")
        time.sleep(self.delay_s)
        chunk = self.actions[self.index:self.index + self.horizon]
        self.index += len(chunk)
        # run_episode validates a whole horizon, but max_episode_steps ensures
        # padded targets in the final chunk are never executed.
        if len(chunk) < self.horizon:
            chunk = np.concatenate((chunk, np.repeat(chunk[-1:], self.horizon - len(chunk), axis=0)))
        return {"actions": chunk}


class FeedbackTrace:
    """Wrap the production environment without changing command dispatch."""

    def __init__(self, environment):
        self.environment = environment
        self.rows = []
        self.observation = None
        self.started = time.monotonic()

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def get_observation(self, prompt):
        observation = self.environment.get_observation(prompt)
        self.observation = (time.monotonic() - self.started, observation["state"].copy())
        return observation

    def apply_action(self, action):
        self.environment.apply_action(action)
        timestamp, state = self.observation
        self.rows.append((timestamp, time.monotonic() - self.started, state, np.asarray(action).copy()))

    def save(self, path, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            timestamps=np.asarray([r[0] for r in self.rows]),
            command_timestamps=np.asarray([r[1] for r in self.rows]),
            states=np.asarray([r[2] for r in self.rows]).reshape(-1, 58),
            actions=np.asarray([r[3] for r in self.rows]).reshape(-1, 58),
            metadata=np.asarray(json.dumps(metadata)),
        )


def _normalise_rotation6d(rotation: np.ndarray) -> np.ndarray:
    """Project two interpolated columns back to the valid 6-D rotation form."""
    first = np.asarray(rotation[:3], dtype=np.float64)
    second = np.asarray(rotation[3:], dtype=np.float64)
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = second - first * float(np.dot(first, second))
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        basis = np.array([0.0, 1.0, 0.0]) if abs(first[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        second = basis - first * float(np.dot(first, basis))
        second /= np.linalg.norm(second)
    else:
        second /= second_norm
    return np.concatenate((first, second))


def _interpolate(start: np.ndarray, target: np.ndarray, amount: float) -> np.ndarray:
    value = start + float(np.clip(amount, 0.0, 1.0)) * (target - start)
    for rotation_slice in TCP_SLICES:
        value[rotation_slice] = _normalise_rotation6d(value[rotation_slice])
    return value


def replay(args):
    # Import exactly the deployment environment/recipe/loop, using the same
    # installed lerobot checkout as production (no sibling sys.path override).
    from examples.bi_tianji_wuji.env import TianjiWujiEnvironment
    from examples.bi_tianji_wuji.main import Args, run_episode
    from examples.bi_tianji_wuji.recipe import load_robot_config
    from lerobot.robots.tianji_arm_wuji import TianjiArmWuji

    actions, metadata = load_trajectory(args.trajectory)
    policy = TrajectoryPolicy(actions, args.action_horizon, args.inference_delay)
    run_args = Args(robot_recipe=args.robot_recipe, runtime_hz=args.hz,
                    action_horizon=args.action_horizon, max_episode_steps=len(actions),
                    reset_on_start=False, dry_run=args.dry_run)
    run_args.validate()
    config = load_robot_config(args.robot_recipe)
    if 1 / args.hz >= config.command_timeout_s:
        raise ValueError("Replay frequency is too low for command_timeout_s")
    environment = TianjiWujiEnvironment(TianjiArmWuji(config), dry_run=args.dry_run)
    trace = FeedbackTrace(environment)
    metadata = {**metadata, "kind": "replay_feedback", "source": str(args.trajectory.resolve()),
                "runtime_hz": args.hz, "action_horizon": args.action_horizon,
                "inference_delay_s": args.inference_delay, "dry_run": args.dry_run,
                "state_semantics": "measured before action", "completed": False,
                "robot_recipe": str(args.robot_recipe), "align_duration_s": args.align_duration}
    try:
        environment.connect()
        if args.align_duration > 0:
            start = environment.get_observation("")["state"]
            count = max(1, int(np.ceil(args.align_duration * args.hz)))
            for index in range(1, count + 1):
                tick = time.monotonic()
                environment.apply_action(_interpolate(start, actions[0], index / count))
                time.sleep(max(0, 1 / args.hz - (time.monotonic() - tick)))
        trace.started = time.monotonic()
        run_episode(trace, policy, run_args)
        metadata["completed"] = True
    finally:
        try:
            environment.hold()
        finally:
            try:
                trace.save(args.feedback, metadata)
                logging.info("Saved %d feedback samples to %s", len(trace.rows), args.feedback)
            finally:
                environment.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("replay",), default="replay",
                        help="optional compatibility argument; recording moved to lerobot-xensehand")
    parser.add_argument("--robot-recipe", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, help="default: TRAJECTORY.replay.npz")
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--inference-delay", type=float, default=0,
                        help="seconds to simulate waiting for each model chunk")
    parser.add_argument("--align-duration", type=float, default=3,
                        help="seconds to approach the first target; 0 requires manual alignment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.align_duration) or args.align_duration < 0:
        parser.error("--align-duration must be finite and nonnegative")
    args.feedback = args.feedback or args.trajectory.with_suffix(".replay.npz")
    if args.feedback.suffix != ".npz":
        parser.error("--feedback must end with .npz")
    if args.feedback.resolve() == args.trajectory.resolve():
        parser.error("--feedback must differ from --trajectory")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    replay(args)


if __name__ == "__main__":
    main()
