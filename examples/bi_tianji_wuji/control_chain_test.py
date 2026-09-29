#!/usr/bin/env python3
"""Small, removable Tianji/Wuji control-chain smoke test.

This file intentionally does not modify or import the OpenPI deployment loop.
It has two modes:

* ``record`` connects the normal Manus + BiPico4 teleoperator and saves only
  robot state and the complete action target (no camera frames).
* ``replay`` loads that file and sends the same 58-D absolute action through
  ``TianjiArmWuji.send_action``.  Before replay, the current robot state is
  linearly aligned to the first recorded target, which avoids a jump when the
  arm was left in another pose.

Example (run from this repository):

    python examples/bi_tianji_wuji/control_chain_test.py record \
      --robot-recipe examples/bi_tianji_wuji/recipes/block-sort.yaml \
      --trajectory /tmp/tianji_wuji_demo.npz --duration 20

    python examples/bi_tianji_wuji/control_chain_test.py replay \
      --robot-recipe examples/bi_tianji_wuji/recipes/block-sort.yaml \
      --trajectory /tmp/tianji_wuji_demo.npz

The recipe and hardware imports are lazy, so ``--help`` and trajectory-file
inspection work on a machine without the robot SDK installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# Keep this schema local.  The script is deliberately removable and does not
# become a dependency of the formal inference path.
ACTION_KEYS = tuple(
    [f"{side}_tcp.{axis}" for side in ("left", "right") for axis in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")]
    + [
        f"{side}_{joint}.pos"
        for side in ("l", "r")
        for joint in (
            "index_finger_mcp_flex", "index_finger_mcp_abd", "index_finger_pip", "index_finger_dip",
            "middle_finger_mcp_flex", "middle_finger_mcp_abd", "middle_finger_pip", "middle_finger_dip",
            "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip",
            "ring_finger_mcp_flex", "ring_finger_mcp_abd", "ring_finger_pip", "ring_finger_dip",
            "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
        )
    ]
)
ARM_FEEDBACK_KEYS = tuple(
    f"{side}_joint_{joint}.{kind}"
    for side in ("left", "right")
    for joint in range(1, 8)
    for kind in ("pos", "vel")
)
TCP_SLICES = (slice(3, 9), slice(12, 18))
FORMAT_VERSION = 1


def _ensure_sibling_lerobot_on_path() -> None:
    """Make the sibling hardware checkout importable without installation."""
    sibling_src = Path(__file__).resolve().parents[2] / ".." / "lerobot-xensehand" / "src"
    sibling_src = sibling_src.resolve()
    if sibling_src.is_dir() and str(sibling_src) not in sys.path:
        sys.path.insert(0, str(sibling_src))


def _resolve_recipe(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = Path(__file__).resolve().parent / "recipes" / value
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix != ".yaml" and candidate.with_suffix(".yaml").is_file():
        return candidate.with_suffix(".yaml").resolve()
    raise FileNotFoundError(f"robot recipe does not exist: {value}")


def _load_robot_config(recipe: str) -> Any:
    """Decode only the robot block, keeping this utility independent of OpenPI."""
    _ensure_sibling_lerobot_on_path()
    import draccus
    import yaml
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.tianji_arm_wuji import TianjiArmWujiConfig

    path = _resolve_recipe(recipe)
    raw = yaml.safe_load(path.read_text())
    block = raw.get("robot") if isinstance(raw, dict) else None
    if not isinstance(block, dict) or block.get("type") != "tianji_arm_wuji":
        raise ValueError(f"{path} must contain robot.type: tianji_arm_wuji")
    block = dict(block)
    if block.get("wuji") is not None:
        raise ValueError("Use flat wuji_* fields, not a nested wuji block")
    # Lifecycle is explicit in this script; connecting must not unexpectedly
    # home the arms, and teardown must not open the hands.
    block.update(go_home_on_connect=False, wuji_return_to_zero_on_disconnect=False)
    config = draccus.decode(RobotConfig, block)
    if not isinstance(config, TianjiArmWujiConfig):
        raise TypeError("Expected TianjiArmWujiConfig")
    if not config.connect_wuji or config.wuji_hand_type != "both":
        raise ValueError("The 58-D test requires both Wuji hands connected")
    return config


def _make_robot(args: argparse.Namespace) -> Any:
    """Construct only the composite robot; replay never imports glove code."""
    _ensure_sibling_lerobot_on_path()
    from lerobot.robots.tianji_arm_wuji import TianjiArmWuji

    return TianjiArmWuji(_load_robot_config(args.robot_recipe))


def _make_hardware(args: argparse.Namespace) -> tuple[Any, Any]:
    """Construct the robot and the normal sibling teleoperator for recording."""
    _ensure_sibling_lerobot_on_path()
    from lerobot.teleoperators.manus_bi_pico4 import ManusBiPico4
    from lerobot.teleoperators.manus_bi_pico4.config_manus_bi_pico4 import ManusBiPico4Config

    robot = _make_robot(args)
    teleop = ManusBiPico4(
        ManusBiPico4Config(
            manus_hand_type=args.manus_hand_type,
            manus_calibration_prefix=args.manus_calibration_prefix,
            manus_mode=args.manus_mode,
            pos_sensitivity=args.pos_sensitivity,
            ori_sensitivity=args.ori_sensitivity,
        )
    )
    return robot, teleop


def _as_action_vector(values: dict[str, Any], fallback: np.ndarray | None = None) -> np.ndarray:
    result = fallback.copy() if fallback is not None else np.zeros(len(ACTION_KEYS), dtype=np.float64)
    for index, key in enumerate(ACTION_KEYS):
        if key in values:
            result[index] = float(np.asarray(values[key]).reshape(()))
    return result


def _observation_vector(observation: dict[str, Any]) -> np.ndarray:
    missing = [key for key in ACTION_KEYS if key not in observation]
    if missing:
        raise RuntimeError(f"robot observation is missing {len(missing)} action fields; first={missing[:3]}")
    values = np.asarray([float(observation[key]) for key in ACTION_KEYS], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("robot observation contains non-finite action fields")
    return values


def _feedback_vector(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(observation.get(key, np.nan)) for key in ARM_FEEDBACK_KEYS], dtype=np.float64)


def _action_dict(vector: np.ndarray) -> dict[str, float]:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (len(ACTION_KEYS),) or not np.all(np.isfinite(vector)):
        raise ValueError("action must be a finite 58-D vector")
    for rotation_slice in TCP_SLICES:
        rotation = vector[rotation_slice]
        if np.linalg.norm(rotation[:3]) < 1e-8 or np.linalg.norm(np.cross(rotation[:3], rotation[3:])) < 1e-8:
            raise ValueError("TCP rotation-6D is degenerate")
    return dict(zip(ACTION_KEYS, vector.tolist(), strict=True))


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


def _validate_trajectory(actions: np.ndarray) -> None:
    if actions.ndim != 2 or actions.shape[1] != len(ACTION_KEYS) or actions.shape[0] == 0:
        raise ValueError(f"trajectory actions must have shape (N, {len(ACTION_KEYS)}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("trajectory contains non-finite actions")
    for rotation_slice in TCP_SLICES:
        rotation = actions[:, rotation_slice]
        if np.any(np.linalg.norm(rotation[:, :3], axis=1) < 1e-8) or np.any(
            np.linalg.norm(np.cross(rotation[:, :3], rotation[:, 3:]), axis=1) < 1e-8
        ):
            raise ValueError("trajectory contains degenerate TCP rotation-6D")


def _save_trajectory(path: Path, timestamps: list[float], states: list[np.ndarray], actions: list[np.ndarray], feedback: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        arm_feedback=np.asarray(feedback, dtype=np.float64),
        metadata=np.asarray(
            json.dumps({"format_version": FORMAT_VERSION, "action_keys": ACTION_KEYS, "feedback_keys": ARM_FEEDBACK_KEYS}),
        ),
    )


def _load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("format_version") != FORMAT_VERSION or tuple(metadata.get("action_keys", ())) != ACTION_KEYS:
            raise ValueError("trajectory format/schema does not match this script")
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        states = np.asarray(data["states"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)
        feedback = np.asarray(data["arm_feedback"], dtype=np.float64)
    if actions.shape[0] != timestamps.shape[0] or states.shape != actions.shape:
        raise ValueError("trajectory arrays have inconsistent lengths/shapes")
    if feedback.shape != (actions.shape[0], len(ARM_FEEDBACK_KEYS)):
        raise ValueError("trajectory arm_feedback has an unexpected shape")
    _validate_trajectory(actions)
    return timestamps, states, actions, {"feedback": feedback}


def _prepare_action(raw: dict[str, Any], teleop: Any, robot: Any, lifecycle: Any, control_state: Any, logger: logging.Logger, dry_run: bool) -> tuple[dict[str, Any] | np.ndarray, bool]:
    """Apply the sibling Tianji/Wuji reset/enable/freshness rules."""
    reset_sides = lifecycle.get_requested_reset_sides(teleop, robot)
    if reset_sides:
        if not dry_run:
            lifecycle.start_robot_reset(robot, reset_sides)
            lifecycle.wait_for_reset_settle(robot, timeout_s=6.0, logger=logger)
            lifecycle.sync_teleop_to_robot(teleop, robot, reset_sides)
        control_state.reset_enable_tracking(reset_sides)
        return _observation_vector(robot.get_observation()), True
    prepared = control_state.prepare_enabled_action(raw, teleop, robot, logger)
    prepared, _, _ = lifecycle.prepare_source_action(prepared, teleop, robot)
    # Missing fields deliberately mean hold.  The caller fills them from the
    # previous complete command before sending/saving.
    return prepared, False


def _record(args: argparse.Namespace) -> None:
    _ensure_sibling_lerobot_on_path()
    from lerobot.robots.tianji_arm_wuji.control_lifecycle import TianjiWujiControlState
    from lerobot.robots.tianji_arm_wuji import control_lifecycle

    logger = logging.getLogger("tianji_wuji_control_test")
    robot, teleop = _make_hardware(args)
    timestamps: list[float] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    feedback: list[np.ndarray] = []
    control_state = TianjiWujiControlState()
    last_command: np.ndarray | None = None
    period = 1.0 / args.hz
    robot_connected = False
    teleop_connected = False
    try:
        robot.connect(calibrate=False)
        robot_connected = True
        left_pose, right_pose = robot.get_current_tcp_pose_quat()
        teleop.connect(calibrate=False, left_tcp_pose_quat=left_pose, right_tcp_pose_quat=right_pose)
        teleop_connected = True
        last_command = _observation_vector(robot.get_observation())
        # Do not count device connection and glove startup against the requested
        # trajectory duration.
        started = time.perf_counter()
        next_tick = started
        logger.info("recording for %.1fs at %.1f Hz; hold the enable pedal to move", args.duration, args.hz)
        while time.perf_counter() - started < args.duration:
            now = time.perf_counter()
            observation = robot.get_observation()
            raw = teleop.get_action()
            requested, reset = _prepare_action(raw, teleop, robot, control_lifecycle, control_state, logger, args.dry_run)
            if reset:
                # The reset may have taken several seconds; pair its hold
                # command with the post-reset measured state.
                observation = robot.get_observation()
            # Missing fields deliberately mean hold. Merge them over the
            # previous complete command; explicit zero-valued finger targets
            # remain valid because presence, rather than truthiness, is used.
            command = np.asarray(requested, dtype=np.float64) if reset else _as_action_vector(requested, last_command)
            if not args.dry_run:
                robot.send_action(_action_dict(command))
            last_command = command
            timestamps.append(now - started)
            states.append(_observation_vector(observation))
            actions.append(command.copy())
            feedback.append(_feedback_vector(observation))
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    finally:
        if teleop_connected:
            teleop.disconnect()
        if robot_connected:
            robot.disconnect()
    _save_trajectory(args.trajectory, timestamps, states, actions, feedback)
    logger.info("saved %d samples to %s", len(actions), args.trajectory)


def _replay(args: argparse.Namespace) -> None:
    logger = logging.getLogger("tianji_wuji_control_test")
    timestamps, _, actions, _ = _load_trajectory(args.trajectory)
    robot = _make_robot(args)
    try:
        robot.connect(calibrate=False)
        current = _observation_vector(robot.get_observation())
        align_duration = max(0.0, args.align_duration)
        if align_duration > 0:
            logger.info("aligning current pose to trajectory start over %.2fs", align_duration)
            _send_interpolated(robot, current, actions[0], align_duration, args.hz, args.dry_run)
        logger.info("replaying %d samples at %.1f Hz", len(actions), args.hz)
        period = 1.0 / args.hz
        next_tick = time.perf_counter()
        for action in actions:
            if not args.dry_run:
                robot.send_action(_action_dict(action))
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    finally:
        robot.disconnect()
    logger.info("replay complete (recorded duration %.2fs)", float(timestamps[-1] if len(timestamps) else 0.0))


def _send_interpolated(robot: Any, start: np.ndarray, target: np.ndarray, duration: float, hz: float, dry_run: bool) -> None:
    period = 1.0 / hz
    count = max(1, int(np.ceil(duration * hz)))
    started = time.perf_counter()
    next_tick = started
    for index in range(1, count + 1):
        amount = index / count
        action = _interpolate(start, target, amount)
        if not dry_run:
            robot.send_action(_action_dict(action))
        next_tick += period
        time.sleep(max(0.0, next_tick - time.perf_counter()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("record", "replay"):
        command = sub.add_parser(name, help="record teleoperation" if name == "record" else "replay a saved action trajectory")
        command.add_argument("--robot-recipe", required=True, help="YAML path, or filename under examples/bi_tianji_wuji/recipes")
        command.add_argument("--trajectory", type=Path, required=True, help=".npz trajectory file")
        command.add_argument("--hz", type=float, default=30.0)
        command.add_argument("--dry-run", action="store_true", help="connect/read but do not send robot commands")
        command.add_argument("--manus-hand-type", choices=("both", "left", "right", "none"), default="both")
        command.add_argument("--manus-calibration-prefix", default="", help="Manus calibration filename prefix")
        command.add_argument("--manus-mode", choices=("integrated", "local", "remote"), default="integrated")
        command.add_argument("--pos-sensitivity", type=float, default=1.0)
        command.add_argument("--ori-sensitivity", type=float, default=1.0)
        if name == "record":
            command.add_argument("--duration", type=float, default=20.0)
        else:
            command.add_argument("--align-duration", type=float, default=3.0, help="seconds to interpolate to the first target")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args()
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")
    if args.mode == "record" and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.mode == "replay" and args.align_duration < 0:
        raise SystemExit("--align-duration cannot be negative")
    if args.mode == "record":
        _record(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
