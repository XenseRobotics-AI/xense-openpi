"""Replay one handheld XTac-UMI episode on a bimanual Flexiv, open loop.

No model and no policy server: the agent turns a recorded episode into arm targets
once, then hands them out one per control step.

The episode is composed onto the arms' start TCPs the same way
``rebase_umi_dataset.py`` re-bases training data (and the Pico teleop drives the
arms), with the episode's own first frame as the anchor::

    p_arm(t) = p_home + Rz · (p_umi(t) - p_umi(0))
    R_arm(t) = Rz · R_umi(t) · R_umi(0)ᵀ · Rzᵀ · R_home

so the first target is the pose the arm is already at. ``home`` is read from the
robot (the first observation after the reset), each arm in its own World frame.
The rotation delta is applied in the world frame, so the UMI/Flexiv gripper axis
conventions cancel and ``gripper_frame.align_gripper_frames`` must not run on top.

``Rz`` is the only unknown between the gravity-aligned Pico world and the arm
World frames (both z up; the two arm frames are assumed parallel). It is estimated
from the hands alone — the 9D headset pose of a 29D recording is ignored: the
operator starts an episode holding the grippers roughly the way the arms hold
theirs, so the yaw is whatever turns each hand's fingertip heading onto its arm's.
Both hands give an estimate; their mean is used and their disagreement is the
measure of how well that start-pose protocol was followed.

Gripper dims pass through unchanged (normalized, 0 closed .. 1 open).
"""

import dataclasses
import json
import os
import pathlib
import time
from typing import override

from lerobot.utils.robot_utils import get_logger
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from xense_client.runtime import agent as _agent

from examples.xtac_umi_bi_flexiv_rizon4_rt import gripper_frame
from examples.xtac_umi_bi_flexiv_rizon4_rt import rebase_umi_dataset as rebase

logger = get_logger("UmiReplayAgent")

HAND_DIM = rebase.HAND_DIM
_SIDES = (("left", rebase.LEFT_TCP), ("right", rebase.RIGHT_TCP))
_GRIPPER_DIMS = (9, 19)

# Which axis of each end frame runs along the fingertips: x for XTac-UMI data,
# z for the Flexiv driver (see gripper_frame.py).
_UMI_FINGERTIP_AXIS = 0
_FLEXIV_FINGERTIP_AXIS = 2
# Above this elevation the fingertip's horizontal heading is too ill-conditioned to align on.
_MAX_FINGERTIP_ELEVATION_DEG = 60.0
# A gripper target crossing this opening is logged as a grasp/release event.
_GRIPPER_EVENT_THRESHOLD = 0.6


@dataclasses.dataclass(frozen=True)
class Episode:
    """One recorded episode, hand dims only. ``actions[t]`` is the pose reached at frame t+1."""

    states: np.ndarray  # [T, 20]
    actions: np.ndarray  # [T, 20]
    fps: float


@dataclasses.dataclass(frozen=True)
class YawEstimate:
    """Pico-world -> arm-World yaw (rad) from each hand's fingertip heading."""

    left: float
    right: float
    mean: float
    spread: float  # |left - right|
    max_elevation_deg: float  # steepest fingertip among both hands and both arms


def resolve_dataset_dir(repo_id: str, dataset_root: pathlib.Path | None = None) -> pathlib.Path:
    """``dataset_root`` if given, else ``$HF_LEROBOT_HOME/<repo_id>``."""
    if dataset_root is not None:
        return pathlib.Path(dataset_root).expanduser()
    home = os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot")
    return pathlib.Path(home) / repo_id


def load_episode(dataset_dir: pathlib.Path, episode_index: int) -> Episode:
    """Read one episode's 20D hand state/action from a LeRobot v3 dataset on disk."""
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    columns = ["observation.state", "action", "frame_index"]
    frames = [
        pd.read_parquet(path, columns=columns, filters=[("episode_index", "==", episode_index)])
        for path in sorted((dataset_dir / "data").rglob("*.parquet"))
    ]
    data = pd.concat(frames).sort_values("frame_index")
    if data.empty:
        raise ValueError(f"Episode {episode_index} not found in {dataset_dir} ({info['total_episodes']} episodes)")
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(data["action"].to_numpy()).astype(np.float64)
    if states.shape[-1] not in (HAND_DIM, HAND_DIM + 9):
        raise ValueError(f"Expected 20D or 29D per-side-grouped state, got {states.shape[-1]}D")
    # 29D G1 recordings append the headset pose; the arms have no head to move.
    return Episode(states=states[:, :HAND_DIM], actions=actions[:, :HAND_DIM], fps=float(info["fps"]))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _heading_and_elevation(axis: np.ndarray) -> tuple[float, float]:
    """Heading about world z (rad) and elevation above the horizontal (deg) of a unit vector."""
    return float(np.arctan2(axis[1], axis[0])), float(np.degrees(np.arcsin(np.clip(abs(axis[2]), 0.0, 1.0))))


def estimate_yaw(umi_start: np.ndarray, home: np.ndarray) -> YawEstimate:
    """Yaw that turns each hand's start fingertip heading onto its arm's home fingertip heading."""
    yaws, elevations = [], []
    for _, tcp in _SIDES:
        umi_axis = gripper_frame.pose9_to_matrix(umi_start[tcp])[:3, _UMI_FINGERTIP_AXIS]
        arm_axis = gripper_frame.pose9_to_matrix(home[tcp])[:3, _FLEXIV_FINGERTIP_AXIS]
        umi_heading, umi_elevation = _heading_and_elevation(umi_axis)
        arm_heading, arm_elevation = _heading_and_elevation(arm_axis)
        yaws.append(_wrap(arm_heading - umi_heading))
        elevations += [umi_elevation, arm_elevation]
    left, right = yaws
    return YawEstimate(
        left=left,
        right=right,
        mean=_wrap(left + _wrap(right - left) / 2),
        spread=abs(_wrap(left - right)),
        max_elevation_deg=max(elevations),
    )


def _resample(poses: np.ndarray, speed_scale: float) -> np.ndarray:
    """Targets for each control step from ``[T+1, 20]`` poses at source frames 0..T.

    Step k aims at source time ``(k+1) * speed_scale``: positions and grippers
    interpolate linearly, orientations by slerp. The last source frame is always the
    last target.
    """
    last = len(poses) - 1
    if speed_scale == 1.0:
        return poses[1:]
    source_times = np.arange(len(poses), dtype=np.float64)
    times = np.arange(1, int(np.floor(last / speed_scale)) + 1) * speed_scale
    if not len(times) or times[-1] < last:
        times = np.append(times, last)

    out = np.stack([np.interp(times, source_times, poses[:, dim]) for dim in range(HAND_DIM)], axis=-1)
    for _, tcp in _SIDES:
        rotations = Rotation.from_matrix(gripper_frame.pose9_to_matrix(poses[:, tcp])[:, :3, :3])
        matrices = Slerp(source_times, rotations)(times).as_matrix()
        out[:, tcp.start + 3 : tcp.start + 6] = matrices[:, :, 0]
        out[:, tcp.start + 6 : tcp.start + 9] = matrices[:, :, 1]
    return out


def build_plan(episode: Episode, home: np.ndarray, yaw: float, speed_scale: float = 1.0) -> np.ndarray:
    """Arm-frame targets ``[N, 20]``, one per control step, for an episode started at ``home``.

    ``yaw`` rotates Pico-world vectors into the arm World frames. ``home`` is the 20D
    robot state at the start; only its TCP dims are used.
    """
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}")
    home = np.asarray(home, dtype=np.float64)
    start = episode.states[0]
    anchors = {side: gripper_frame.pose9_to_matrix(start[tcp]) for side, tcp in _SIDES}
    homes = {side: gripper_frame.pose9_to_matrix(home[tcp]) for side, tcp in _SIDES}
    # Source frames 0..T: the start state, then where each action lands.
    source = np.concatenate((start[None], episode.actions), axis=0)
    # rebase_poses takes the yaw to *remove* (it applies rot_z(-yaw)).
    poses = rebase.rebase_vectors(source, anchors, homes, -yaw)
    return _resample(poses, speed_scale)


class UmiReplayAgent(_agent.Agent):
    """Hands out a recorded XTac-UMI episode as Flexiv targets, one per step.

    The plan is built in :meth:`warmup`, from the observation the runtime takes right
    after the reset and before the control loop, so every check below fails before
    any target reaches the arms:

    * a tracker relocalization in the source episode (a single-frame step above
      ``max_step_m`` / ``max_step_deg``);
    * hands that did not start the way the arms do (the two yaw estimates disagree by
      more than ``max_yaw_spread_deg``, or a fingertip is too vertical to give one) —
      skipped when ``yaw_deg`` is given;
    * a target further than ``max_displacement_m`` from its arm's home.
    """

    def __init__(
        self,
        episode: Episode,
        *,
        yaw_deg: float | None = None,
        speed_scale: float = 1.0,
        max_step_m: float = 0.1,
        max_step_deg: float = 30.0,
        max_yaw_spread_deg: float = 15.0,
        max_displacement_m: float = 0.6,
    ) -> None:
        self._episode = episode
        self._yaw_deg = yaw_deg
        self._speed_scale = speed_scale
        self._max_step_m = max_step_m
        self._max_step_deg = max_step_deg
        self._max_yaw_spread_deg = max_yaw_spread_deg
        self._max_displacement_m = max_displacement_m
        self._plan: np.ndarray | None = None
        self._cursor = 0
        self._tracking_errors: list[np.ndarray] = []
        self._home: np.ndarray | None = None
        self._yaw: float = 0.0
        self._states: list[np.ndarray] = []
        self._times: list[float] = []

    @property
    def done(self) -> bool:
        """True once every target has been handed out."""
        return self._plan is not None and self._cursor >= len(self._plan)

    @override
    def reset(self) -> None:
        self._plan = None
        self._cursor = 0
        self._tracking_errors = []
        self._home = None
        self._states = []
        self._times = []

    @override
    def warmup(self, observation: dict) -> None:
        self._plan = self._build(np.asarray(observation["state"], dtype=np.float64))

    @override
    def get_action(self, observation: dict) -> dict:
        state = np.asarray(observation["state"], dtype=np.float64)
        if self._plan is None:
            self._plan = self._build(state)
        if self._cursor > 0:
            previous = self._plan[min(self._cursor, len(self._plan)) - 1]
            self._tracking_errors.append(
                np.array([np.linalg.norm(state[tcp][:3] - previous[tcp][:3]) for _, tcp in _SIDES])
            )
        # Past the end, hold the last target.
        index = min(self._cursor, len(self._plan) - 1)
        target = self._plan[index]
        self._states.append(state)
        self._times.append(time.time())
        self._log_gripper_events(index, state)
        self._cursor += 1
        return {"actions": target.astype(np.float32)}

    def _log_gripper_events(self, index: int, state: np.ndarray) -> None:
        """Log where each arm is, relative to its home, when its gripper target closes or opens.

        These are the moments the TCP should be at an object, so they are the points to
        compare against the scene when the replay lands off target.
        """
        if index == 0 or self._home is None:
            return
        for (side, tcp), dim in zip(_SIDES, _GRIPPER_DIMS, strict=True):
            before = self._plan[index - 1, dim] < _GRIPPER_EVENT_THRESHOLD
            after = self._plan[index, dim] < _GRIPPER_EVENT_THRESHOLD
            if before == after:
                continue
            planned = self._plan[index, tcp][:3] - self._home[tcp][:3]
            measured = state[tcp][:3] - self._home[tcp][:3]
            logger.info(
                f"{side} gripper {'CLOSE' if after else 'OPEN'} at step {index}: offset from home "
                f"planned {planned.round(3).tolist()} m, measured {measured.round(3).tolist()} m"
            )

    def save_trajectory(self, path: pathlib.Path) -> None:
        """Write what was planned and what the arms did to an ``.npz`` for offline analysis.

        ``states[k]`` is the robot state read just before ``plan[k]`` was sent, so
        ``states[k + 1]`` is the response to ``plan[k]``.
        """
        if self._plan is None or self._home is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            plan=self._plan,
            states=np.stack(self._states) if self._states else np.zeros((0, HAND_DIM)),
            times=np.asarray(self._times),
            home=self._home,
            yaw_deg=np.degrees(self._yaw),
            speed_scale=self._speed_scale,
            fps=self._episode.fps,
            umi_start=self._episode.states[0],
        )
        logger.info(f"Trajectory saved to {path} ({len(self._states)} of {len(self._plan)} steps executed)")

    def tracking_summary(self) -> str:
        """Distance between each sent target and the TCP one step later (includes the servo lag)."""
        if not self._tracking_errors:
            return "Tracking: no steps executed"
        errors = np.stack(self._tracking_errors) * 1000
        return "Tracking error, target vs TCP one step later: " + ", ".join(
            f"{side} mean {errors[:, i].mean():.1f} mm / max {errors[:, i].max():.1f} mm"
            for i, (side, _) in enumerate(_SIDES)
        )

    def _build(self, home: np.ndarray) -> np.ndarray:
        if home.shape != (HAND_DIM,):
            raise ValueError(f"Expected a 20D robot state as home, got {home.shape}")
        episode = self._episode
        source = np.concatenate((episode.states[:1], episode.actions), axis=0)
        for side, tcp in _SIDES:
            step_m, step_deg = rebase.max_frame_steps(source[:, tcp])
            if step_m > self._max_step_m or step_deg > self._max_step_deg:
                raise ValueError(
                    f"{side} tracker jumps {step_m:.3f} m / {step_deg:.1f} deg in one frame "
                    f"(limits {self._max_step_m} m / {self._max_step_deg} deg): relocalization, not replayable"
                )

        estimate = estimate_yaw(episode.states[0], home)
        logger.info(
            f"Yaw from fingertip headings: left {np.degrees(estimate.left):+.1f} deg, "
            f"right {np.degrees(estimate.right):+.1f} deg (spread {np.degrees(estimate.spread):.1f} deg, "
            f"steepest fingertip {estimate.max_elevation_deg:.0f} deg)"
        )
        if self._yaw_deg is not None:
            yaw = np.radians(self._yaw_deg)
            logger.info(f"Using the given yaw {self._yaw_deg:+.1f} deg")
        else:
            if estimate.max_elevation_deg > _MAX_FINGERTIP_ELEVATION_DEG:
                raise ValueError(
                    f"A fingertip is {estimate.max_elevation_deg:.0f} deg from horizontal, too steep to "
                    "take a heading from; pass the yaw explicitly"
                )
            if np.degrees(estimate.spread) > self._max_yaw_spread_deg:
                raise ValueError(
                    f"The two hands disagree on the yaw by {np.degrees(estimate.spread):.1f} deg "
                    f"(limit {self._max_yaw_spread_deg} deg): the episode did not start in the arms' "
                    "start pose. Pass the yaw explicitly to replay it anyway"
                )
            yaw = estimate.mean
            logger.info(f"Using the mean yaw {np.degrees(yaw):+.1f} deg")

        self._home, self._yaw = home, float(yaw)
        logger.info(
            "Home TCP (m): "
            + ", ".join(f"{side} {home[tcp][:3].round(4).tolist()}" for side, tcp in _SIDES)
            + f"; TCP-to-TCP distance at the UMI start {np.linalg.norm(episode.states[0][rebase.LEFT_TCP][:3] - episode.states[0][rebase.RIGHT_TCP][:3]):.3f} m"
        )
        plan = build_plan(episode, home, yaw, self._speed_scale)
        for side, tcp in _SIDES:
            offsets = plan[:, tcp][:, :3] - home[tcp][:3]
            reach = float(np.linalg.norm(offsets, axis=-1).max())
            step_m, step_deg = rebase.max_frame_steps(np.concatenate((home[None, tcp], plan[:, tcp])))
            logger.info(
                f"{side}: offset from home min {offsets.min(0).round(3).tolist()} "
                f"max {offsets.max(0).round(3).tolist()} m, reach {reach:.3f} m, "
                f"peak {step_m * episode.fps:.2f} m/s / {step_deg * episode.fps:.0f} deg/s"
            )
            if reach > self._max_displacement_m:
                raise ValueError(f"{side} target goes {reach:.3f} m from home (limit {self._max_displacement_m} m)")
        logger.info(
            f"Plan: {len(plan)} steps, {len(plan) / episode.fps:.1f} s at {episode.fps:g} Hz "
            f"(source {len(episode.actions) / episode.fps:.1f} s, speed x{self._speed_scale:g})"
        )
        return plan
