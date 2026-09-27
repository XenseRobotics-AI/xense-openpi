"""Observation-only statistics and logging for online RLT (nothing here changes training).

``RunLogger`` logs to W&B on three independent axes - one row per round, per
gradient update and per executed chunk - each with its own step counter, and
mirrors every row to ``events.jsonl`` when a dump directory is set. Logging
failures are warned about once and never interrupt the robot loop.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import time
from typing import Any

import numpy as np

from openpi.rlt import action_space as _action_space

AXES = ("round", "update", "chunk")


class RunLogger:
    def __init__(self, run: Any | None, dump_dir: pathlib.Path | None = None):
        self._run = run
        self._dump_dir = dump_dir
        self._warned = False
        if run is not None:
            for axis in AXES:
                run.define_metric(f"{axis}/step")
                run.define_metric(f"{axis}/*", step_metric=f"{axis}/step")

    def log(self, axis: str, step: int, values: dict[str, Any]) -> None:
        try:
            row = {key: float(value) for key, value in values.items() if value is not None}
            if self._run is not None:
                self._run.log({**{f"{axis}/{k}": v for k, v in row.items()}, f"{axis}/step": step})
            if self._dump_dir is not None:
                with (self._dump_dir / "events.jsonl").open("a") as f:
                    f.write(json.dumps({"axis": axis, "step": step, "time": time.time(), **row}) + "\n")
        except Exception:
            if not self._warned:
                logging.warning("RLT metric logging failed; further failures are silent.", exc_info=True)
                self._warned = True


def _groups(space: _action_space.ActionSpace) -> dict[str, list[int]]:
    rotation = [i for a, b in space.rot6d_blocks for i in range(a, b)]
    position = [i for i in np.flatnonzero(space.delta_mask) if i not in rotation]
    return {"position": position, "rotation": rotation, "gripper": list(space.gripper_dims)}


def _rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    a = rot6d[..., :3] / np.linalg.norm(rot6d[..., :3], axis=-1, keepdims=True)
    b = rot6d[..., 3:] - np.sum(a * rot6d[..., 3:], axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def output_metrics(
    space: _action_space.ActionSpace, actions: np.ndarray, reference: np.ndarray, state: np.ndarray
) -> dict[str, float]:
    """An actor chunk against its VLA reference, both normalized ``(T, A)`` or ``(B, T, A)``.

    Per-group mean/std of the normalized output track drift and collapse; the residual
    to the reference is measured in execution space: position in mm, rotation in rad,
    gripper opening.
    """
    actions, reference, state = (np.asarray(x, np.float32) for x in (actions, reference, state))
    if actions.ndim == 2:
        actions, reference, state = actions[None], reference[None], state[None]
    metrics = {"nonfinite": float((~np.isfinite(actions)).sum())}
    groups = _groups(space)
    for name, dims in groups.items():
        metrics[f"{name}_mean"] = float(actions[..., dims].mean())
        metrics[f"{name}_std"] = float(actions[..., dims].std())
    actor = np.asarray(space.decode(actions, state))
    ref = np.asarray(space.decode(reference, state))
    residuals = {
        "position_mm": 1000
        * np.concatenate(
            [np.linalg.norm(actor[..., a : a + 3] - ref[..., a : a + 3], axis=-1).ravel() for a, _ in arm_blocks(space)]
        ),
        "rotation_rad": np.concatenate(
            [
                _angle(_rotation_matrix(actor[..., a:b]), _rotation_matrix(ref[..., a:b])).ravel()
                for a, b in space.rot6d_blocks
            ]
        ),
        "gripper": np.abs(actor[..., groups["gripper"]] - ref[..., groups["gripper"]]).ravel(),
    }
    for name, values in residuals.items():
        metrics[f"residual_{name}_mean"] = float(values.mean())
        metrics[f"residual_{name}_max"] = float(values.max())
    return metrics


def arm_blocks(space: _action_space.ActionSpace) -> list[tuple[int, int]]:
    """(xyz start, rot6d end) per arm."""
    return [(start - 3, end) for start, end in space.rot6d_blocks]


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = (np.einsum("...ij,...ij->...", a, b) - 1) / 2
    return np.arccos(np.clip(cos, -1.0, 1.0))


def discount_horizon(gamma: float, horizon: int) -> str:
    """How far a terminal reward reaches, in robot steps and chunks."""
    reach = math.log(0.01) / math.log(gamma)
    return (
        f"gamma={gamma:g} over C={horizon}: per-chunk discount {gamma**horizon:.3f}, "
        f"1% reach {reach:.0f} steps ({reach / horizon:.1f} chunks)"
    )


def warmup_estimate(warm_up: int, transitions: int, phases: int, phase_steps: int) -> str:
    """Labeled data still needed for warm_up, extrapolated from the phases committed so far."""
    if transitions <= 0 or phases <= 0:
        return "unknown until the first labeled phase is committed"
    scale = warm_up / transitions
    return f"~{scale * phases:.1f} labeled phases / {scale * phase_steps:.0f} phase steps in total"


def dump_transitions(path: pathlib.Path, rows: list[dict], **metadata: Any) -> None:
    """One round's committed transitions as an ``.npz`` (training never reads these).

    ``executed_actions`` are the absolute actions the robot ran; ``ref_exec`` is the raw VLA
    reference in execution space, before any normalization or clipping - the input for
    auditing action ranges against the VLA's q01/q99.
    """
    if not rows:
        return
    payload = {key: np.asarray(value) for key, value in metadata.items()}
    for key in (
        "executed_actions",
        "ref_exec",
        "actions",
        "chunk_rewards",
        "intervention_mask",
        "action_source",
        "terminated",
        "success",
        "actor_enabled",
        "episode_id",
        "round_id",
    ):
        payload[key] = np.stack([np.asarray(row[key]) for row in rows])
    for side in ("curr_obs", "next_obs"):
        for key in ("z_rl", "state", "proprio", "ref_chunk"):
            payload[f"{side}_{key}"] = np.stack([row[side][key] for row in rows])
    try:
        np.savez(path, **payload)
    except OSError:
        logging.warning("Transition dump to %s failed.", path, exc_info=True)
