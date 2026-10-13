"""Re-base an XTac-UMI dataset onto a bimanual Flexiv's start TCPs (offline teleop).

Handheld UMI poses live in the Pico4 world frame, whose origin is wherever the
headset was when the Unity app started. This script replays them the way
``lerobot-xense``'s Pico teleop drives the arms, with the hand as the controller::

    p_arm(t) = p_home + Rz · (p_umi(t) - p_anchor)
    R_arm(t) = Rz · R_umi(t) · R_anchorᵀ · Rzᵀ · R_home

Each arm uses its own home pose, in its own World frame, so no extrinsic between the two arm bases is needed.
``R_home`` is in the Flexiv TCP convention, so the output already is too. Deploy it
with ``--args.no-align-gripper-frames``; ``gripper_frame.py`` must not run on top.

The anchor is per *session*, not per episode. Within one Unity session the Pico
frame is fixed to the scene: the table-contact height repeats to ~4 mm. So one anchor
keeps each episode's real start offset relative to the scene. Re-anchoring every
episode would erase that offset and twist each episode by its own start-orientation
error. The anchor of each session and each hand is the mean start pose of that session.
Sessions are split where the episode start jumps, which is where the Unity origin moved.

``Rz`` removes the session's mean headset yaw at episode start (29D G1 recordings
only). After that, "forward for the operator" maps to the arm's +x.

Episodes in which a tracker relocalizes are dropped. A relocalization is a
single-frame step larger than ``max_step_m`` / ``max_step_deg`` that does not come back.

Output: a new LeRobot v3 dataset. state/action keep the per-side-grouped 20D layout
that ``LeRobotXtacUmiDataConfig`` reads, and the headset pose is dropped. Videos are hard-linked.
Provenance (homes, sessions, anchors, dropped episodes) goes to ``meta/rebase.json``.

    python -m examples.xtac_umi_bi_flexiv_rizon4_rt.rebase_umi_dataset \\
        --src-repo-id Xense/xtac-umi-g1-clean-plates-with-a-sponge \\
        --dst-repo-id Xense/xtac-umi-g1-clean-plates-with-a-sponge-fwd04 \\
        --left-home  x y z qw qx qy qz \\
        --right-home x y z qw qx qy qz

Home poses are the Flexiv ``tcp_pose`` (``[x, y, z, qw, qx, qy, qz]``, arm World frame)
read with the arm at the recipe's start joints.
"""

import dataclasses
import json
import logging
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import tyro

from examples.xtac_umi_bi_flexiv_rizon4_rt import gripper_frame

logger = logging.getLogger(__name__)

# Per-side-grouped source layout (see openpi.policies.xtac_umi_policy).
LEFT_TCP = slice(0, 9)
RIGHT_TCP = slice(10, 19)
HEAD_POSE = slice(20, 29)
HAND_DIM = 20


@dataclasses.dataclass
class Args:
    src_repo_id: str = "Xense/xtac-umi-g1-clean-plates-with-a-sponge"
    dst_repo_id: str = "Xense/xtac-umi-g1-clean-plates-with-a-sponge-fwd04"
    # Arm start TCPs, Flexiv tcp_pose order [x, y, z, qw, qx, qy, qz], each in its arm's World frame.
    left_home: tuple[float, float, float, float, float, float, float] | None = None
    right_home: tuple[float, float, float, float, float, float, float] | None = None
    # Dataset root containing org/name dirs; defaults to $HF_LEROBOT_HOME (~/.cache/huggingface/lerobot).
    lerobot_home: pathlib.Path | None = None
    # Explicit session start episodes (e.g. 0 15). Default: detect from start-position jumps.
    session_starts: tuple[int, ...] | None = None
    # A new session starts where the hands' mean start position moves more than this (m).
    session_jump_m: float = 0.15
    # Undo the session's mean headset yaw (needs the 29D headset pose).
    yaw_from_head: bool = True
    # A single-frame step above either threshold (either hand) marks a relocalization.
    max_step_m: float = 0.1
    max_step_deg: float = 30.0
    overwrite: bool = False


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def home_to_matrix(home: tuple[float, ...]) -> np.ndarray:
    """Flexiv ``[x, y, z, qw, qx, qy, qz]`` -> 4x4."""
    matrix = np.eye(4)
    matrix[:3, :3] = quat_wxyz_to_matrix(np.asarray(home[3:7]))
    matrix[:3, 3] = home[:3]
    return matrix


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal L2 mean: project the summed matrices back onto SO(3)."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def heading(rotation: np.ndarray) -> float:
    """Yaw of the frame's x axis about world z."""
    return float(np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0]))


def detect_sessions(start_points: np.ndarray, jump_m: float) -> np.ndarray:
    """Session id per episode; a new session wherever consecutive starts jump > jump_m."""
    steps = np.linalg.norm(np.diff(start_points, axis=0), axis=-1)
    return np.concatenate(([0], np.cumsum(steps > jump_m)))


def max_frame_steps(poses: np.ndarray) -> tuple[float, float]:
    """Largest per-frame translation (m) and rotation (deg) step of a [T, 9] pose track."""
    if len(poses) < 2:
        return 0.0, 0.0
    matrices = gripper_frame.pose9_to_matrix(poses.astype(np.float64))
    translation = np.linalg.norm(np.diff(matrices[:, :3, 3], axis=0), axis=-1)
    relative = np.einsum("tji,tjk->tik", matrices[:-1, :3, :3], matrices[1:, :3, :3])
    cos = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    return float(translation.max()), float(np.degrees(np.arccos(cos)).max())


def rebase_poses(poses: np.ndarray, anchor: np.ndarray, home: np.ndarray, yaw: float) -> np.ndarray:
    """Apply the module-docstring formula to ``[..., 9]`` poses. ``anchor``/``home`` are 4x4."""
    matrices = gripper_frame.pose9_to_matrix(poses.astype(np.float64))
    rz = rot_z(-yaw)
    out = np.zeros_like(matrices)
    out[..., :3, 3] = home[:3, 3] + (matrices[..., :3, 3] - anchor[:3, 3]) @ rz.T
    delta = rz @ matrices[..., :3, :3] @ anchor[:3, :3].T @ rz.T
    out[..., :3, :3] = delta @ home[:3, :3]
    out[..., 3, 3] = 1.0
    return gripper_frame.matrix_to_pose9(out)


def rebase_vectors(vectors: np.ndarray, anchors: dict, homes: dict, yaw: float) -> np.ndarray:
    """Per-side-grouped [..., >=20] -> rebased [..., 20]; gripper dims pass through."""
    out = np.array(vectors[..., :HAND_DIM], dtype=np.float64)
    for side, tcp in (("left", LEFT_TCP), ("right", RIGHT_TCP)):
        out[..., tcp] = rebase_poses(vectors[..., tcp], anchors[side], homes[side], yaw)
    return out


def _stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def _link_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Hard-link every file (videos are GBs); fall back to copying across filesystems."""
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def main(args: Args) -> None:
    if args.left_home is None or args.right_home is None:
        raise SystemExit("--left-home and --right-home are required (Flexiv tcp_pose: x y z qw qx qy qz).")
    home_root = args.lerobot_home or pathlib.Path(
        os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot")
    )
    src, dst = home_root / args.src_repo_id, home_root / args.dst_repo_id
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to replace it.")
        shutil.rmtree(dst)
    homes = {"left": home_to_matrix(args.left_home), "right": home_to_matrix(args.right_home)}

    info = json.loads((src / "meta/info.json").read_text())
    data = pd.concat([pd.read_parquet(p) for p in sorted((src / "data").rglob("*.parquet"))])
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(data["action"].to_numpy()).astype(np.float64)
    if states.shape[-1] not in (HAND_DIM, HAND_DIM + 9):
        raise SystemExit(f"Expected 20D or 29D per-side-grouped state, got {states.shape[-1]}D")
    has_head = states.shape[-1] == HAND_DIM + 9
    episode_ids = np.unique(data["episode_index"].to_numpy())
    rows = {e: np.flatnonzero(data["episode_index"].to_numpy() == e) for e in episode_ids}

    # Relocalizations: drop the whole episode, the pose after the step is not trustworthy.
    dropped = {}
    for e in episode_ids:
        for side, tcp in (("left", LEFT_TCP), ("right", RIGHT_TCP)):
            step_m, step_deg = max_frame_steps(states[rows[e], tcp])
            if step_m > args.max_step_m or step_deg > args.max_step_deg:
                dropped[int(e)] = f"{side} step {step_m:.3f} m / {step_deg:.1f} deg"
                break
    for e, why in dropped.items():
        logger.warning("Dropping episode %d: %s", e, why)
    kept = np.array([e for e in episode_ids if int(e) not in dropped])

    # Sessions are detected over all episodes: a dropped one still sits in the recording order.
    starts = np.stack([states[rows[e][0]] for e in episode_ids])
    if args.session_starts is not None:
        session_of = np.searchsorted(np.asarray(args.session_starts), episode_ids, side="right") - 1
    else:
        midpoints = (starts[:, LEFT_TCP][:, :3] + starts[:, RIGHT_TCP][:, :3]) / 2
        session_of = detect_sessions(midpoints, args.session_jump_m)
    session_by_episode = dict(zip(episode_ids.tolist(), session_of.tolist(), strict=True))

    sessions = {}
    for session in np.unique(session_of):
        members = [e for e in kept if session_by_episode[int(e)] == session]
        if not members:
            continue
        first = np.stack([states[rows[e][0]] for e in members])
        anchors = {}
        for side, tcp in (("left", LEFT_TCP), ("right", RIGHT_TCP)):
            matrices = gripper_frame.pose9_to_matrix(first[:, tcp])
            anchor = np.eye(4)
            anchor[:3, :3] = mean_rotation(matrices[:, :3, :3])
            anchor[:3, 3] = matrices[:, :3, 3].mean(0)
            anchors[side] = anchor
        yaw = 0.0
        if args.yaw_from_head and has_head:
            head = gripper_frame.pose9_to_matrix(first[:, HEAD_POSE])
            yaw = heading(mean_rotation(head[:, :3, :3]))
        spread = {
            side: np.linalg.norm(gripper_frame.pose9_to_matrix(first[:, tcp])[:, :3, 3] - anchors[side][:3, 3], axis=-1)
            for side, tcp in (("left", LEFT_TCP), ("right", RIGHT_TCP))
        }
        sessions[int(session)] = {
            "episodes": [int(e) for e in members],
            "yaw_deg": float(np.degrees(yaw)),
            "anchors": {side: m.tolist() for side, m in anchors.items()},
            "start_dist_to_anchor_m_median": {side: float(np.median(v)) for side, v in spread.items()},
            "start_dist_to_anchor_m_max": {side: float(np.max(v)) for side, v in spread.items()},
        }
        logger.info(
            "Session %d: episodes %d-%d (%d kept), yaw %.1f deg, start-to-anchor median L %.3f R %.3f m",
            session,
            members[0],
            members[-1],
            len(members),
            np.degrees(yaw),
            np.median(spread["left"]),
            np.median(spread["right"]),
        )
        for e in members:
            idx = rows[e]
            states[idx, :HAND_DIM] = rebase_vectors(states[idx], anchors, homes, yaw)
            actions[idx, :HAND_DIM] = rebase_vectors(actions[idx], anchors, homes, yaw)

    # ---- write the new dataset
    renumber = {int(e): i for i, e in enumerate(kept)}
    out = data[data["episode_index"].isin(kept)].copy()
    keep_rows = out.index.to_numpy()
    out["observation.state"] = list(states[keep_rows, :HAND_DIM].astype(np.float32))
    out["action"] = list(actions[keep_rows, :HAND_DIM].astype(np.float32))
    out["episode_index"] = out["episode_index"].map(renumber)
    out["index"] = np.arange(len(out))
    (dst / "data/chunk-000").mkdir(parents=True)
    out.reset_index(drop=True).to_parquet(dst / "data/chunk-000/file-000.parquet", index=False)

    episodes = pd.concat([pd.read_parquet(p) for p in sorted((src / "meta/episodes").rglob("*.parquet"))])
    episodes = episodes[episodes["episode_index"].isin(kept)].sort_values("episode_index").copy()
    episodes["episode_index"] = episodes["episode_index"].map(renumber)
    episodes["dataset_to_index"] = np.cumsum(episodes["length"].to_numpy())
    episodes["dataset_from_index"] = episodes["dataset_to_index"] - episodes["length"]
    episodes["data/chunk_index"] = 0
    episodes["data/file_index"] = 0
    episodes["meta/episodes/chunk_index"] = 0
    episodes["meta/episodes/file_index"] = 0
    (dst / "meta/episodes/chunk-000").mkdir(parents=True)
    episodes.to_parquet(dst / "meta/episodes/chunk-000/file-000.parquet", index=False)
    shutil.copy2(src / "meta/tasks.parquet", dst / "meta/tasks.parquet")
    _link_tree(src / "videos", dst / "videos")

    for key in ("observation.state", "action"):
        feature = info["features"][key]
        feature["shape"] = [HAND_DIM]
        feature["names"] = feature["names"][:HAND_DIM]
    info["total_episodes"] = len(kept)
    info["total_frames"] = len(out)
    info["splits"] = {"train": f"0:{len(kept)}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    stats = json.loads((src / "meta/stats.json").read_text())
    stats["observation.state"] = _stats(states[keep_rows, :HAND_DIM])
    stats["action"] = _stats(actions[keep_rows, :HAND_DIM])
    (dst / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    provenance = {
        "source": args.src_repo_id,
        "formula": "p = p_home + Rz(p_umi - p_anchor); R = Rz R_umi R_anchor^T Rz^T R_home (per session, per arm)",
        "output_tcp_convention": "flexiv (from R_home); deploy with --args.no-align-gripper-frames",
        "homes_flexiv_tcp_pose": {"left": list(args.left_home), "right": list(args.right_home)},
        "sessions": sessions,
        "dropped_episodes": dropped,
        "episode_renumbering": {str(k): v for k, v in renumber.items()},
        "thresholds": {"max_step_m": args.max_step_m, "max_step_deg": args.max_step_deg},
    }
    (dst / "meta/rebase.json").write_text(json.dumps(provenance, indent=2))
    logger.info("Wrote %d episodes / %d frames to %s (dropped %d)", len(kept), len(out), dst, len(dropped))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main(tyro.cli(Args))
