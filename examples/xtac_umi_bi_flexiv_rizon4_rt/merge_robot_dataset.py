"""Append a real-robot recording to a re-based XTac-UMI dataset.

``rebase_umi_dataset`` puts handheld UMI episodes into the Flexiv arms' frames, so they
can be trained on together with real bimanual Flexiv episodes of the same task. A training
config reads one repo id, so this script writes one LeRobot v3 dataset: the UMI episodes
first, with their indices unchanged, then the robot episodes.

The two sources order the same 20 state/action dims differently::

    UMI    [left_tcp(9), left_gripper.pos, right_tcp(9), right_gripper.pos]
    robot  [left_tcp(9), right_tcp(9), left_gripper.pos, right_gripper.pos]

``LeRobotXtacUmiDataConfig`` reads the UMI order, so the robot dims are moved into it.
Each dim is found by its feature name in ``meta/info.json``, and the two sources must have
the same names. The robot values are not changed otherwise: they are true robot poses,
so there is nothing to re-base.

Only what training reads is kept: ``observation.state``, ``action``, the index, timestamp
and task columns, and the two wrist cameras. Statistics of state and action are rebuilt
from the merged values, per episode and in ``meta/stats.json``.

Videos are hard-linked and never re-encoded. The UMI files keep their names, and the robot
files get the file indices after them. Each episode keeps its own from/to timestamps,
which still hold because the files are unchanged.

The two sources record the wrist cameras at different resolutions. ``meta/info.json``
declares one shape per camera, and it keeps the UMI's, so that shape is wrong for the
robot episodes. LeRobot does not use it to decode: each frame comes out at the resolution
of its own file, and the training transforms resize every frame to the model's input
size. ``meta/merge.json`` records the shape of each source, together with the source repo
ids, the episodes and frames that came from each, the dim order and the linked files.

    python -m examples.xtac_umi_bi_flexiv_rizon4_rt.merge_robot_dataset \\
        --umi-repo-id Xense/xtac-umi-g1-pick-place-261119-fwd04 \\
        --robot-repo-id Xense/pick-place-1119 \\
        --dst-repo-id Xense/xtac-umi-g1-pick-place-261119-fwd04-plus-real-1119
"""

import dataclasses
import json
import logging
import os
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd
import tyro

from examples.xtac_umi_bi_flexiv_rizon4_rt import rebase_umi_dataset as rebase

logger = logging.getLogger(__name__)

VECTOR_KEYS = ("observation.state", "action")
VIDEO_KEYS = ("observation.images.left_wrist", "observation.images.right_wrist")
FRAME_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
VIDEO_FIELDS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
TCP_AXES = ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")
UMI_NAMES = [
    *(f"left_tcp.{axis}" for axis in TCP_AXES),
    "left_gripper.pos",
    *(f"right_tcp.{axis}" for axis in TCP_AXES),
    "right_gripper.pos",
]


@dataclasses.dataclass
class Args:
    # Output of rebase_umi_dataset: 20D per-side-grouped state/action.
    umi_repo_id: str = "Xense/xtac-umi-g1-pick-place-261119-fwd04"
    # Real bimanual Flexiv recording of the same task.
    robot_repo_id: str = "Xense/pick-place-1119"
    dst_repo_id: str = "Xense/xtac-umi-g1-pick-place-261119-fwd04-plus-real-1119"
    # Dataset root containing org/name dirs; defaults to $HF_LEROBOT_HOME (~/.cache/huggingface/lerobot).
    lerobot_home: pathlib.Path | None = None
    overwrite: bool = False


@dataclasses.dataclass
class Source:
    root: pathlib.Path
    info: dict
    data: pd.DataFrame
    episodes: pd.DataFrame
    tasks: dict[str, int]


def _read_parquets(directory: pathlib.Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(directory.rglob("*.parquet"))], ignore_index=True)


def load_source(root: pathlib.Path) -> Source:
    """Read one dataset; its frames are sorted by episode and frame, its episodes by index."""
    info = json.loads((root / "meta/info.json").read_text())
    data = _read_parquets(root / "data").sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    episodes = _read_parquets(root / "meta/episodes").sort_values("episode_index").reset_index(drop=True)
    # The episode offsets below need episodes 0..N-1 whose lengths match the frames.
    grouped_frames = data.groupby("episode_index", sort=False)
    frames_per_episode = grouped_frames.size()
    if (
        episodes["episode_index"].tolist() != list(range(len(episodes)))
        or frames_per_episode.index.tolist() != list(range(len(episodes)))
        or frames_per_episode.tolist() != episodes["length"].tolist()
    ):
        raise SystemExit(f"{root}: the frames in data/ do not match meta/episodes (expected episodes 0..N-1)")
    if not np.array_equal(data["index"], np.arange(len(data))) or not np.array_equal(
        data["frame_index"], grouped_frames.cumcount()
    ):
        raise SystemExit(f"{root}: index or frame_index is not contiguous")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    task_index = {str(task): int(index) for task, index in tasks["task_index"].items()}
    return Source(root, info, data, episodes, task_index)


def dim_order(umi_names: list[str], robot_names: list[str]) -> list[int]:
    """For each dim of the UMI layout, the index of the robot dim with the same name."""
    if umi_names != UMI_NAMES:
        raise SystemExit(f"The UMI dataset is not in the 20D per-side-grouped layout: {umi_names}")
    if sorted(umi_names) != sorted(robot_names):
        raise SystemExit(f"The feature names differ.\n  UMI:   {umi_names}\n  robot: {robot_names}")
    return [robot_names.index(name) for name in umi_names]


def video_files(episodes: pd.DataFrame, key: str) -> list[tuple[int, int]]:
    """The (chunk_index, file_index) of every video file the episodes point at, sorted."""
    columns = episodes[[f"videos/{key}/chunk_index", f"videos/{key}/file_index"]]
    return sorted({(int(chunk), int(file)) for chunk, file in columns.itertuples(index=False)})


def _video_path(info: dict, key: str, chunk: int, file: int) -> str:
    name = info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
    path = pathlib.Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Video path must be relative to the dataset: {name}")
    return name


def _vectors(source: Source, key: str) -> np.ndarray:
    values = np.stack(source.data[key].to_numpy())
    if values.shape[1:] != (rebase.HAND_DIM,) or values.dtype != np.float32:
        raise SystemExit(f"{source.root}: {key} must contain 20D float32 vectors")
    return values


def main(args: Args) -> None:
    home_root = args.lerobot_home or pathlib.Path(
        os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot")
    )
    dst = home_root / args.dst_repo_id
    umi_root, robot_root = home_root / args.umi_repo_id, home_root / args.robot_repo_id
    resolved_dst = dst.resolve()
    for src in (umi_root.resolve(), robot_root.resolve()):
        if resolved_dst.is_relative_to(src) or src.is_relative_to(resolved_dst):
            raise SystemExit(f"--dst-repo-id {args.dst_repo_id} overlaps a source: {src}")
    if dst.is_symlink():
        raise SystemExit(f"{dst} is a symlink; choose a dataset directory.")
    if dst.exists() and not args.overwrite:
        raise SystemExit(f"{dst} exists; pass --overwrite to replace it.")
    umi, robot = load_source(umi_root), load_source(robot_root)

    if umi.info["fps"] != robot.info["fps"]:
        raise SystemExit(f"fps differs: UMI {umi.info['fps']}, robot {robot.info['fps']}")
    if umi.tasks != robot.tasks:
        raise SystemExit(f"The tasks differ.\n  UMI:   {umi.tasks}\n  robot: {robot.tasks}")
    order = {
        key: dim_order(umi.info["features"][key]["names"], robot.info["features"][key]["names"]) for key in VECTOR_KEYS
    }

    # ---- frames: the UMI rows as they are, then the robot rows with their dims regrouped
    robot_data = robot.data.copy()
    for key in VECTOR_KEYS:
        robot_data[key] = list(_vectors(robot, key)[:, order[key]])
    robot_data["episode_index"] += len(umi.episodes)
    columns = [*VECTOR_KEYS, *FRAME_COLUMNS]
    out = pd.concat([umi.data[columns], robot_data[columns]], ignore_index=True)
    out["index"] = np.arange(len(out))

    # ---- videos: the robot files are numbered on from the last UMI file of each camera
    chunks_size = umi.info["chunks_size"]
    robot_episodes = robot.episodes.copy()
    robot_episodes["episode_index"] += len(umi.episodes)
    links, linked_files, shapes = [], {}, {}
    for key in VIDEO_KEYS:
        umi_files = video_files(umi.episodes, key)
        last_chunk, last_file = umi_files[-1]
        first_free = last_chunk * chunks_size + last_file + 1
        moved = {old: divmod(first_free + i, chunks_size) for i, old in enumerate(video_files(robot.episodes, key))}
        chunk_column, file_column = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        new = [moved[old] for old in zip(robot_episodes[chunk_column], robot_episodes[file_column], strict=True)]
        robot_episodes[chunk_column] = [chunk for chunk, _ in new]
        robot_episodes[file_column] = [file for _, file in new]

        linked_files[key] = {
            "umi": [_video_path(umi.info, key, *old) for old in umi_files],
            "robot": {
                _video_path(robot.info, key, *old): _video_path(umi.info, key, *new) for old, new in moved.items()
            },
        }
        links += [(umi.root / name, name) for name in linked_files[key]["umi"]]
        links += [(robot.root / old, new) for old, new in linked_files[key]["robot"].items()]
        shapes[key] = {"umi": umi.info["features"][key]["shape"], "robot": robot.info["features"][key]["shape"]}
        if shapes[key]["umi"] != shapes[key]["robot"]:
            logger.warning(
                "%s: UMI frames are %s and robot frames are %s; info.json keeps the UMI shape",
                key,
                shapes[key]["umi"],
                shapes[key]["robot"],
            )
    if missing := [str(src) for src, _ in links if not src.is_file()]:
        raise SystemExit(f"Missing video files: {missing}")

    # ---- episodes: only the columns LeRobot reads, plus the rebuilt state/action stats
    kept = [
        "episode_index",
        "tasks",
        "length",
        *(f"videos/{key}/{field}" for key in VIDEO_KEYS for field in VIDEO_FIELDS),
    ]
    episodes = pd.concat([umi.episodes[kept], robot_episodes[kept]], ignore_index=True)
    episodes["dataset_to_index"] = np.cumsum(episodes["length"].to_numpy())
    episodes["dataset_from_index"] = episodes["dataset_to_index"] - episodes["length"]
    for column in ("data/chunk_index", "data/file_index", "meta/episodes/chunk_index", "meta/episodes/file_index"):
        episodes[column] = 0
    stats = {}
    for key in VECTOR_KEYS:
        values = np.concatenate((_vectors(umi, key), np.stack(robot_data[key].to_numpy()))).astype(np.float64)
        stats[key] = rebase._stats(values)
        per_episode = [
            rebase._stats(values[start:stop])
            for start, stop in zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True)
        ]
        for stat in stats[key]:
            episodes[f"stats/{key}/{stat}"] = [episode_stats[stat] for episode_stats in per_episode]

    info = dict(umi.info)
    info["features"] = {key: feature for key, feature in umi.info["features"].items() if key in (*columns, *VIDEO_KEYS)}
    info["total_episodes"] = len(episodes)
    info["total_frames"] = len(out)
    info["splits"] = {"train": f"0:{len(episodes)}"}

    provenance = {
        "sources": {"umi": args.umi_repo_id, "robot": args.robot_repo_id},
        # Half-open ranges in the merged dataset. Robot episode e is merged episode e + the UMI episode count.
        "episodes": {"umi": [0, len(umi.episodes)], "robot": [len(umi.episodes), len(episodes)]},
        "frames": {"umi": [0, len(umi.data)], "robot": [len(umi.data), len(out)]},
        # merged[i] = robot[robot_dim_order[i]]
        "robot_dim_order": order,
        "video_files": linked_files,
        "video_shapes": shapes,
    }

    # ---- stage next to the destination so a failed write or video link leaves the old dataset alone.
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent, prefix=f".{dst.name}-") as tmp:
        staging = pathlib.Path(tmp) / "dataset"
        (staging / "data/chunk-000").mkdir(parents=True)
        out.to_parquet(staging / "data/chunk-000/file-000.parquet", index=False)
        (staging / "meta/episodes/chunk-000").mkdir(parents=True)
        episodes.to_parquet(staging / "meta/episodes/chunk-000/file-000.parquet", index=False)
        shutil.copy2(umi.root / "meta/tasks.parquet", staging / "meta/tasks.parquet")
        (staging / "meta/info.json").write_text(json.dumps(info, indent=4))
        (staging / "meta/stats.json").write_text(json.dumps(stats, indent=4))
        (staging / "meta/merge.json").write_text(json.dumps(provenance, indent=2))
        for src_file, name in links:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(src_file, target)
        if dst.exists():
            if not args.overwrite:
                raise SystemExit(f"{dst} exists; pass --overwrite to replace it.")
            shutil.rmtree(dst)
        staging.rename(dst)
    logger.info(
        "Wrote %d + %d episodes / %d + %d frames to %s",
        len(umi.episodes),
        len(robot.episodes),
        len(umi.data),
        len(robot.data),
        dst,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main(tyro.cli(Args))
