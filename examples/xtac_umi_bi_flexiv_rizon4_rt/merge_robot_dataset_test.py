"""Tests for appending a real-robot recording to a re-based UMI dataset.

The training config reads the merged vectors in the per-side-grouped layout without
looking at the names, so pin where the robot dims land. Also pin that the UMI rows are
unchanged and that every episode still points at its own frames and video file.
"""

import dataclasses
import errno
import json

import numpy as np
import pandas as pd
import pytest

from examples.xtac_umi_bi_flexiv_rizon4_rt import merge_robot_dataset as merge

_POSE = ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")
_LEFT_TCP = [f"left_tcp.{axis}" for axis in _POSE]
_RIGHT_TCP = [f"right_tcp.{axis}" for axis in _POSE]
UMI_NAMES = [*_LEFT_TCP, "left_gripper.pos", *_RIGHT_TCP, "right_gripper.pos"]
ROBOT_NAMES = [*_LEFT_TCP, *_RIGHT_TCP, "left_gripper.pos", "right_gripper.pos"]
LEFT, RIGHT, HEAD = "observation.images.left_wrist", "observation.images.right_wrist", "observation.images.head"
# The UMI source has three left-wrist files and two right-wrist files, so the robot's file-000 gets a
# different new index for each camera.
UMI_LENGTHS, UMI_FILES = [4, 3, 5], {LEFT: [0, 1, 2], RIGHT: [0, 0, 1], HEAD: [0, 0, 0]}
ROBOT_LENGTHS, ROBOT_FILES = [6, 2], {LEFT: [0, 0], RIGHT: [0, 0], HEAD: [0, 0]}


def _write_source(root, names, lengths, files, *, height, fps=30, task="pick up the cube", seed=0):
    """Minimal LeRobot v3 tree with random vectors. ``files[key][e]`` is the video file of episode ``e``."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for e, length in enumerate(lengths):
        vectors = rng.standard_normal((length, len(names))).astype(np.float32)
        for t in range(length):
            rows.append(
                {
                    "action": vectors[min(t + 1, length - 1)],
                    "observation.state": vectors[t],
                    "observation.extra": np.zeros(2, dtype=np.float32),
                    "timestamp": np.float32(t / fps),
                    "frame_index": t,
                    "episode_index": e,
                    "index": len(rows),
                    "task_index": 0,
                }
            )
        episode_meta = {"episode_index": e, "tasks": [task], "length": length, "stats/timestamp/min": [0.0]}
        for key, file_of in files.items():
            start = sum(n for n, file in zip(lengths[:e], file_of, strict=False) if file == file_of[e]) / fps
            episode_meta[f"videos/{key}/chunk_index"] = 0
            episode_meta[f"videos/{key}/file_index"] = file_of[e]
            episode_meta[f"videos/{key}/from_timestamp"] = start
            episode_meta[f"videos/{key}/to_timestamp"] = start + length / fps
            video = root / f"videos/{key}/chunk-000/file-{file_of[e]:03d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_text(f"{root.name} {key} {file_of[e]}")
        meta.append(episode_meta)
    frames = pd.DataFrame(rows)
    (root / "data/chunk-000").mkdir(parents=True)
    frames.to_parquet(root / "data/chunk-000/file-000.parquet")
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(meta).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=[task]).to_parquet(root / "meta/tasks.parquet")
    features = {key: {"dtype": "float32", "shape": [len(names)], "names": names} for key in merge.VECTOR_KEYS}
    features["observation.extra"] = {"dtype": "float32", "shape": [2], "names": ["a", "b"]}
    features |= {key: {"dtype": "video", "shape": [height, 640, 3]} for key in files}
    features |= {column: {"dtype": "int64", "shape": [1], "names": None} for column in merge.FRAME_COLUMNS}
    info = {
        "fps": fps,
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "total_episodes": len(lengths),
        "total_frames": len(frames),
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    return frames


def _merge(tmp_path, **robot_kwargs):
    """Write a UMI and a robot source, merge them, and return (UMI frames, robot frames, merged root)."""
    umi = _write_source(tmp_path / "Xense/umi", UMI_NAMES, UMI_LENGTHS, UMI_FILES, height=1080, seed=0)
    robot_kwargs = {"names": ROBOT_NAMES, "height": 480, "seed": 1} | robot_kwargs
    robot = _write_source(tmp_path / "Xense/robot", lengths=ROBOT_LENGTHS, files=ROBOT_FILES, **robot_kwargs)
    args = merge.Args(
        umi_repo_id="Xense/umi",
        robot_repo_id="Xense/robot",
        dst_repo_id="Xense/dst",
        lerobot_home=tmp_path,
        overwrite=True,
    )
    merge.main(args)
    return umi, robot, tmp_path / "Xense/dst"


def test_dim_order_follows_names():
    assert merge.dim_order(UMI_NAMES, ROBOT_NAMES) == [*range(9), 18, *range(9, 18), 19]
    assert merge.dim_order(UMI_NAMES, UMI_NAMES) == list(range(20))


def test_dim_order_rejects_other_layouts():
    with pytest.raises(SystemExit, match="feature names differ"):
        merge.dim_order(UMI_NAMES, [*ROBOT_NAMES[:-1], "right_gripper.width"])
    with pytest.raises(SystemExit, match="feature names differ"):
        merge.dim_order(UMI_NAMES, ROBOT_NAMES[:-1])
    # The two repo ids swapped: the "UMI" dataset is in the robot's layout.
    with pytest.raises(SystemExit, match="not in the 20D per-side-grouped layout"):
        merge.dim_order(ROBOT_NAMES, UMI_NAMES)
    # The policy consumes dimensions positionally, even if both datasets have the same names.
    misordered = [UMI_NAMES[1], UMI_NAMES[0], *UMI_NAMES[2:]]
    with pytest.raises(SystemExit, match="not in the 20D per-side-grouped layout"):
        merge.dim_order(misordered, ROBOT_NAMES)


def test_frames_keep_umi_rows_and_regroup_robot_dims(tmp_path):
    umi, robot, dst = _merge(tmp_path)
    out = pd.read_parquet(dst / "data/chunk-000/file-000.parquet")

    assert list(out.columns) == [*merge.VECTOR_KEYS, *merge.FRAME_COLUMNS]  # no observation.extra
    np.testing.assert_array_equal(out["index"], np.arange(len(umi) + len(robot)))
    np.testing.assert_array_equal(out["episode_index"], np.repeat(np.arange(5), [*UMI_LENGTHS, *ROBOT_LENGTHS]))
    np.testing.assert_array_equal(out["frame_index"], np.concatenate([umi["frame_index"], robot["frame_index"]]))
    np.testing.assert_array_equal(out["timestamp"], np.concatenate([umi["timestamp"], robot["timestamp"]]))
    assert out["timestamp"].dtype == np.float32

    for key in merge.VECTOR_KEYS:
        merged = np.stack(out[key].to_numpy())
        assert merged.dtype == np.float32
        assert merged[: len(umi)].tobytes() == np.stack(umi[key].to_numpy()).tobytes()
        from_robot, source = merged[len(umi) :], np.stack(robot[key].to_numpy())
        np.testing.assert_array_equal(from_robot[:, 0:9], source[:, 0:9])  # left TCP
        np.testing.assert_array_equal(from_robot[:, 9], source[:, 18])  # left gripper
        np.testing.assert_array_equal(from_robot[:, 10:19], source[:, 9:18])  # right TCP
        np.testing.assert_array_equal(from_robot[:, 19], source[:, 19])  # right gripper


def test_episodes_point_at_their_frames_and_video_files(tmp_path):
    umi, _, dst = _merge(tmp_path)
    out = pd.read_parquet(dst / "data/chunk-000/file-000.parquet")
    episodes = pd.read_parquet(dst / "meta/episodes/chunk-000/file-000.parquet")
    umi_episodes = pd.read_parquet(tmp_path / "Xense/umi/meta/episodes/chunk-000/file-000.parquet")
    robot_episodes = pd.read_parquet(tmp_path / "Xense/robot/meta/episodes/chunk-000/file-000.parquet")

    np.testing.assert_array_equal(episodes["episode_index"], np.arange(5))
    np.testing.assert_array_equal(episodes["length"], [*UMI_LENGTHS, *ROBOT_LENGTHS])
    np.testing.assert_array_equal(episodes["dataset_from_index"], [0, 4, 7, 12, 18])
    np.testing.assert_array_equal(episodes["dataset_to_index"], [4, 7, 12, 18, 20])
    assert [list(tasks) for tasks in episodes["tasks"]] == [["pick up the cube"]] * 5
    assert not [column for column in episodes if "head" in column or column.startswith("stats/timestamp")]

    # The robot's file-000 comes after the UMI files of the same camera; timestamps inside the files are unchanged.
    np.testing.assert_array_equal(episodes[f"videos/{LEFT}/file_index"], [0, 1, 2, 3, 3])
    np.testing.assert_array_equal(episodes[f"videos/{RIGHT}/file_index"], [0, 0, 1, 2, 2])
    for key in merge.VIDEO_KEYS:
        np.testing.assert_array_equal(episodes[f"videos/{key}/chunk_index"], 0)
        for field in ("from_timestamp", "to_timestamp"):
            column = f"videos/{key}/{field}"
            np.testing.assert_array_equal(episodes[column], [*umi_episodes[column], *robot_episodes[column]])
    for key, robot_file in ((LEFT, 3), (RIGHT, 2)):
        linked = dst / f"videos/{key}/chunk-000/file-{robot_file:03d}.mp4"
        assert linked.samefile(tmp_path / f"Xense/robot/videos/{key}/chunk-000/file-000.mp4")
        assert linked.read_text() == f"robot {key} 0"
        for umi_file in set(UMI_FILES[key]):
            name = f"videos/{key}/chunk-000/file-{umi_file:03d}.mp4"
            assert (dst / name).samefile(tmp_path / "Xense/umi" / name)
        assert len(list((dst / "videos" / key).rglob("*.mp4"))) == robot_file + 1
    assert sorted(path.name for path in (dst / "videos").iterdir()) == sorted(merge.VIDEO_KEYS)

    global_stats = json.loads((dst / "meta/stats.json").read_text())
    assert sorted(global_stats) == sorted(merge.VECTOR_KEYS)
    for key in merge.VECTOR_KEYS:
        values = np.stack(out[key].to_numpy()).astype(np.float64)
        np.testing.assert_allclose(global_stats[key]["mean"], values.mean(axis=0))
        np.testing.assert_allclose(global_stats[key]["q99"], np.quantile(values, 0.99, axis=0))
        for _, episode in episodes.iterrows():
            rows = values[episode["dataset_from_index"] : episode["dataset_to_index"]]
            assert rows.shape[0] == episode["length"]
            np.testing.assert_allclose(episode[f"stats/{key}/min"], rows.min(axis=0))
            np.testing.assert_allclose(episode[f"stats/{key}/std"], rows.std(axis=0))
            np.testing.assert_allclose(episode[f"stats/{key}/q50"], np.quantile(rows, 0.5, axis=0))
            np.testing.assert_array_equal(episode[f"stats/{key}/count"], [len(rows)])

    info = json.loads((dst / "meta/info.json").read_text())
    assert sorted(info["features"]) == sorted([*merge.VECTOR_KEYS, *merge.VIDEO_KEYS, *merge.FRAME_COLUMNS])
    assert info["features"]["action"]["names"] == UMI_NAMES
    assert info["features"][LEFT]["shape"] == [1080, 640, 3]
    assert (info["total_episodes"], info["total_frames"], info["splits"]) == (5, 20, {"train": "0:5"})
    assert pd.read_parquet(dst / "meta/tasks.parquet").index.tolist() == ["pick up the cube"]

    provenance = json.loads((dst / "meta/merge.json").read_text())
    assert provenance["sources"] == {"umi": "Xense/umi", "robot": "Xense/robot"}
    assert provenance["episodes"] == {"umi": [0, 3], "robot": [3, 5]}
    assert provenance["frames"] == {"umi": [0, len(umi)], "robot": [len(umi), 20]}
    assert provenance["robot_dim_order"]["observation.state"] == [*range(9), 18, *range(9, 18), 19]
    assert provenance["video_files"][LEFT]["robot"] == {
        f"videos/{LEFT}/chunk-000/file-000.mp4": f"videos/{LEFT}/chunk-000/file-003.mp4"
    }
    assert provenance["video_shapes"][RIGHT] == {"umi": [1080, 640, 3], "robot": [480, 640, 3]}


@pytest.mark.parametrize(
    ("robot_kwargs", "message"),
    [
        ({"fps": 15}, "fps differs"),
        ({"task": "pick up the sponge"}, "tasks differ"),
        ({"names": [*ROBOT_NAMES[:-1], "right_gripper.width"]}, "feature names differ"),
    ],
)
def test_mismatched_sources_are_rejected_and_keep_the_destination(tmp_path, robot_kwargs, message):
    sentinel = tmp_path / "Xense/dst/existing.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep existing dataset")
    with pytest.raises(SystemExit, match=message):
        _merge(tmp_path, **robot_kwargs)
    assert sentinel.read_text() == "keep existing dataset"


def test_existing_destination_needs_overwrite(tmp_path):
    _merge(tmp_path)
    args = merge.Args(
        umi_repo_id="Xense/umi", robot_repo_id="Xense/robot", dst_repo_id="Xense/dst", lerobot_home=tmp_path
    )
    with pytest.raises(SystemExit, match="pass --overwrite"):
        merge.main(args)
    for destination in ("Xense/umi", "Xense/../Xense/umi", "Xense/umi/child", "Xense"):
        with pytest.raises(SystemExit, match="overlaps a source"):
            merge.main(dataclasses.replace(args, dst_repo_id=destination, overwrite=True))
    assert (tmp_path / "Xense/umi/meta/info.json").exists()


def test_bad_frame_index_is_rejected(tmp_path):
    root = tmp_path / "umi"
    frames = _write_source(root, UMI_NAMES, UMI_LENGTHS, UMI_FILES, height=1080)
    frames.loc[1, "frame_index"] = 0
    frames.to_parquet(root / "data/chunk-000/file-000.parquet")
    with pytest.raises(SystemExit, match="index or frame_index is not contiguous"):
        merge.load_source(root)


def test_failed_hard_link_keeps_the_destination(tmp_path, monkeypatch):
    sentinel = tmp_path / "Xense/dst/existing.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep existing dataset")

    def fail_link(*args):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(merge.os, "link", fail_link)
    with pytest.raises(OSError, match="cross-device link"):
        _merge(tmp_path)
    assert sentinel.read_text() == "keep existing dataset"
    assert not list(sentinel.parent.parent.glob(".dst-*"))
