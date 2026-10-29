"""Tests for the offline UMI -> Flexiv re-basing.

Every training label goes through ``rebase_poses``, so pin the teleop semantics:
the anchor lands on home, world-frame motion is preserved, and rotation deltas are
applied in the world frame (left-multiplied), independent of the gripper axes.
"""

import json

import numpy as np
import pandas as pd

from examples.xtac_umi_bi_flexiv_rizon4_rt import gripper_frame
from examples.xtac_umi_bi_flexiv_rizon4_rt import rebase_umi_dataset as rebase


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def _transform(rotation: np.ndarray, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _pose9(matrix: np.ndarray) -> np.ndarray:
    return gripper_frame.matrix_to_pose9(matrix)


def test_anchor_maps_to_home():
    rng = np.random.default_rng(0)
    anchor = _transform(_random_rotation(rng), rng.standard_normal(3))
    home = _transform(_random_rotation(rng), rng.standard_normal(3))
    for yaw in (0.0, 0.4):
        out = gripper_frame.pose9_to_matrix(rebase.rebase_poses(_pose9(anchor), anchor, home, yaw))
        np.testing.assert_allclose(out, home, atol=1e-9)


def test_world_frame_motion_is_preserved():
    """Hand translates by d and rotates by Q about the world -> arm does the same about its home."""
    rng = np.random.default_rng(1)
    anchor = _transform(_random_rotation(rng), rng.standard_normal(3))
    home = _transform(_random_rotation(rng), rng.standard_normal(3))
    world_rotation, d = _random_rotation(rng), rng.standard_normal(3)
    moved = _transform(world_rotation @ anchor[:3, :3], anchor[:3, 3] + d)
    out = gripper_frame.pose9_to_matrix(rebase.rebase_poses(_pose9(moved), anchor, home, 0.0))
    np.testing.assert_allclose(out[:3, 3], home[:3, 3] + d, atol=1e-9)
    np.testing.assert_allclose(out[:3, :3], world_rotation @ home[:3, :3], atol=1e-9)


def test_yaw_maps_operator_forward_to_arm_x():
    """With the operator facing +y (yaw 90 deg), a hand step along +y becomes +x on the arm."""
    anchor, home = np.eye(4), _transform(np.eye(3), [0.7, 0.2, 0.1])
    moved = _transform(np.eye(3), [0.0, 0.1, 0.0])
    out = rebase.rebase_poses(_pose9(moved), anchor, home, np.pi / 2)
    np.testing.assert_allclose(out[:3], [0.8, 0.2, 0.1], atol=1e-9)


def test_detect_sessions_splits_on_jump():
    points = np.array([[0, 0, 0], [0.02, 0, 0], [0.5, 0, 0.3], [0.51, 0, 0.3]])
    np.testing.assert_array_equal(rebase.detect_sessions(points, 0.15), [0, 0, 1, 1])


def test_mean_rotation_of_symmetric_spread_is_center():
    rng = np.random.default_rng(2)
    center = _random_rotation(rng)
    spread = [center @ rebase.rot_z(a) for a in (-0.2, 0.2)]
    np.testing.assert_allclose(rebase.mean_rotation(np.stack(spread)), center, atol=1e-9)


def test_max_frame_steps_catches_relocalization():
    track = np.stack([_pose9(_transform(np.eye(3), [0.001 * t, 0, 0])) for t in range(5)])
    track[3:, 0] += 0.4
    step_m, step_deg = rebase.max_frame_steps(track)
    assert step_m > 0.39
    assert step_deg < 1e-6


def _hand(matrix: np.ndarray, grip: float) -> list:
    return [*_pose9(matrix), grip]


def _write_source(root, episodes):
    """Minimal 20D LeRobot v3 tree; ``episodes`` is a list of [T, 20] state arrays."""
    rows, meta, index = [], [], 0
    for e, states in enumerate(episodes):
        for t, state in enumerate(states):
            nxt = states[min(t + 1, len(states) - 1)]
            rows.append(
                {
                    "observation.state": state.astype(np.float32),
                    "action": nxt.astype(np.float32),
                    "timestamp": np.float32(t / 30),
                    "frame_index": t,
                    "episode_index": e,
                    "index": index,
                    "task_index": 0,
                }
            )
            index += 1
        meta.append({"episode_index": e, "length": len(states), "dataset_from_index": index - len(states)})
    (root / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet")
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(meta).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=["task"]).to_parquet(root / "meta/tasks.parquet")
    names = [f"d{i}" for i in range(20)]
    features = {k: {"dtype": "float32", "shape": [20], "names": names} for k in ("observation.state", "action")}
    (root / "meta/info.json").write_text(json.dumps({"features": features, "total_episodes": len(episodes)}))
    (root / "meta/stats.json").write_text(json.dumps({}))


def test_end_to_end_sessions_anchor_and_drop(tmp_path):
    left0 = _transform(np.eye(3), [0.3, 0.15, -0.3])
    right0 = _transform(np.eye(3), [0.3, -0.15, -0.3])
    session_shift = np.array([-0.17, 0.24, 0.36])  # Unity origin moved between sessions

    def episode(shift, wiggle, jump=False):
        states = []
        for t in range(6):
            offset = shift + np.array([0.01 * t + wiggle, 0, 0])
            if jump and t >= 3:
                offset = offset + np.array([0.4, 0, 0])
            states.append(
                _hand(_transform(np.eye(3), left0[:3, 3] + offset), 1.0)
                + _hand(_transform(np.eye(3), right0[:3, 3] + offset), 0.5)
            )
        return np.array(states)

    zero = np.zeros(3)
    src = tmp_path / "Xense/src"
    _write_source(
        src,
        [
            episode(zero, -0.01),
            episode(zero, 0.01),
            episode(zero, 0.0, jump=True),
            episode(session_shift, -0.01),
            episode(session_shift, 0.01),
        ],
    )
    left_home = (0.7, 0.185, 0.15, 1.0, 0.0, 0.0, 0.0)
    right_home = (0.7, -0.185, 0.15, 1.0, 0.0, 0.0, 0.0)
    rebase.main(
        rebase.Args(
            src_repo_id="Xense/src",
            dst_repo_id="Xense/dst",
            left_home=left_home,
            right_home=right_home,
            lerobot_home=tmp_path,
        )
    )

    dst = tmp_path / "Xense/dst"
    provenance = json.loads((dst / "meta/rebase.json").read_text())
    assert list(provenance["dropped_episodes"]) == ["2"]
    assert [s["episodes"] for s in provenance["sessions"].values()] == [[0, 1], [3, 4]]

    out = pd.read_parquet(dst / "data/chunk-000/file-000.parquet")
    assert sorted(out["episode_index"].unique()) == [0, 1, 2, 3]
    np.testing.assert_array_equal(out["index"], np.arange(len(out)))
    states = np.stack(out["observation.state"].to_numpy())
    starts = states[out["frame_index"].to_numpy() == 0]
    # Each session's anchor is its mean start, so starts keep their real +-1 cm spread around home.
    np.testing.assert_allclose(starts[:, :3].mean(0), left_home[:3], atol=1e-6)
    np.testing.assert_allclose(starts[:, 0], [0.69, 0.71, 0.69, 0.71], atol=1e-6)
    np.testing.assert_allclose(starts[:, 10:13].mean(0), right_home[:3], atol=1e-6)
    np.testing.assert_allclose(states[:, [9, 19]], np.tile([1.0, 0.5], (len(states), 1)))

    info = json.loads((dst / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [20]
    assert info["total_episodes"] == 4
    episodes = pd.read_parquet(dst / "meta/episodes/chunk-000/file-000.parquet")
    np.testing.assert_array_equal(episodes["dataset_from_index"], [0, 6, 12, 18])
