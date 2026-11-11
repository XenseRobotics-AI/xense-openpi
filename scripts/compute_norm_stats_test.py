import dataclasses
import json
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.policies import xtac_umi_policy
from openpi.training import config as _config

from . import compute_norm_stats

_G1_RELATIVE_CONFIG = "pi05_base_bi_flexiv_umi_g1_clean_plates_1011_h100"


def _pose9(rng: np.random.Generator, n: int) -> np.ndarray:
    position = rng.normal(size=(n, 3))
    q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    return np.concatenate((position, q[..., :, 0], q[..., :, 1]), axis=-1)


def _write_g1_dataset(root: pathlib.Path, episode_lengths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """29D G1 layout: [left tcp(9), left grip, right tcp(9), right grip, head pose(9)]."""
    rng = np.random.default_rng(0)
    n = sum(episode_lengths)
    state = np.concatenate(
        (_pose9(rng, n), rng.uniform(size=(n, 1)), _pose9(rng, n), rng.uniform(size=(n, 1)), _pose9(rng, n)),
        axis=-1,
    ).astype(np.float32)
    actions = (state + rng.normal(scale=0.01, size=state.shape)).astype(np.float32)
    episode = np.repeat(np.arange(len(episode_lengths)), episode_lengths)
    frame = np.concatenate([np.arange(length) for length in episode_lengths])

    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    (root / "data" / "chunk-000").mkdir(parents=True)
    table = pa.table(
        {
            "episode_index": episode,
            "frame_index": frame,
            "observation.state": list(state),
            "action": list(actions),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return state, actions


def test_parquet_stats_use_training_state_action_space(tmp_path: pathlib.Path):
    config = _config.get_config(_G1_RELATIVE_CONFIG)
    data_config = config.data.create(config.assets_dirs, config.model)
    horizon = 4
    state, actions = _write_g1_dataset(tmp_path, (7, 5))

    loader, _ = compute_norm_stats.create_parquet_dataloader(
        data_config, horizon, batch_size=1000, dataset_root=tmp_path
    )
    (batch,) = list(loader)

    # Reference: the training-time state/action transforms applied to the first episode.
    idx = np.clip(np.arange(7)[:, None] + np.arange(horizon)[None, :], 0, 6)
    expected = {"state": state[:7], "actions": actions[:7][idx]}
    expected = xtac_umi_policy.XtacUmiStateActions(drop_head_pose=True)(expected)
    expected = xtac_umi_policy.XtacUmiRelativeActions()(expected)

    assert batch["state"].shape == (12, 2)
    assert batch["actions"].shape == (12, horizon, xtac_umi_policy.STATE_DIM)
    np.testing.assert_allclose(batch["state"][:7], expected["state"], atol=1e-6)
    np.testing.assert_allclose(batch["actions"][:7], expected["actions"], atol=1e-5)


def test_resolve_dataset_root_prefers_config_then_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    env_root, config_root = tmp_path / "env", tmp_path / "config"
    monkeypatch.delenv(_config.DATASET_ROOT_ENV, raising=False)
    assert _config.resolve_dataset_root(_config.DataConfig()) is None

    monkeypatch.setenv(_config.DATASET_ROOT_ENV, str(env_root))
    assert _config.resolve_dataset_root(_config.DataConfig()) == env_root
    assert _config.resolve_dataset_root(_config.DataConfig(dataset_root=str(config_root))) == config_root


def test_training_dataset_uses_env_dataset_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    from openpi.training import data_loader

    seen = []

    class _StopError(Exception):
        pass

    def fake_metadata(repo_id, root=None):
        seen.append(root)
        raise _StopError

    monkeypatch.setenv(_config.DATASET_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", fake_metadata)
    config = _config.get_config(_G1_RELATIVE_CONFIG)
    data_config = dataclasses.replace(config.data.create(config.assets_dirs, config.model), dataset_root=None)
    with pytest.raises(_StopError):
        data_loader.create_torch_dataset(data_config, 4, config.model)
    assert seen == [tmp_path]
