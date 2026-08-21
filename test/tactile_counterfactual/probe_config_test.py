"""Tests for probe YAML parsing and static validation."""

from __future__ import annotations

import pathlib

import pytest

from test.tactile_counterfactual.probe_config import ProbeConfig
from test.tactile_counterfactual.probe_config import from_dict
from test.tactile_counterfactual.probe_config import load_probe_config

MINIMAL = {
    "dataset": {"repo_id": "Xense/bottle-sorting-0810", "revision": None, "root": None},
    "model": {
        "config_name": "pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100",
        "checkpoint_dir": "checkpoints/xxx/39999",
    },
    "experiment": {"inference_mode": "rtc", "num_steps": 10, "base_seed": 12345, "batch_size": 1},
    "output": {
        "dir": "outputs/tactile_counterfactual",
        "shard_size": 50,
        "hidden_storage_dtype": "float16",
    },
    "pairs": [
        {
            "pair_id": 0,
            "full": {"episode_index": 10, "frame_index": 245},
            "empty": {"episode_index": 510, "frame_index": 231},
        },
        {
            "pair_id": 1,
            "full": {"episode_index": 11, "frame_index": 1},
            "empty": {"episode_index": 511, "frame_index": 2},
        },
    ],
}


def test_parse_minimal(tmp_path: pathlib.Path):
    path = tmp_path / "probe.yaml"
    path.write_text(_dump(MINIMAL))
    cfg = load_probe_config(path)
    assert isinstance(cfg, ProbeConfig)
    assert len(cfg.pairs) == 2
    assert cfg.pairs[0].pair_id == 0
    assert cfg.pairs[0].full.episode_index == 10
    assert cfg.pairs[0].full.frame_index == 245
    assert cfg.pairs[0].empty.episode_index == 510
    assert cfg.experiment.inference_mode == "rtc"
    assert cfg.experiment.num_steps == 10
    assert cfg.experiment.base_seed == 12345
    assert cfg.output.shard_size == 50
    assert cfg.output.save_action_hidden is True
    assert cfg.output.hidden_storage_dtype == "float16"


def test_defaults():
    cfg = from_dict(dict(MINIMAL))
    assert cfg.experiment.inference_mode == "rtc"
    assert cfg.output.save_fastvit_features is True
    assert cfg.output.save_suffix_tokens is False


def test_duplicate_pair_id_rejected():
    raw = dict(MINIMAL)
    raw["pairs"] = [
        {"pair_id": 0, "full": {"episode_index": 1, "frame_index": 1}, "empty": {"episode_index": 2, "frame_index": 2}},
        {"pair_id": 0, "full": {"episode_index": 3, "frame_index": 3}, "empty": {"episode_index": 4, "frame_index": 4}},
    ]
    with pytest.raises(ValueError, match="duplicate pair_id"):
        from_dict(raw)


def test_missing_pairs_rejected():
    raw = dict(MINIMAL)
    del raw["pairs"]
    with pytest.raises(ValueError, match="pairs"):
        from_dict(raw)


def test_bad_inference_mode_rejected():
    raw = dict(MINIMAL)
    raw["experiment"] = {**raw["experiment"], "inference_mode": "bogus"}
    with pytest.raises(ValueError, match="inference_mode"):
        from_dict(raw)


def test_negative_frame_rejected():
    raw = dict(MINIMAL)
    raw["pairs"] = [
        {
            "pair_id": 0,
            "full": {"episode_index": 1, "frame_index": -3},
            "empty": {"episode_index": 2, "frame_index": 2},
        },
    ]
    with pytest.raises(ValueError, match="frame_index"):
        from_dict(raw)


def test_bool_not_accepted_as_int():
    raw = dict(MINIMAL)
    raw["pairs"] = [
        {
            "pair_id": 0,
            "full": {"episode_index": 1, "frame_index": True},
            "empty": {"episode_index": 2, "frame_index": 2},
        },
    ]
    with pytest.raises(ValueError, match="frame_index"):
        from_dict(raw)


def test_repeated_frame_warns(capsys):
    raw = dict(MINIMAL)
    raw["pairs"] = [
        {"pair_id": 0, "full": {"episode_index": 7, "frame_index": 7}, "empty": {"episode_index": 2, "frame_index": 2}},
        {"pair_id": 1, "full": {"episode_index": 7, "frame_index": 7}, "empty": {"episode_index": 4, "frame_index": 4}},
    ]
    cfg = from_dict(raw)  # must not raise
    out = capsys.readouterr().out
    assert "appears more than once" in out
    assert len(cfg.pairs) == 2


def test_inline_model_requires_data():
    raw = dict(MINIMAL)
    raw["model"] = {"checkpoint_dir": "ckpt/39999", "inline": {"type": "Pi0TactileFastVitConfig"}}
    raw.pop("data", None)  # MINIMAL has no data block; ensure it stays absent
    with pytest.raises(ValueError, match="data"):
        from_dict(raw)


def test_inline_model_with_data_ok():
    raw = dict(MINIMAL)
    raw["model"] = {
        "checkpoint_dir": "ckpt/39999",
        "inline": {
            "type": "Pi0TactileFastVitConfig",
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "pi05": True,
        },
    }
    raw["data"] = {"type": "LeRobotBiFlexivTactileDataConfig", "repo_id": "Xense/bottle-sorting-0810"}
    cfg = from_dict(raw)
    assert cfg.model.inline["type"] == "Pi0TactileFastVitConfig"


def test_missing_checkpoint_dir_rejected():
    raw = dict(MINIMAL)
    del raw["model"]["checkpoint_dir"]
    with pytest.raises(ValueError, match="checkpoint_dir"):
        from_dict(raw)


def _dump(raw: dict) -> str:
    import yaml

    return yaml.safe_dump(raw, sort_keys=False)
