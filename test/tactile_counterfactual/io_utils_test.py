"""Tests for run output writing (YAML + npz + progress)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

from test.tactile_counterfactual.io_utils import RunWriter
from test.tactile_counterfactual.io_utils import create_run_dir
from test.tactile_counterfactual.probe_config import OutputSpec
from test.tactile_counterfactual.trace_sampler import ActionTrace
from test.tactile_counterfactual.trace_sampler import StepTrace

TACTILE_KEYS = ("tactile_0_rgb", "tactile_1_rgb", "tactile_2_rgb", "tactile_3_rgb")


def _fake_trace(num_steps: int = 2, ah: int = 5, ad: int = 3, emb: int = 8) -> ActionTrace:
    steps = []
    for i in range(num_steps):
        steps.append(  # noqa: PERF401 — multi-field StepTrace, not a one-liner
            StepTrace(
                step=i,
                timestep=np.float32(1.0 - i / num_steps),
                x_t_before=np.ones((1, ah, ad), dtype=np.float32) * i,
                time_masked=np.ones((1, ah), dtype=np.float32) * i,
                suffix_tokens=np.ones((1, 4 + ah, emb), dtype=np.float32) * i,
                suffix_input_mask=np.ones((1, 4 + ah), dtype=bool),
                suffix_ar_mask=np.ones((4 + ah,), dtype=bool),
                adarms_cond=np.ones((1, 4 + ah, emb), dtype=np.float32) * i,
                action_hidden=np.ones((1, ah, emb), dtype=np.float32) * i,
                v_t=np.ones((1, ah, ad), dtype=np.float32) * i,
                x_t_after=np.ones((1, ah, ad), dtype=np.float32) * (i + 1),
            )
        )
    return ActionTrace(
        final_action=np.ones((1, ah, ad), dtype=np.float32),
        fastvit_features=np.ones((1, 4, 16), dtype=np.float32),
        tactile_tokens=np.ones((1, 4, emb), dtype=np.float32),
        tactile_tokens_from_suffix=np.ones((1, 4, emb), dtype=np.float32),
        tactile_mask=np.ones((1, 4), dtype=bool),
        steps=steps,
        inference_mode="rtc",
        num_steps=num_steps,
    )


@pytest.fixture
def run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "runs_root"
    d.mkdir()
    (d / "runs").mkdir()
    return d


def test_create_run_dir(tmp_path: pathlib.Path):
    out = create_run_dir(tmp_path / "out")
    assert out.is_dir()
    assert (out / "runs").is_dir()
    assert out.name == out.name  # timestamped folder


def test_save_run_yaml_and_npz(run_dir: pathlib.Path):
    spec = OutputSpec(dir=str(run_dir), shard_size=50, hidden_storage_dtype="float16", save_suffix_tokens=True)
    writer = RunWriter(run_dir, spec, TACTILE_KEYS)
    metadata = {
        "pair_id": 3,
        "condition": "F_F",
        "base_label": "full",
        "noise_seed": 12345,
        "action_horizon": 5,
        "action_dim": 3,
    }
    trace = _fake_trace()
    out = writer.save_run("0003_F_F", metadata, trace)

    yaml_path = run_dir / "runs" / "0003_F_F.yaml"
    assert yaml_path.is_file()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "0003_F_F"
    assert data["condition"] == "F_F"
    assert data["pair_id"] == 3
    assert data["noise_seed"] == 12345
    assert data["shapes"]["final_action"] == [5, 3]
    assert data["shapes"]["fastvit_features"] == [4, 16]
    assert data["shapes"]["tactile_tokens"] == [4, 8]
    assert data["shapes"]["action_hidden"] == [2, 5, 8]
    assert data["shapes"]["suffix_tokens"] == [2, 9, 8]
    assert data["shapes"]["adarms_cond"] == [2, 9, 8]
    assert len(data["steps"]) == 2
    assert len(data["final_action"]) == 5
    assert data["tactile_keys"] == list(TACTILE_KEYS)
    assert data["tactile_masks"] == [True, True, True, True]
    assert data["trace_npz"] == "0003_F_F_trace.npz"

    npz_path = run_dir / "runs" / "0003_F_F_trace.npz"
    assert npz_path.is_file()
    with np.load(npz_path) as z:
        assert z["action_hidden"].dtype == np.float16
        assert z["action_hidden"].shape == (2, 5, 8)
        assert z["suffix_tokens"].shape == (2, 9, 8)
        assert z["adarms_cond"].shape == (2, 9, 8)
        assert "v_t" not in z  # small arrays live in the YAML


def test_save_run_without_bulky_arrays(run_dir: pathlib.Path):
    spec = OutputSpec(
        dir=str(run_dir),
        save_action_hidden=False,
        save_suffix_tokens=False,
        save_adarms_cond=False,
    )
    writer = RunWriter(run_dir, spec, TACTILE_KEYS)
    out = writer.save_run("0000_F_F", {"condition": "F_F"}, _fake_trace())
    assert "trace_npz" not in out
    assert "action_hidden" not in out["shapes"]
    assert not (run_dir / "runs" / "0000_F_F_trace.npz").exists()


def test_progress_roundtrip(run_dir: pathlib.Path):
    spec = OutputSpec(dir=str(run_dir))
    writer = RunWriter(run_dir, spec, TACTILE_KEYS)
    assert writer.load_progress() == set()
    writer.save_progress({0, 1, 2}, runs_written=12)
    assert writer.load_progress() == {0, 1, 2}
    writer.save_progress({0, 1, 2, 3}, runs_written=16)
    assert writer.load_progress() == {0, 1, 2, 3}


def test_yaml_is_readable_text(run_dir: pathlib.Path):
    spec = OutputSpec(dir=str(run_dir))
    writer = RunWriter(run_dir, spec, TACTILE_KEYS)
    writer.save_run("0001_F_E", {"condition": "F_E", "pair_id": 1}, _fake_trace(num_steps=1))
    text = (run_dir / "runs" / "0001_F_E.yaml").read_text(encoding="utf-8")
    assert "pair_id" in text
    assert "condition: F_E" in text
    assert "final_action" in text
