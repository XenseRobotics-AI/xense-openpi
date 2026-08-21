"""End-to-end runner test with a fake dataset and dummy model.

Skips the real LeRobot dataset / checkpoint loading (covered by the CLI on a
real machine) and exercises the pairing, four-condition execution, file
writing and metrics paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import openpi.models.model as _model
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
from openpi.shared import array_typing as at
from test.tactile_counterfactual.probe_config import DatasetSpec
from test.tactile_counterfactual.probe_config import ExperimentSpec
from test.tactile_counterfactual.probe_config import FrameRef
from test.tactile_counterfactual.probe_config import ModelSpec
from test.tactile_counterfactual.probe_config import OutputSpec
from test.tactile_counterfactual.probe_config import PairSpec
from test.tactile_counterfactual.probe_config import ProbeConfig
from test.tactile_counterfactual.runner import ProbeRunner
from test.tactile_counterfactual.runner import paired_noise
from test.tactile_counterfactual.trace_sampler import TraceSampler

NUM_STEPS = 2
AH, AD = 50, 32


class FakeDataset:
    """Minimal ProbeDataset stand-in producing random observations."""

    repo_id = "fake/dataset"
    num_episodes = 10
    num_frames = 1000

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def has_sample(self, episode_index: int, frame_index: int) -> bool:
        return 0 <= episode_index < 10 and 0 <= frame_index < 100

    def get_sample(self, episode_index: int, frame_index: int) -> dict:
        return {"episode_index": episode_index, "frame_index": frame_index}

    def observation_from_sample(self, sample: dict) -> _model.Observation:
        rng = np.random.default_rng(sample["episode_index"] * 1000 + sample["frame_index"])
        keys = (
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
            "tactile_0_rgb",
            "tactile_1_rgb",
            "tactile_2_rgb",
            "tactile_3_rgb",
        )
        images = {k: rng.uniform(-1, 1, (1, 32, 32, 3)).astype(np.float32) for k in keys}
        masks = {k: np.ones((1,), dtype=bool) for k in keys}
        with at.disable_typechecking():
            obs = _model.Observation(
                images=images,
                image_masks=masks,
                state=rng.uniform(-1, 1, (1, AD)).astype(np.float32),
                tokenized_prompt=np.zeros((1, 200), dtype=np.int32),
                tokenized_prompt_mask=np.ones((1, 200), dtype=bool),
            )
        # Same conversion the production policy path applies.
        return jax.tree.map(jnp.asarray, obs)


@pytest.fixture(scope="module")
def model():
    cfg = Pi0TactileFastVitConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        enable_training_time_rtc=True,
        max_delay=10,
        tactile_pretrained_path=None,
    )
    return cfg.create(jax.random.key(0))


@pytest.fixture(scope="module")
def runner(model, tmp_path_factory) -> ProbeRunner:
    tmp = tmp_path_factory.mktemp("runner_out")
    probe = ProbeConfig(
        dataset=DatasetSpec(repo_id="fake/dataset"),
        model=ModelSpec(checkpoint_dir="ckpt/39999", inline={"type": "Pi0TactileFastVitConfig"}),
        experiment=ExperimentSpec(inference_mode="rtc", num_steps=NUM_STEPS, base_seed=12345, batch_size=1),
        output=OutputSpec(dir=str(tmp), shard_size=2, hidden_storage_dtype="float16"),
        pairs=(
            PairSpec(pair_id=0, full=FrameRef(0, 10), empty=FrameRef(1, 10)),
            PairSpec(pair_id=1, full=FrameRef(2, 10), empty=FrameRef(3, 10)),
        ),
        data={"type": "LeRobotBiFlexivTactileDataConfig", "repo_id": "fake/dataset"},
    )
    # The runner needs a TrainConfig-like object; build one from the inline blocks.
    from test.tactile_counterfactual.runner import build_train_config

    train_config = build_train_config(probe)
    sampler = TraceSampler(model, train_config.model.tactile_image_keys, inference_mode="rtc")
    return ProbeRunner(probe, train_config, model, FakeDataset(), sampler, run_dir=tmp)


def test_run_pair_writes_four_runs(runner: ProbeRunner):
    rng = jax.random.key(0)
    metadatas, metrics = runner.run_pair(runner.probe.pairs[0], rng)

    assert set(metadatas) == {"F_F", "F_E", "E_E", "E_F"}
    runs_dir = runner.run_dir / "runs"
    for cond in ("F_F", "F_E", "E_E", "E_F"):
        yaml_path = runs_dir / f"0000_{cond}.yaml"
        assert yaml_path.is_file()
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert data["condition"] == cond
        assert data["pair_id"] == 0
        assert data["base_label"] in ("full", "empty")
        assert data["tactile_label"] in ("full", "empty")
        assert data["noise_seed"] == 12345 if cond.startswith("F") else data["noise_seed"] == 12346
        assert data["shapes"]["final_action"] == [AH, AD]

    # Metrics per todolist section 14.
    assert set(metrics) >= {
        "final_action_rms_F",
        "final_action_l2_F",
        "final_action_rms_E",
        "final_action_l2_E",
        "per_step_l2_F",
        "per_action_dim_rms_F",
        "action_hidden_rms_difference",
        "v_t_rms_difference",
        "x_t_rms_difference",
    }
    assert len(metrics["per_step_l2_F"]) == AH
    assert len(metrics["per_action_dim_rms_F"]) == AD
    assert len(metrics["action_hidden_rms_difference"]) == NUM_STEPS


def test_paired_noise_used_per_condition(runner: ProbeRunner, monkeypatch):
    """F_F and F_E must receive the identical noise array."""
    seen: dict[str, np.ndarray] = {}
    original = runner.run_condition

    def spy(observation, noise, rng):
        # First call is F_F (conditions run in order F_F, F_E, E_E, E_F).
        if "noise_ff" not in seen:
            seen["noise_ff"] = noise
        return original(observation, noise, rng)

    monkeypatch.setattr(runner, "run_condition", spy)
    rng = jax.random.key(0)
    runner.run_pair(runner.probe.pairs[1], rng)

    # run_pair calls conditions in order F_F, F_E, E_E, E_F; the spy records
    # the last noise, so verify seeds used were 12347 and 12348 (pair 1).
    assert seen["noise_ff"].shape == (1, AH, AD)
    np.testing.assert_array_equal(seen["noise_ff"], paired_noise(12345 + 2 * 1, 1, AH, AD))


def test_dry_run_writes_summary(runner: ProbeRunner):
    summary = runner.run(dry_run=True)
    assert summary["dry_run"] is True
    assert "equivalence" in summary
    assert "sanity_checks" in summary
    assert (runner.run_dir / "summary.yaml").is_file()


def test_full_run_two_pairs(runner: ProbeRunner):
    summary = runner.run(max_pairs=2)
    assert set(summary["pairs"]) == {"0", "1"}
    assert (runner.run_dir / "progress.json").is_file()
    progress = runner.writer.load_progress()
    assert progress == {0, 1}
    runs = list((runner.run_dir / "runs").glob("*.yaml"))
    assert len(runs) == 8  # 2 pairs x 4 conditions (plus none from dry-run reuse)
