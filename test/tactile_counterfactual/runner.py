"""Probe runner: orchestrates loading, validation, paired inference and saving.

Flow (per todolist):

1. Load the probe YAML, resolve the TrainConfig (by name or inline), resolve
   norm stats (checkpoint assets first, then the config assets dir).
2. Load the checkpoint into a Pi0TactileFastVit model; verify the checkpoint
   actually contains tactile weights.
3. Build the dataset index and validate every pair.
4. Mandatory gate: trace-vs-production equivalence test on a real observation.
5. Optional sanity checks on the first pair (checks 1-7, fail-fast).
6. Run the four conditions per pair with paired fixed noise, save per-run
   YAML (+ npz attachments), compute pair metrics, update progress/summary.
"""

# F_F/F_E/E_E/E_F condition names are experiment terms of art (todolist
# section 1), not PEP8 naming violations.
# ruff: noqa: N806

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import openpi.models.model as _model
from openpi.models.pi0_tactile_fastvit_config import Pi0TactileFastVitConfig
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.yaml_loader as _yaml_loader
from test.tactile_counterfactual.counterfactual import assert_swap_clean
from test.tactile_counterfactual.counterfactual import make_counterfactual_observation
from test.tactile_counterfactual.dataset_index import ProbeDataset
from test.tactile_counterfactual.io_utils import RunWriter
from test.tactile_counterfactual.io_utils import create_run_dir
from test.tactile_counterfactual.metrics import compute_pair_metrics
from test.tactile_counterfactual.probe_config import ProbeConfig
from test.tactile_counterfactual.sanity_checks import log_sanity_report
from test.tactile_counterfactual.sanity_checks import run_sanity_checks
from test.tactile_counterfactual.trace_sampler import TraceSampler
from test.tactile_counterfactual.trace_sampler import verify_trace_equivalence

logger = logging.getLogger("tactile_counterfactual")

CONDITIONS = ("F_F", "F_E", "E_E", "E_F")


# --------------------------------------------------------------------------- #
# Config / model / dataset loading                                            #
# --------------------------------------------------------------------------- #


def build_train_config(probe: ProbeConfig) -> _config.TrainConfig:
    """Resolve the TrainConfig from model.config_name or the inline blocks."""
    if probe.model.config_name is not None:
        try:
            return _config.get_config(probe.model.config_name)
        except Exception as exc:
            raise ValueError(
                f"Failed to resolve model.config_name={probe.model.config_name!r} "
                f"(is the training config present in this repo?). Error: {exc}"
            ) from exc

    # Inline model + data blocks -> build a TrainConfig through the same YAML
    # loader the repo uses (polymorphic `type:` dispatch).
    if probe.model.inline is None or probe.data is None:
        raise ValueError("inline model config requires an inline 'data' block")
    train_yaml = yaml.safe_dump({"model": probe.model.inline, "data": probe.data}, sort_keys=False)
    return _yaml_loader.loads(train_yaml, name="probe_inline")


def resolve_data_config(
    train_config: _config.TrainConfig,
    probe: ProbeConfig,
    checkpoint_dir: pathlib.Path,
) -> _config.DataConfig:
    """DataConfig with norm stats guaranteed present.

    Priority: config assets dir (compute_norm_stats.py output), then the
    checkpoint's own assets (saved by train.py at save time). Missing norm
    stats raise a clear error — silently skipping normalization would produce
    observations inconsistent with training.
    """
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if data_config.norm_stats is not None:
        return data_config

    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError(
            "Norm stats are required for the probe but the data config has no asset_id; "
            "cannot locate normalization stats."
        )

    # 1) checkpoint assets: <checkpoint_dir>/assets/<asset_id>/
    ckpt_assets = checkpoint_dir / "assets" / asset_id
    if ckpt_assets.is_dir():
        norm_stats = _normalize.load(ckpt_assets)
        logger.info("Loaded norm stats from checkpoint assets: %s", ckpt_assets)
        return dataclasses.replace(data_config, norm_stats=norm_stats)

    # 2) config assets dir: <assets>/<config_name>/<asset_id>/
    config_assets = train_config.assets_dirs / asset_id
    if config_assets.is_dir():
        norm_stats = _normalize.load(config_assets)
        logger.info("Loaded norm stats from config assets: %s", config_assets)
        return dataclasses.replace(data_config, norm_stats=norm_stats)

    raise FileNotFoundError(
        f"Norm stats not found for asset_id={asset_id!r}. Looked in:\n"
        f"  {ckpt_assets}\n  {config_assets}\n"
        "Run scripts/compute_norm_stats.py for this config, or point the probe "
        "at a checkpoint that contains its assets."
    )


def load_model(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
) -> tuple[_model.BaseModel, dict[str, Any]]:
    """Load the checkpoint into a model; verify tactile weights are present."""
    model_config = train_config.model
    if not isinstance(model_config, Pi0TactileFastVitConfig):
        raise TypeError(
            f"Probe requires a Pi0TactileFastVitConfig model, got {type(model_config).__name__}. "
            "The counterfactual experiment is only defined for tactile models."
        )

    # The pretrained FastViT path on the training machine may not exist here;
    # weights come from the checkpoint anyway, so drop the path (the encoder
    # structure is identical).
    if (
        model_config.tactile_pretrained_path is not None
        and not pathlib.Path(model_config.tactile_pretrained_path).exists()
    ):
        logger.warning(
            "tactile_pretrained_path %s does not exist locally; ignoring it (weights will come from the checkpoint).",
            model_config.tactile_pretrained_path,
        )
        model_config = dataclasses.replace(model_config, tactile_pretrained_path=None)

    params = _model.restore_params(checkpoint_dir)
    flat = _flatten_keys(params)
    missing_tactile = [k for k in ("tactile_encoder", "tactile_proj") if not any(k in key for key in flat)]
    if missing_tactile:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} has no tactile weights ({missing_tactile}); "
            "cannot run the tactile counterfactual probe. Available top-level keys: "
            f"{sorted({k.split('/')[0] for k in flat})}"
        )

    logger.info("Loading model from checkpoint %s", checkpoint_dir)
    model = model_config.load(params)
    logger.info("Model class: %s", type(model).__name__)
    return model, {"checkpoint_tactile_keys": [k for k in flat if "tactile" in k]}


def _flatten_keys(tree: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flatten_keys(v, key + "/"))
        else:
            out.append(key)
    return out


def build_probe_dataset(
    probe: ProbeConfig,
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> ProbeDataset:
    repo_id = probe.dataset.repo_id or data_config.repo_id
    if repo_id is None:
        raise ValueError("dataset.repo_id is required (and no TrainConfig data repo_id is available)")
    return ProbeDataset(
        repo_id,
        data_config,
        action_horizon=train_config.model.action_horizon,
        revision=probe.dataset.revision,
        root=probe.dataset.root,
    )


def paired_noise(seed: int, batch_size: int, action_horizon: int, action_dim: int) -> np.ndarray:
    """Deterministic flow initial noise for a given seed (paired conditions)."""
    return np.asarray(
        jax.random.normal(jax.random.key(seed), (batch_size, action_horizon, action_dim)),
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #


class ProbeRunner:
    def __init__(
        self,
        probe: ProbeConfig,
        train_config: _config.TrainConfig,
        model: _model.BaseModel,
        dataset: ProbeDataset,
        sampler: TraceSampler,
        run_dir: pathlib.Path | None = None,
    ) -> None:
        self.probe = probe
        self.train_config = train_config
        self.model = model
        self.dataset = dataset
        self.sampler = sampler
        self.tactile_keys = tuple(model._tactile_keys)
        self.run_dir = run_dir or create_run_dir(probe.output.dir)
        self.writer = RunWriter(self.run_dir, probe.output, self.tactile_keys)

    # ------------------------------------------------------------------ #

    def log_header(self) -> None:
        cfg = self.probe
        model_cfg = self.train_config.model
        logger.info(
            "Loaded dataset: %s (%d episodes, %d frames)",
            self.dataset.repo_id,
            self.dataset.num_episodes,
            self.dataset.num_frames,
        )
        logger.info("Pairs: %d", len(cfg.pairs))
        logger.info("Expected runs: %d", len(cfg.pairs) * 4)
        logger.info("Model: %s", type(self.model).__name__)
        logger.info("Checkpoint: %s", cfg.model.checkpoint_dir)
        logger.info("Action horizon: %d", model_cfg.action_horizon)
        logger.info("Action dim: %d", model_cfg.action_dim)
        logger.info("FastViT dim: %d", self.model.tactile_encoder.feature_dim)
        logger.info(
            "Action expert width: %d",
            self.model.action_in_proj.out_features,
        )
        logger.info("Tactile keys: %s", self.tactile_keys)
        logger.info("Inference mode: %s (%d steps)", self.sampler.inference_mode, cfg.experiment.num_steps)
        logger.info("Output dir: %s", self.run_dir)

    # ------------------------------------------------------------------ #

    def validate_pairs(self) -> None:
        """Every pair's (episode, frame) must exist in the dataset."""
        missing = []
        for pair in self.probe.pairs:
            for role, ref in (("full", pair.full), ("empty", pair.empty)):
                if not self.dataset.has_sample(ref.episode_index, ref.frame_index):
                    missing.append(f"pair {pair.pair_id} {role}: ({ref.episode_index}, {ref.frame_index})")
        if missing:
            raise ValueError(
                f"{len(missing)} sample(s) not present in dataset {self.dataset.repo_id}:\n  " + "\n  ".join(missing)
            )
        logger.info("All %d pairs validated against the dataset index.", len(self.probe.pairs))

    # ------------------------------------------------------------------ #

    def get_pair_observations(self, pair) -> tuple[dict, dict]:
        """Load + transform the two base samples of a pair."""
        full_sample = self.dataset.get_sample(pair.full.episode_index, pair.full.frame_index)
        empty_sample = self.dataset.get_sample(pair.empty.episode_index, pair.empty.frame_index)
        obs_full = self.dataset.observation_from_sample(full_sample)
        obs_empty = self.dataset.observation_from_sample(empty_sample)
        self._check_tactile_keys(obs_full)
        self._check_tactile_keys(obs_empty)
        return obs_full, obs_empty

    def _check_tactile_keys(self, obs) -> None:
        """Fail loudly if the transformed observation lacks the tactile keys."""
        missing = [k for k in self.tactile_keys if k not in obs.images]
        if missing:
            raise ValueError(
                f"Transformed observation is missing tactile keys {missing}; "
                f"available image keys: {sorted(obs.images)}. The data pipeline "
                "(repack/data transforms) does not expose the tactile cameras."
            )

    # ------------------------------------------------------------------ #

    def build_conditions(self, obs_full, obs_empty) -> dict[str, tuple[Any, dict]]:
        """Build the four condition observations.

        Returns {condition: (observation, swap_check_payload)}.
        """
        obs_FF = obs_full
        obs_FE = make_counterfactual_observation(obs_full, obs_empty, self.tactile_keys)
        obs_EE = obs_empty
        obs_EF = make_counterfactual_observation(obs_empty, obs_full, self.tactile_keys)
        return {
            "F_F": (obs_FF, {}),
            "F_E": (obs_FE, {"base": obs_full, "donor": obs_empty}),
            "E_E": (obs_EE, {}),
            "E_F": (obs_EF, {"base": obs_empty, "donor": obs_full}),
        }

    # ------------------------------------------------------------------ #

    def run_condition(self, observation, noise: np.ndarray, rng) -> Any:
        """Run the trace sampler for one condition."""
        return self.sampler(
            rng,
            observation,
            num_steps=self.probe.experiment.num_steps,
            noise=noise,
        )

    # ------------------------------------------------------------------ #

    def equivalence_gate(self, obs_full) -> dict[str, float]:
        """Mandatory trace-vs-production equivalence test on a real observation."""
        logger.info("Running trace-vs-production equivalence gate ...")
        noise = paired_noise(0, 1, self.train_config.model.action_horizon, self.train_config.model.action_dim)
        result = verify_trace_equivalence(
            self.model,
            obs_full,
            num_steps=self.probe.experiment.num_steps,
            noise=jnp.asarray(noise),
            inference_mode=self.sampler.inference_mode,
            sampler=self.sampler,
        )
        logger.info("Equivalence gate PASSED: %s", result)
        return result

    # ------------------------------------------------------------------ #

    def run_sanity_checks(self, obs_full, obs_empty, *, strict: bool = True) -> dict:
        """Checks 1-7 on the first pair (fail-fast by default)."""
        logger.info("Running counterfactual sanity checks on the first pair ...")
        noise_full = paired_noise(
            self.probe.experiment.base_seed,
            1,
            self.train_config.model.action_horizon,
            self.train_config.model.action_dim,
        )
        noise_empty = paired_noise(
            self.probe.experiment.base_seed + 1,
            1,
            self.train_config.model.action_horizon,
            self.train_config.model.action_dim,
        )
        conditions = self.build_conditions(obs_full, obs_empty)

        # Check 1-3, 7 operate on the swapped observations; check 4 on the raw pair.
        for obs, payload in conditions.values():
            if payload:
                assert_swap_clean(payload["base"], obs, self.tactile_keys)
                # donor tactile must actually differ from base tactile
                for key in self.tactile_keys:
                    a = np.asarray(payload["base"].images[key])
                    b = np.asarray(payload["donor"].images[key])
                    if np.array_equal(a, b):
                        raise AssertionError(
                            f"pair base/donor tactile image '{key}' is identical; "
                            "the counterfactual swap would be a no-op"
                        )

        final_actions: dict[str, np.ndarray] = {}
        rng = jax.random.key(0)
        for cond, (obs, _) in conditions.items():
            noise = noise_full if cond.startswith("F") else noise_empty
            trace = self.run_condition(obs, noise, rng)
            final_actions[cond] = np.asarray(trace.final_action)[0]

        # Check 5: repeat F_F with the same noise.
        repeat = self.run_condition(conditions["F_F"][0], noise_full, rng)
        final_actions["F_F_repeat"] = np.asarray(repeat.final_action)[0]

        report = run_sanity_checks(
            obs_full=obs_full,
            obs_empty=obs_empty,
            obs_FF=conditions["F_F"][0],
            obs_FE=conditions["F_E"][0],
            obs_EE=conditions["E_E"][0],
            obs_EF=conditions["E_F"][0],
            tactile_keys=self.tactile_keys,
            noise_full=noise_full,
            noise_empty=noise_empty,
            final_actions=final_actions,
            strict=strict,
        )
        log_sanity_report(report)
        logger.info("Sanity checks PASSED.")
        return report.to_dict()

    # ------------------------------------------------------------------ #

    def run_pair(self, pair, rng) -> tuple[dict, dict]:
        """Run the four conditions for one pair; returns (run_metadatas, metrics_dict)."""
        obs_full, obs_empty = self.get_pair_observations(pair)
        conditions = self.build_conditions(obs_full, obs_empty)

        seed_full = self.probe.experiment.base_seed + 2 * pair.pair_id
        seed_empty = self.probe.experiment.base_seed + 2 * pair.pair_id + 1
        noise_full = paired_noise(
            seed_full, 1, self.train_config.model.action_horizon, self.train_config.model.action_dim
        )
        noise_empty = paired_noise(
            seed_empty, 1, self.train_config.model.action_horizon, self.train_config.model.action_dim
        )

        base_label = {"F": "full", "E": "empty"}
        traces: dict[str, Any] = {}
        run_metadatas: dict[str, dict] = {}
        for cond in CONDITIONS:
            base_is_full = cond.startswith("F")
            tactile_is_full = cond.endswith("F")
            obs = conditions[cond][0]
            noise = noise_full if base_is_full else noise_empty
            trace = self.run_condition(obs, noise, rng)
            traces[cond] = trace

            metadata = {
                "pair_id": pair.pair_id,
                "condition": cond,
                "base_label": base_label[cond[0]],
                "tactile_label": base_label[cond[-1]],
                "base_episode_index": (pair.full if base_is_full else pair.empty).episode_index,
                "base_frame_index": (pair.full if base_is_full else pair.empty).frame_index,
                "tactile_episode_index": (pair.full if tactile_is_full else pair.empty).episode_index,
                "tactile_frame_index": (pair.full if tactile_is_full else pair.empty).frame_index,
                "noise_seed": seed_full if base_is_full else seed_empty,
                "config_name": self.probe.model.config_name,
                "checkpoint_dir": self.probe.model.checkpoint_dir,
                "num_steps": self.probe.experiment.num_steps,
                "action_horizon": self.train_config.model.action_horizon,
                "action_dim": self.train_config.model.action_dim,
            }
            run_id = f"{pair.pair_id:04d}_{cond}"
            run_metadatas[cond] = self.writer.save_run(run_id, metadata, trace)
            logger.info("  %s done", cond)

        # Pair metrics (F_F vs F_E; E_E vs E_F).
        def _final(name: str) -> np.ndarray:
            return np.asarray(traces[name].final_action)[0]

        def _per_step(name: str, attr: str):
            return [np.asarray(getattr(s, attr))[0] for s in traces[name].steps]

        metrics = compute_pair_metrics(
            pair_id=pair.pair_id,
            final_action_FF=_final("F_F"),
            final_action_FE=_final("F_E"),
            final_action_EE=_final("E_E"),
            final_action_EF=_final("E_F"),
            action_hidden_FF=_per_step("F_F", "action_hidden"),
            action_hidden_FE=_per_step("F_E", "action_hidden"),
            v_t_FF=_per_step("F_F", "v_t"),
            v_t_FE=_per_step("F_E", "v_t"),
            x_t_FF=_per_step("F_F", "x_t_before"),
            x_t_FE=_per_step("F_E", "x_t_before"),
        )
        logger.info(
            "[%d/%d] delta_F action RMS: %.6g  delta_E action RMS: %.6g",
            pair.pair_id + 1,
            len(self.probe.pairs),
            metrics.to_dict()["final_action_rms_F"],
            metrics.to_dict()["final_action_rms_E"],
        )
        return run_metadatas, metrics.to_dict()

    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        pair_start: int = 0,
        pair_end: int | None = None,
        max_pairs: int | None = None,
        dry_run: bool = False,
        resume: bool = False,
        skip_strict_validation: bool = False,
    ) -> dict:
        """Run the experiment (or a slice of it)."""
        pairs = self.probe.pairs
        if pair_end is None:
            pair_end = len(pairs)
        pair_end = min(pair_end, len(pairs))
        if max_pairs is not None:
            pair_end = min(pair_end, pair_start + max_pairs)
        selected = pairs[pair_start:pair_end]

        self.log_header()
        self.validate_pairs()

        if not pairs:
            raise ValueError("The probe config defines no pairs; nothing to run.")

        # Load the first pair's observations once for the gates.
        first_pair = pairs[0]
        obs_full, obs_empty = self.get_pair_observations(first_pair)

        # Mandatory equivalence gate.
        eq_result = self.equivalence_gate(obs_full)

        # Tactile token sanity: manual FastViT branch vs embed_suffix tokens.
        self.verify_tactile_tokens(obs_full)

        summary: dict[str, Any] = {
            "probe_config": _config_to_plain(self.probe),
            "equivalence": eq_result,
            "pairs": {},
        }

        if dry_run:
            logger.info("DRY RUN: running first-pair sanity checks only (no experiment loop).")
            summary["sanity_checks"] = self.run_sanity_checks(obs_full, obs_empty, strict=not skip_strict_validation)
            summary["dry_run"] = True
            self.writer.write_summary(summary)
            return summary

        # Sanity checks on the first pair (fail-fast).
        summary["sanity_checks"] = self.run_sanity_checks(obs_full, obs_empty, strict=not skip_strict_validation)

        completed = self.writer.load_progress() if resume else set()
        skipped = len(completed & {p.pair_id for p in selected})
        if skipped:
            logger.info("Resume: skipping %d already-completed pair(s).", skipped)

        rng = jax.random.key(self.probe.experiment.base_seed)
        runs_written = 0
        for pair in selected:
            if pair.pair_id in completed:
                continue
            logger.info("[%d/%d] pair_id=%d", pair.pair_id + 1, len(pairs), pair.pair_id)
            _, metrics = self.run_pair(pair, rng)
            summary["pairs"][str(pair.pair_id)] = metrics
            runs_written += 4
            completed.add(pair.pair_id)

            if runs_written % (4 * self.probe.output.shard_size) == 0:
                self.writer.write_summary(summary)
                self.writer.save_progress(completed, runs_written)
                logger.info(
                    "Checkpoint: %d runs written, progress saved (%d pairs done).", runs_written, len(completed)
                )

        self.writer.write_summary(summary)
        self.writer.save_progress(completed, runs_written)
        logger.info(
            "Done: %d pairs, %d runs written to %s",
            len(completed),
            runs_written,
            self.run_dir,
        )
        return summary

    # ------------------------------------------------------------------ #

    def verify_tactile_tokens(self, obs_full) -> None:
        """Sanity: manual FastViT features/tokens equal the embed_suffix ones."""
        trace = self.sampler(
            jax.random.key(0),
            obs_full,
            num_steps=min(self.probe.experiment.num_steps, 2),
            noise=jnp.asarray(
                paired_noise(1, 1, self.train_config.model.action_horizon, self.train_config.model.action_dim)
            ),
        )
        manual = np.asarray(trace.tactile_tokens)[0]
        from_suffix = np.asarray(trace.tactile_tokens_from_suffix)[0]
        max_diff = float(np.abs(manual.astype(np.float64) - from_suffix.astype(np.float64)).max())
        logger.info("Tactile token check: manual FastViT branch vs embed_suffix max abs diff=%.3e", max_diff)
        np.testing.assert_allclose(manual, from_suffix, rtol=1e-5, atol=1e-6)


def _config_to_plain(probe: ProbeConfig) -> dict:
    return {
        "dataset": dataclasses.asdict(probe.dataset),
        "model": {
            "config_name": probe.model.config_name,
            "checkpoint_dir": probe.model.checkpoint_dir,
            "inline": probe.model.inline,
        },
        "experiment": dataclasses.asdict(probe.experiment),
        "output": dataclasses.asdict(probe.output),
        "num_pairs": len(probe.pairs),
    }
