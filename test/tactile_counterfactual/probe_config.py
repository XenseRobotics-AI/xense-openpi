"""Probe YAML schema: parsing and static validation.

The probe spec is a standalone YAML file (not a TrainConfig) that describes
the dataset, the trained checkpoint, the experiment parameters and the
explicit full/empty pairs:

    dataset:
      repo_id: Xense/bottle-sorting-0810
      revision: null
      root: null
    model:
      config_name: pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100
      checkpoint_dir: checkpoints/.../39999
    experiment:
      inference_mode: rtc
      num_steps: 10
      base_seed: 12345
      batch_size: 1
    output:
      dir: outputs/tactile_counterfactual
      shard_size: 50
      ...
    pairs:
      - pair_id: 0
        full: {episode_index: 10, frame_index: 245}
        empty: {episode_index: 510, frame_index: 231}
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

INFERENCE_MODES = ("standard", "rtc")
STORAGE_DTYPES = ("float16", "bfloat16", "float32")


@dataclasses.dataclass(frozen=True)
class FrameRef:
    episode_index: int
    frame_index: int

    def key(self) -> tuple[int, int]:
        return (self.episode_index, self.frame_index)


@dataclasses.dataclass(frozen=True)
class PairSpec:
    pair_id: int
    full: FrameRef
    empty: FrameRef


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    repo_id: str | None = None
    revision: str | None = None
    root: str | None = None


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    # Optional name of a TrainConfig in this repo (configs/*.yaml or a
    # generated config). When set, the TrainConfig provides the model config,
    # the data config factory and the norm-stats asset id. When unset, the
    # model config must be inlined via `inline` (type: Pi0TactileFastVitConfig
    # plus fields, resolved by the openpi YAML loader).
    config_name: str | None = None
    # Path to the trained checkpoint step directory (e.g. .../39999).
    checkpoint_dir: str | None = None
    # Optional inline model config mapping (openpi polymorphic form).
    inline: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class ExperimentSpec:
    # "rtc" (training-time RTC sampler, what the deployment policy uses when
    # the model was trained with enable_training_time_rtc=True) or "standard".
    inference_mode: str = "rtc"
    num_steps: int = 10
    base_seed: int = 12345
    batch_size: int = 1


@dataclasses.dataclass(frozen=True)
class OutputSpec:
    dir: str = "outputs/tactile_counterfactual"
    shard_size: int = 50

    save_fastvit_features: bool = True
    save_tactile_tokens: bool = True
    save_suffix_tokens: bool = False
    save_adarms_cond: bool = True
    save_action_hidden: bool = True
    save_v_t: bool = True
    save_x_t: bool = True
    # Storage dtype for the bulky per-step traces (action_hidden,
    # suffix_tokens, 3D adarms_cond). They are written to .npz attachments;
    # everything else goes into the human-readable run YAML.
    hidden_storage_dtype: str = "float16"


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    dataset: DatasetSpec
    model: ModelSpec
    experiment: ExperimentSpec
    output: OutputSpec
    pairs: tuple[PairSpec, ...]
    # Optional inline data config (openpi polymorphic form, e.g.
    # {type: LeRobotBiFlexivTactileDataConfig, repo_id: ...}). Only needed
    # when model.config_name is not set (no TrainConfig to inherit data from).
    data: dict[str, Any] | None = None


def load_probe_config(path: str | pathlib.Path) -> ProbeConfig:
    """Parse and statically validate a probe YAML file."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Probe config not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Probe config root must be a mapping, got {type(raw).__name__}: {path}")
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> ProbeConfig:
    """Build a ProbeConfig from a plain dict (used by tests too)."""
    if "pairs" not in raw:
        raise ValueError("Probe config requires a 'pairs' list")

    dataset_raw = raw.get("dataset") or {}
    model_raw = raw.get("model") or {}
    experiment_raw = raw.get("experiment") or {}
    output_raw = raw.get("output") or {}

    pairs = tuple(_parse_pair(item, i) for i, item in enumerate(raw["pairs"]))

    dataset = DatasetSpec(
        repo_id=dataset_raw.get("repo_id"),
        revision=dataset_raw.get("revision"),
        root=dataset_raw.get("root"),
    )
    model = ModelSpec(
        config_name=model_raw.get("config_name"),
        checkpoint_dir=model_raw.get("checkpoint_dir"),
        inline=model_raw.get("inline") or model_raw.get("model"),
    )
    experiment = ExperimentSpec(
        inference_mode=experiment_raw.get("inference_mode", "rtc"),
        num_steps=int(experiment_raw.get("num_steps", 10)),
        base_seed=int(experiment_raw.get("base_seed", 12345)),
        batch_size=int(experiment_raw.get("batch_size", 1)),
    )
    output = OutputSpec(
        dir=output_raw.get("dir", "outputs/tactile_counterfactual"),
        shard_size=int(output_raw.get("shard_size", 50)),
        save_fastvit_features=bool(output_raw.get("save_fastvit_features", True)),
        save_tactile_tokens=bool(output_raw.get("save_tactile_tokens", True)),
        save_suffix_tokens=bool(output_raw.get("save_suffix_tokens", False)),
        save_adarms_cond=bool(output_raw.get("save_adarms_cond", True)),
        save_action_hidden=bool(output_raw.get("save_action_hidden", True)),
        save_v_t=bool(output_raw.get("save_v_t", True)),
        save_x_t=bool(output_raw.get("save_x_t", True)),
        hidden_storage_dtype=output_raw.get("hidden_storage_dtype", "float16"),
    )

    cfg = ProbeConfig(
        dataset=dataset,
        model=model,
        experiment=experiment,
        output=output,
        pairs=pairs,
        data=raw.get("data"),
    )
    validate(cfg)
    return cfg


def _parse_pair(item: Any, index: int) -> PairSpec:
    if not isinstance(item, dict):
        raise ValueError(f"pairs[{index}] must be a mapping, got {type(item).__name__}")
    if "pair_id" not in item:
        raise ValueError(f"pairs[{index}] is missing 'pair_id'")
    if "full" not in item or "empty" not in item:
        raise ValueError(f"pairs[{index}] requires both 'full' and 'empty' entries")
    return PairSpec(
        pair_id=_parse_int(item, "pair_id", index),
        full=_parse_frame(item["full"], f"pairs[{index}].full"),
        empty=_parse_frame(item["empty"], f"pairs[{index}].empty"),
    )


def _parse_int(item: dict[str, Any], key: str, index: int) -> int:
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"pairs[{index}].{key} must be an integer, got {value!r}")
    return value


def _parse_frame(item: Any, where: str) -> FrameRef:
    if not isinstance(item, dict):
        raise ValueError(f"{where} must be a mapping with episode_index/frame_index, got {type(item).__name__}")
    if "episode_index" not in item or "frame_index" not in item:
        raise ValueError(f"{where} requires 'episode_index' and 'frame_index'")
    ep = item["episode_index"]
    fr = item["frame_index"]
    if isinstance(ep, bool) or not isinstance(ep, int) or ep < 0:
        raise ValueError(f"{where}.episode_index must be a non-negative integer, got {ep!r}")
    if isinstance(fr, bool) or not isinstance(fr, int) or fr < 0:
        raise ValueError(f"{where}.frame_index must be a non-negative integer, got {fr!r}")
    return FrameRef(episode_index=ep, frame_index=fr)


def validate(cfg: ProbeConfig) -> None:
    """Static validation of the probe config (no dataset access needed)."""
    errors: list[str] = []

    if cfg.model.checkpoint_dir is None or not str(cfg.model.checkpoint_dir).strip():
        errors.append("model.checkpoint_dir is required")
    if cfg.model.config_name is None and cfg.model.inline is None:
        errors.append("model.config_name or an inline model config is required")
    if cfg.model.config_name is not None and cfg.model.inline is not None:
        errors.append("model.config_name and inline model config are mutually exclusive")
    if cfg.model.config_name is None and cfg.data is None:
        errors.append("an inline 'data' block is required when model.config_name is not set")

    if cfg.experiment.inference_mode not in INFERENCE_MODES:
        errors.append(
            f"experiment.inference_mode must be one of {INFERENCE_MODES}, got {cfg.experiment.inference_mode!r}"
        )
    if cfg.experiment.num_steps < 1:
        errors.append(f"experiment.num_steps must be >= 1, got {cfg.experiment.num_steps}")
    if cfg.experiment.batch_size < 1:
        errors.append(f"experiment.batch_size must be >= 1, got {cfg.experiment.batch_size}")
    if not isinstance(cfg.experiment.base_seed, int):
        errors.append(f"experiment.base_seed must be an integer, got {cfg.experiment.base_seed!r}")

    if cfg.output.shard_size < 1:
        errors.append(f"output.shard_size must be >= 1, got {cfg.output.shard_size}")
    if cfg.output.hidden_storage_dtype not in STORAGE_DTYPES:
        errors.append(
            f"output.hidden_storage_dtype must be one of {STORAGE_DTYPES}, got {cfg.output.hidden_storage_dtype!r}"
        )

    seen_ids: set[int] = set()
    for pair in cfg.pairs:
        if pair.pair_id in seen_ids:
            errors.append(f"duplicate pair_id {pair.pair_id}")
        seen_ids.add(pair.pair_id)

    # Warn (not error) on repeated (episode, frame) across the pair table.
    seen_frames: dict[tuple[int, int], list[str]] = {}
    for pair in cfg.pairs:
        for role, ref in (("full", pair.full), ("empty", pair.empty)):
            seen_frames.setdefault(ref.key(), []).append(f"pair {pair.pair_id} {role}")
    for key, uses in seen_frames.items():
        if len(uses) > 1:
            print(f"[probe config] WARNING: (episode, frame) {key} appears more than once: {', '.join(uses)}")

    if errors:
        raise ValueError("Invalid probe config:\n  - " + "\n  - ".join(errors))
