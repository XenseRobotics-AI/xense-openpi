from __future__ import annotations

"""Dynamic data loader for append-only LeRobot dataset trees.

This loader scans a fixed root directory for ready-to-train LeRobot dataset
subdirectories, rebuilds a snapshot on demand, and presents the discovered
datasets as one balanced random-access dataset.

The intended layout is:

    dynamic_root/
      dataset_a/
        meta/info.json
        data/**/*.parquet
      dataset_b/
        meta/info.json
        data/**/*.parquet

New datasets can be dropped into ``dynamic_root`` and picked up after calling
``refresh()`` or starting a new epoch.
"""

import dataclasses
import json
import logging
import pathlib
from typing import SupportsIndex, TypeVar

import numpy as np

import jax
import lerobot.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


def _has_parquet_under(data_dir: pathlib.Path) -> bool:
    return data_dir.is_dir() and any(data_dir.rglob("*.parquet"))


def _is_lerobot_dataset_root(root: pathlib.Path) -> bool:
    return (root / "meta" / "info.json").is_file() and _has_parquet_under(root / "data")



@dataclasses.dataclass(frozen=True)
class DatasetRootInfo:
    root: pathlib.Path
    repo_id: str
    fps: float


@dataclasses.dataclass(frozen=True)
class DatasetSnapshot:
    roots: tuple[DatasetRootInfo, ...]

    @property
    def is_empty(self) -> bool:
        return not self.roots


def _load_root_info(root: pathlib.Path) -> DatasetRootInfo:
    info_path = root / "meta" / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)
    repo_id = info.get("repo_id") or info.get("dataset_id") or root.name
    fps = float(info["fps"])
    return DatasetRootInfo(root=root, repo_id=repo_id, fps=fps)


def discover_dataset_roots(dynamic_root: str | pathlib.Path) -> DatasetSnapshot:
    """Discover ready-to-train LeRobot dataset roots under ``dynamic_root``."""
    base = pathlib.Path(dynamic_root).expanduser().resolve()
    logger.info("[DAGGER SCAN] scanning dynamic_root=%s", base)
    if not base.exists():
        logger.warning("[DAGGER SCAN] dynamic_root does not exist: %s", base)
        return DatasetSnapshot(roots=())

    roots: list[DatasetRootInfo] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not _is_lerobot_dataset_root(child):
            logger.info("[DAGGER SCAN] skip %s: missing meta/info.json or data/**/*.parquet", child)
            continue
        root_info = _load_root_info(child)
        roots.append(root_info)
        logger.info(
            "[DAGGER SCAN] found dataset: name=%s root=%s repo_id=%s fps=%s",
            child.name,
            child,
            root_info.repo_id,
            root_info.fps,
        )

    logger.info("[DAGGER SCAN] complete: found %d ready dataset(s) under %s", len(roots), base)
    return DatasetSnapshot(roots=tuple(roots))


def _convert_tasks_to_dict(tasks):
    if isinstance(tasks, dict):
        return tasks
    try:
        import pandas as pd

        if isinstance(tasks, pd.DataFrame):
            return {int(row["task_index"]): task_name for task_name, row in tasks.iterrows()}
    except ImportError:
        pass
    if hasattr(tasks, "iterrows"):
        return {int(row["task_index"]): task_name for task_name, row in tasks.iterrows()}
    return tasks


class BalancedConcatDatasetView(_data_loader.Dataset[T_co]):
    """Random-access view over multiple datasets with equal per-dataset weight.

    The virtual epoch length is ``max(dataset_lengths) * num_datasets``. Each
    dataset is cycled independently so smaller datasets contribute the same
    number of samples as larger ones.
    """

    def __init__(self, datasets: list[_data_loader.Dataset]):
        self._datasets = list(datasets)
        self._lengths = [len(dataset) for dataset in self._datasets]
        self._num_datasets = len(self._datasets)
        self._samples_per_dataset = max(self._lengths, default=0)
        self._epoch_length = self._samples_per_dataset * self._num_datasets

    def __len__(self) -> int:
        return self._epoch_length

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = index.__index__()
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._num_datasets == 0:
            raise IndexError(idx)

        dataset_idx = idx % self._num_datasets
        local_rank = idx // self._num_datasets
        local_idx = local_rank % self._lengths[dataset_idx]
        return self._datasets[dataset_idx][local_idx]


# Backward-compatible name used by the rest of this module.
ConcatDatasetView = BalancedConcatDatasetView


class AnnotatedDataset(_data_loader.Dataset[T_co]):
    """Attach a dataset source index to each sample."""

    def __init__(self, dataset: _data_loader.Dataset, source_index: int, source_name: str):
        self._dataset = dataset
        self._source_index = source_index
        self._source_name = source_name

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample = self._dataset[index]
        return {
            **sample,
            "_dagger_dataset_source_index": np.asarray(self._source_index, dtype=np.int32),
            "_dagger_dataset_source_name": np.asarray(self._source_name),
        }


class DynamicDatasetRegistry:
    """Tracks the latest dataset snapshot under a root folder."""

    def __init__(self, dynamic_root: str | pathlib.Path):
        self._dynamic_root = pathlib.Path(dynamic_root).expanduser().resolve()
        self._snapshot = discover_dataset_roots(self._dynamic_root)

    @property
    def root(self) -> pathlib.Path:
        return self._dynamic_root

    @property
    def snapshot(self) -> DatasetSnapshot:
        return self._snapshot

    def refresh(self) -> DatasetSnapshot:
        self._snapshot = discover_dataset_roots(self._dynamic_root)
        return self._snapshot


def _build_lerobot_dataset(root_info: DatasetRootInfo, action_horizon: int, data_config: _config.DataConfig) -> _data_loader.Dataset:
    dataset = lerobot_dataset.LeRobotDataset(
        root_info.repo_id,
        root=root_info.root,
        delta_timestamps={
            key: [t / root_info.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        tolerance_s=1e-2,
    )

    if data_config.prompt_from_task:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(root_info.repo_id, root=root_info.root)
        tasks_dict = _convert_tasks_to_dict(dataset_meta.tasks)
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks_dict)])
    return dataset


def build_dynamic_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    dynamic_root: str | pathlib.Path,
) -> ConcatDatasetView:
    """Build a concatenated dataset from all discovered LeRobot dataset roots."""
    registry = DynamicDatasetRegistry(dynamic_root)
    datasets = [
        AnnotatedDataset(
            _data_loader.transform_dataset(
                _build_lerobot_dataset(root_info, action_horizon, data_config),
                data_config,
                skip_norm_stats=False,
            ),
            i,
            root_info.root.name,
        )
        for i, root_info in enumerate(registry.snapshot.roots)
    ]
    return ConcatDatasetView(datasets)


class DynamicTorchDataLoader:
    """Refreshable torch-style loader for append-only dataset roots."""

    def __init__(
        self,
        data_config: _config.DataConfig,
        model_config: _model.BaseModelConfig,
        action_horizon: int,
        batch_size: int,
        dynamic_root: str | pathlib.Path,
        *,
        sharding: jax.sharding.Sharding | None = None,
        skip_norm_stats: bool = False,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        self._data_config = data_config
        self._model_config = model_config
        self._action_horizon = action_horizon
        self._batch_size = batch_size
        self._dynamic_root = pathlib.Path(dynamic_root).expanduser().resolve()
        self._sharding = sharding
        self._skip_norm_stats = skip_norm_stats
        self._shuffle = shuffle
        self._num_batches = num_batches
        self._num_workers = num_workers
        self._seed = seed
        self._framework = framework
        self._registry = DynamicDatasetRegistry(self._dynamic_root)
        self._loader = None
        self._dataset_names: list[str] = []
        self._dataset_counts: dict[str, int] = {}
        self._dataset_cache: dict[str, _data_loader.Dataset] = {}
        self.refresh()

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def snapshot(self) -> DatasetSnapshot:
        return self._registry.snapshot

    @property
    def dataset_names(self) -> list[str]:
        return list(self._dataset_names)

    def dataset_usage_counts(self) -> dict[str, int]:
        return dict(self._dataset_counts)

    def refresh(self) -> DatasetSnapshot:
        """Re-scan the dynamic root and rebuild the underlying loader."""
        snapshot = self._registry.refresh()
        if snapshot.is_empty:
            logging.warning("No ready LeRobot datasets found under %s", self._dynamic_root)
            if self._dataset_names:
                logger.info("[DAGGER BUILD] all datasets disappeared; clearing loader")
                self._dataset_names = []
                self._dataset_cache.clear()
                self._loader = None
            return snapshot

        new_dataset_names = [root_info.root.name for root_info in snapshot.roots]
        if self._loader is not None and new_dataset_names == self._dataset_names:
            logger.info("[DAGGER BUILD] no dataset-name changes; reusing existing loader")
            return snapshot
        self._dataset_names = new_dataset_names
        logger.info(
            "[DAGGER BUILD] refreshing %d dataset(s): %s",
            len(snapshot.roots),
            ", ".join(self._dataset_names),
        )
        active_names = set(self._dataset_names)
        for removed_name in set(self._dataset_cache) - active_names:
            logger.info("[DAGGER BUILD] removing dataset from loader: name=%s", removed_name)
            del self._dataset_cache[removed_name]

        datasets = []
        for i, root_info in enumerate(snapshot.roots):
            cached = self._dataset_cache.get(root_info.root.name)
            if cached is not None:
                dataset = cached
                logger.info(
                    "[DAGGER BUILD] reuse dataset %d/%d: name=%s samples=%d",
                    i + 1,
                    len(snapshot.roots),
                    root_info.root.name,
                    len(dataset),
                )
            else:
                logger.info(
                    "[DAGGER BUILD] start dataset %d/%d: name=%s root=%s",
                    i + 1,
                    len(snapshot.roots),
                    root_info.root.name,
                    root_info.root,
                )
                dataset = _build_lerobot_dataset(root_info, self._action_horizon, self._data_config)
                dataset = _data_loader.transform_dataset(
                    dataset,
                    self._data_config,
                    skip_norm_stats=self._skip_norm_stats,
                )
                self._dataset_cache[root_info.root.name] = dataset
                logger.info(
                    "[DAGGER BUILD] ready dataset %d/%d: name=%s samples=%d",
                    i + 1,
                    len(snapshot.roots),
                    root_info.root.name,
                    len(dataset),
                )
            datasets.append(AnnotatedDataset(dataset, i, root_info.root.name))
        dataset = ConcatDatasetView(datasets)

        local_batch_size = self._batch_size // jax.process_count() if self._framework != "pytorch" else self._batch_size
        logger.info(
            "[DAGGER BUILD] creating loader: datasets=%d samples=%d batch_size=%d num_workers=%d shuffle=%s",
            len(datasets),
            len(dataset),
            local_batch_size,
            self._num_workers,
            self._shuffle,
        )
        self._loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=self._sharding if self._framework != "pytorch" else None,
            shuffle=self._shuffle,
            num_batches=self._num_batches,
            num_workers=self._num_workers,
            seed=self._seed,
            framework=self._framework,
        )
        return snapshot

    def __iter__(self):
        if self._loader is None:
            self.refresh()
        if self._loader is None:
            return

        epoch = 0
        while True:
            # Re-read the loader every epoch so a refresh() that swapped it in is
            # picked up at the epoch boundary without restarting this iterator.
            loader = self._loader
            if loader is None:
                logger.info("[DAGGER LOADER] no datasets left; stopping iteration")
                return

            num_batches = 0
            for batch in loader.torch_loader:
                # These annotation keys are bookkeeping only and must not reach the model.
                source_idx = batch.pop("_dagger_dataset_source_index", None)
                batch.pop("_dagger_dataset_source_name", None)
                if source_idx is not None:
                    values = np.asarray(source_idx).reshape(-1)
                    for idx in values:
                        if 0 <= int(idx) < len(self._dataset_names):
                            name = self._dataset_names[int(idx)]
                            self._dataset_counts[name] = self._dataset_counts.get(name, 0) + 1
                num_batches += 1
                yield _model.Observation.from_dict(batch), batch["actions"]

            if num_batches == 0:
                # Nothing was produced (e.g. every dataset shrank below the batch
                # size). Returning beats spinning in a tight, silent loop.
                logger.error("[DAGGER LOADER] epoch %d produced no batches; stopping iteration", epoch)
                return

            epoch += 1
            logger.info("[DAGGER LOADER] dataset exhausted after %d batches; starting epoch %d", num_batches, epoch)


def create_dynamic_data_loader(
    config: _config.TrainConfig,
    dynamic_root: str | pathlib.Path,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "jax",
) -> DynamicTorchDataLoader:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_config)
    return DynamicTorchDataLoader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        dynamic_root=dynamic_root,
        sharding=sharding,
        skip_norm_stats=skip_norm_stats,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
