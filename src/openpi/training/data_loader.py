from collections.abc import Iterator, Sequence
import functools
import logging
import multiprocessing
import os
import time
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar, override

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _convert_tasks_to_dict(tasks):
    """Convert lerobot v3.0 tasks DataFrame to v2.1 compatible dict format.

    In v3.0, tasks is a pandas DataFrame with:
    - Index: task names (strings)
    - Column 'task_index': task indices (integers)

    We convert it to {task_index: task_name} dict format for compatibility.

    Args:
        tasks: Either a DataFrame (v3.0) or dict (v2.1)

    Returns:
        Dictionary mapping task_index (int) to task_name (str)
    """
    # If it's already a dict (v2.1 format), return as is
    if isinstance(tasks, dict):
        return tasks

    # Convert DataFrame (v3.0) to dict
    # DataFrame has task names as index and 'task_index' column
    try:
        import pandas as pd

        if isinstance(tasks, pd.DataFrame):
            # Create dict: {task_index: task_name}
            # Convert task_index to Python int for full compatibility
            return {int(row["task_index"]): task_name for task_name, row in tasks.iterrows()}
    except ImportError:
        pass

    # Fallback: try to convert assuming it has .iterrows() method
    if hasattr(tasks, "iterrows"):
        return {int(row["task_index"]): task_name for task_name, row in tasks.iterrows()}

    # If all else fails, return as is and let it fail later with better error message
    return tasks


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

    def profile_stats(self) -> dict[str, float] | None:
        """Return timings for the most recently yielded batch, when profiling is enabled."""
        return None

    def close(self) -> None:
        """Release worker processes and other loader resources."""


class _TimedDataset:
    """Attach worker-side __getitem__ latency to samples for profiling."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __getitem__(self, index: SupportsIndex):
        started = time.monotonic()
        sample = self._dataset[index]
        return sample, time.monotonic() - started

    def __len__(self) -> int:
        return len(self._dataset)


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


# Video keys whose name contains this marker are treated as tactile streams. A substring
# rather than an exact name because both spellings are on disk: lerobot-xense suffixed the
# tactile cameras by USB enumeration order until `1146d034` and by the jaw the pad sits on
# after it, so a dataset carries either `observation.images.{left,right}_tactile_{0,1}` or
# `..._tactile_{left,right}` depending on when it was recorded. A recorded stream is named
# after its lerobot camera key (see examples/bi_flexiv_rizon4_rt/recorder.py), so this has
# to match whatever that repo emitted at record time.
TACTILE_KEY_MARKER = "tactile"


class SelectiveVideoLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobotDataset that only decodes a whitelisted subset of the video streams.

    `LeRobotDataset.__getitem__` decodes every key in `meta.video_keys`, and H.264 random
    seek is ~200x slower than sequential decode, so streams the model never sees are pure
    waste. Filtering the query timestamps means `_query_videos` is never asked for them.

    `decode_video_keys` is set by `create_torch_dataset` right after construction. It must
    be a plain attribute (not a constructor arg) because the dataset is pickled to spawned
    dataloader workers.

    It defaults to `None`, meaning "no filtering": a construction site that forgets to set it
    behaves exactly like a plain `LeRobotDataset`. The alternative default (an empty
    whitelist) would silently decode nothing and train the model on missing images.
    """

    decode_video_keys: frozenset[str] | None = None

    @override
    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = super()._get_query_timestamps(current_ts, query_indices)
        if self.decode_video_keys is None:
            return query_timestamps
        return {key: ts for key, ts in query_timestamps.items() if key in self.decode_video_keys}


def _repack_source_keys(data_config: _config.DataConfig) -> set[str]:
    """Flat dataset column names that the repack transforms read."""
    return {
        leaf
        for transform in data_config.repack_transforms.inputs
        if isinstance(transform, _transforms.RepackTransform)
        for leaf in jax.tree.leaves(transform.structure)
        if isinstance(leaf, str)
    }


def _resolve_decode_video_keys(data_config: _config.DataConfig, video_keys: Sequence[str]) -> frozenset[str]:
    """Decide which video streams to decode, and fail loudly on a config that needs more."""
    if data_config.tactile:
        return frozenset(video_keys)

    decode_keys = frozenset(key for key in video_keys if TACTILE_KEY_MARKER not in key)
    if skipped := sorted(set(video_keys) - decode_keys):
        logging.info(
            f"tactile=False: skipping video decode for {len(skipped)} of {len(video_keys)} stream(s): {skipped}"
        )
    if missing := sorted((_repack_source_keys(data_config) & set(video_keys)) - decode_keys):
        raise ValueError(
            f"repack_transforms reads video stream(s) {missing}, but tactile=False disabled their decode. "
            "Set `tactile: true` under the data config's `base_config` to decode them."
        )
    return decode_keys


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = SelectiveVideoLeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        # Increase tolerance_s slightly to handle floating-point precision issues in video timestamps.
        # The default 1e-4 can fail when timestamp differences are exactly at the boundary.
        tolerance_s=1e-2,
    )
    dataset.decode_video_keys = _resolve_decode_video_keys(data_config, dataset_meta.video_keys)

    if data_config.prompt_from_task:
        # Convert v3.0 tasks DataFrame to v2.1 compatible dict format
        # In v3.0, tasks is a DataFrame with task names as index and task_index as column
        # We need to convert it to {task_index: task_name} dict format
        tasks_dict = _convert_tasks_to_dict(dataset_meta.tasks)
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks_dict)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> DroidRldsDataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        strict_batch_order=config.strict_batch_order,
        profile_data_pipeline=config.profile_data_pipeline,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    strict_batch_order: bool = False,
    profile_data_pipeline: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        strict_batch_order=strict_batch_order,
        profile_data_pipeline=profile_data_pipeline,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class InfiniteSampler(torch.utils.data.Sampler[int]):
    """Yields dataset indices forever, reshuffling on every pass.

    A finite sampler makes the torch iterator raise `StopIteration` at the end of each epoch,
    which forces `TorchDataLoader.__iter__` to build a new iterator: the prefetch pipeline is
    torn down and refilled, and the training loop waits for a single worker to assemble a
    whole batch on its own. With `num_workers` workers each producing full batches, that is a
    stall of `batch_size x per-sample latency` at every epoch boundary.

    Iterating forever removes the boundary entirely. Nothing is lost by doing so: openpi does
    not checkpoint data loader state (`checkpoints.restore_state` drops it), so the batch
    sequence was never resumable in the first place.
    """

    def __init__(self, num_samples: int, *, shuffle: bool, seed: int):
        self._num_samples = num_samples
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self) -> Iterator[int]:
        epoch = 0
        while True:
            if self._shuffle:
                generator = torch.Generator()
                generator.manual_seed(self._seed + epoch)
                yield from torch.randperm(self._num_samples, generator=generator).tolist()
            else:
                yield from range(self._num_samples)
            epoch += 1

    def __len__(self) -> int:
        """One pass. Only used for `len(DataLoader)`; iteration itself never stops."""
        return self._num_samples


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        strict_batch_order: bool = False,
        profile_data_pipeline: bool = False,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            strict_batch_order: If true, deliver batches in strict sampler order. Reproducible,
                but one slow worker blocks every batch behind it. See TrainConfig.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if profile_data_pipeline:
            dataset = _TimedDataset(dataset)

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._profile_data_pipeline = profile_data_pipeline
        self._last_profile_stats: dict[str, float] | None = None
        self._active_iterator = None

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        # Without an explicit sampler (the JAX path; the PyTorch DDP path passes its own),
        # iterate forever so the epoch boundary never tears the prefetch pipeline down.
        if sampler is None:
            sampler = InfiniteSampler(len(dataset), shuffle=shuffle, seed=seed)

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=False,  # Ordering is the sampler's job; torch forbids both at once.
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            prefetch_factor=16
            if num_workers > 0
            else None,  # stall-fix: 32 over-buffered for cold start, 4 was too low vs slow workers, 16 is the sweet spot
            collate_fn=functools.partial(_collate_fn, profile=profile_data_pipeline),
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            # Hand batches over as workers finish them. In strict order, the one worker that is
            # slow this round blocks every batch queued behind it, which is a stall every
            # `num_workers` steps even with dozens of completed batches already sitting in
            # torch's reorder buffer. Batch order is not meaningful here -- the sampler already
            # shuffles, and it still visits every index exactly once per pass.
            in_order=strict_batch_order,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        try:
            while True:
                t_inner_iter = time.monotonic()
                logging.info("[TorchDataLoader] calling iter(self._data_loader) (spawning workers) ...")
                data_iter = iter(self._data_loader)
                self._active_iterator = data_iter
                logging.info(f"[TorchDataLoader] iter() returned in {time.monotonic() - t_inner_iter:.1f}s")
                while True:
                    if self._num_batches is not None and num_items >= self._num_batches:
                        return
                    t_batch = time.monotonic()
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break  # We've exhausted the dataset. Create a new iterator and start over.
                    dt = time.monotonic() - t_batch
                    worker_stats: dict[str, float] = {}
                    if self._profile_data_pipeline:
                        batch, worker_stats = batch
                    if num_items < 3 or dt > 3.0:
                        logging.info(f"[TorchDataLoader] batch #{num_items} next() took {dt:.2f}s")
                    num_items += 1
                    # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                    t_array = time.monotonic()
                    if self._sharding is not None:
                        batch = jax.tree.map(
                            lambda x: jax.make_array_from_process_local_data(self._sharding, _to_numpy_view(x)),
                            batch,
                        )
                    else:
                        batch = jax.tree.map(torch.as_tensor, batch)
                    array_construct_s = time.monotonic() - t_array

                    h2d_wait_s = 0.0
                    if self._profile_data_pipeline and self._sharding is not None:
                        t_h2d = time.monotonic()
                        jax.block_until_ready(batch)
                        h2d_wait_s = time.monotonic() - t_h2d

                    if self._profile_data_pipeline:
                        self._last_profile_stats = {
                            "main_queue_wait_s": dt,
                            **worker_stats,
                            "jax_array_construct_s": array_construct_s,
                            "h2d_wait_s": h2d_wait_s,
                        }
                    yield batch
        finally:
            self.close()

    def profile_stats(self) -> dict[str, float] | None:
        return self._last_profile_stats

    def close(self) -> None:
        """Shut down the active multiprocessing iterator before interpreter teardown."""
        data_iter = self._active_iterator
        if data_iter is None:
            return

        workers = tuple(getattr(data_iter, "_workers", ()))
        try:
            shutdown_workers = getattr(data_iter, "_shutdown_workers", None)
            if shutdown_workers is not None:
                shutdown_workers()
        except RuntimeError as error:
            # Native libraries used by a worker can abort while reacting to the shutdown
            # signal. PyTorch still terminates all remaining workers in its finally block;
            # handle the resulting SIGCHLD report here instead of leaving it for __del__.
            logging.warning(f"DataLoader worker exited while shutting down: {error}")
        finally:
            for worker in workers:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            self._active_iterator = None
            if getattr(self._data_loader, "_iterator", None) is data_iter:
                self._data_loader._iterator = None


def _to_numpy_view(value):
    """Return a NumPy view of a CPU tensor without copying its shared storage."""
    if isinstance(value, torch.Tensor):
        return value.numpy()
    return np.asarray(value)


def _collate_fn(items, *, profile: bool = False):
    """Collate batch elements into CPU tensors so workers use shared-memory IPC."""
    getitem_s = 0.0
    if profile:
        samples, durations = zip(*items, strict=True)
        items = samples
        getitem_s = sum(durations)
    started = time.monotonic()
    # Some inputs are JAX arrays, so stack through NumPy first. Returning Torch tensors is
    # important here: the DataLoader multiprocessing reducer transfers their storage through
    # shared memory instead of pickling each ~900 MiB NumPy batch into the result queue.
    batch = jax.tree.map(
        lambda *xs: torch.from_numpy(np.stack([np.asarray(x) for x in xs], axis=0)),
        *items,
    )
    if not profile:
        return batch
    return batch, {
        "worker_getitem_s": getitem_s,
        "worker_collate_s": time.monotonic() - started,
    }


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # stall-fix: cap per-worker thread pools so 64 workers don't explode into
    # thousands of runnable threads fighting for 192 logical CPUs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                    batch,
                )


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

    def profile_stats(self) -> dict[str, float] | None:
        profile_stats = getattr(self._data_loader, "profile_stats", None)
        return profile_stats() if profile_stats is not None else None

    def close(self) -> None:
        close = getattr(self._data_loader, "close", None)
        if close is not None:
            close()
