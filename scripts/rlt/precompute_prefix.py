"""RLT phase one, step 1: cache the frozen VLA's prefix hidden states.

Runs the VLA once over the dataset (every ``token_training.frame_stride``-th
frame) and writes each frame's prefix hidden states and mask to
``token_training.prefix_cache_dir`` (format: ``openpi.rlt.prefix_cache``).
``scripts/rlt/train_token.py`` then trains on the cache without the VLA.

Single process, all local GPUs: the batch is sharded across devices, the VLA
params are replicated. Interrupted runs resume where they stopped; a cache
built from a different VLA checkpoint, norm stats or dataset is refused
(``--overwrite`` rebuilds it).

    mamba activate lerobot-xense
    python scripts/rlt/precompute_prefix.py <rlt_config> [--batch-size 64]
"""

import argparse
import logging
import math
import pathlib
import shutil
import time

import flax.nnx as nnx
import jax
import numpy as np
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
from openpi.rlt import config as _rlt_config
from openpi.rlt import prefix_cache
from openpi.rlt import vla as _vla
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

_FLUSH_EVERY = 50  # batches


class _PendingFrames:
    """Yields ``{"row": cache row, **transformed sample}`` for rows not cached yet.

    Padded up to ``min_len`` by repeating rows: writes are idempotent, and the
    loader needs at least one full batch.
    """

    def __init__(self, dataset, frame_index: np.ndarray, rows: np.ndarray, min_len: int):
        self._dataset, self._frame_index, self._rows = dataset, frame_index, rows
        self._len = max(len(rows), min_len)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i):
        row = self._rows[i % len(self._rows)]
        sample = dict(self._dataset[int(self._frame_index[row])])
        sample.pop("actions", None)
        return {"row": np.int64(row), **sample}


def _prefix_shape(model: _model.BaseModel, model_config: _model.BaseModelConfig) -> tuple[int, int]:
    hidden, _ = jax.eval_shape(model.extract_prefix_hidden, model_config.fake_obs(1))
    return hidden.shape[1], hidden.shape[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="RLT config name (configs/rlt/<name>.yaml)")
    parser.add_argument("--batch-size", type=int, default=64, help="Global VLA batch; a multiple of the device count")
    parser.add_argument("--num-workers", type=int, default=None, help="Default: token_training.num_workers")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Cache only the first N strided frames (smoke tests)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing cache first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    token_training = _rlt_config.get_config(args.config).token_training
    if args.batch_size % jax.device_count():
        raise ValueError(f"--batch-size {args.batch_size} must be a multiple of {jax.device_count()} devices.")
    root = pathlib.Path(token_training.prefix_cache_dir).resolve()
    if args.overwrite and root.exists():
        prefix_cache.delete(root)

    frozen = _vla.resolve(token_training)
    model_config = frozen.train_config.model
    dataset = _data_loader.transform_dataset(
        _data_loader.create_torch_dataset(frozen.data_config, model_config.action_horizon, model_config),
        frozen.data_config,
    )
    frame_index = np.arange(0, len(dataset), token_training.frame_stride)[: args.max_frames]
    logging.info("Dataset %s: %d frames, caching %d", frozen.data_config.repo_id, len(dataset), len(frame_index))

    model = frozen.load_model()
    seq_len, hidden_dim = _prefix_shape(model, model_config)
    writer = prefix_cache.CacheWriter(
        root,
        identity=frozen.cache_identity(token_training),
        frame_index=frame_index,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
    )
    pending = np.flatnonzero(~writer.done)
    if len(pending) == 0:
        logging.info("All %d frames already cached.", len(frame_index))
    else:
        need = prefix_cache.dense_bytes(len(pending), seq_len, hidden_dim)
        free = shutil.disk_usage(root).free
        logging.info(
            "Caching %d frames of (%d, %d) bf16: at most %.1f GiB (padding stays sparse), %.1f GiB free at %s",
            len(pending), seq_len, hidden_dim, need / 2**30, free / 2**30, root,
        )  # fmt: skip
        if need > free:
            raise OSError(f"Not enough disk at {root}: need up to {need / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free.")
        _run(args, token_training, model, writer, dataset, frame_index, pending)

    metadata = writer.finalize()
    used = sum(p.stat().st_blocks * 512 for p in root.iterdir())
    logging.info(
        "Cache complete: %s (%.1f GiB on disk, max_valid_len=%d)", root, used / 2**30, metadata["max_valid_len"]
    )


def _run(args, token_training, model, writer, dataset, frame_index, pending) -> None:
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    graphdef, state = nnx.split(model)
    state = jax.device_put(state, replicated)
    extract = jax.jit(
        lambda state, obs: nnx.merge(graphdef, state).extract_prefix_hidden(obs),
        in_shardings=(replicated, data_sharding),
        out_shardings=data_sharding,
    )

    num_batches = math.ceil(len(pending) / args.batch_size)
    loader = _data_loader.TorchDataLoader(
        _PendingFrames(dataset, frame_index, pending, args.batch_size),
        local_batch_size=args.batch_size,
        sharding=data_sharding,
        shuffle=False,
        num_batches=num_batches,
        num_workers=token_training.num_workers if args.num_workers is None else args.num_workers,
        # The loop stops after num_batches; out-of-order delivery could swap a pending
        # batch for a wrapped-around repeat and leave rows unwritten.
        strict_batch_order=True,
    )

    def store(result) -> None:
        rows, hidden, mask = jax.device_get(result)
        writer.write(rows, hidden, mask)

    in_flight = None
    start = time.monotonic()
    for i, batch in enumerate(tqdm.tqdm(loader, total=num_batches, desc="Precompute prefix")):
        # Dispatch this batch before writing the previous one, so disk writes overlap the VLA forward.
        result = (batch.pop("row"), *extract(state, _model.Observation.from_dict(batch)))
        if in_flight is not None:
            store(in_flight)
        in_flight = result
        if (i + 1) % _FLUSH_EVERY == 0:
            writer.flush()
    store(in_flight)
    elapsed = time.monotonic() - start
    logging.info("Precomputed %d frames in %.0fs (%.1f frames/s)", len(pending), elapsed, len(pending) / elapsed)


if __name__ == "__main__":
    main()
