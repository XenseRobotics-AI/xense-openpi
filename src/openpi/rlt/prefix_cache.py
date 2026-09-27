"""On-disk cache of frozen-VLA prefix hidden states for RLT token training.

``scripts/rlt/precompute_prefix.py`` writes it once; ``scripts/rlt/train_token.py``
reads it and never touches the VLA. Layout under the cache root::

    metadata.json   identity (VLA config/checkpoint fingerprints, dataset) and shapes
    hidden.bin      raw bfloat16 (num_frames, seq_len, hidden_dim), row-major, no header
    mask.npy        bool (num_frames, seq_len), True = valid prefix token
    frame_index.npy int64 (num_frames,), dataset index of each cached frame
    done.npy        bool (num_frames,), rows already written (makes precompute resumable)

Only valid token rows are ever written to ``hidden.bin``; padded slots read back
as zeros. On sparse-file filesystems (ext4, xfs) the large padded runs - the
prompt padding at the end of each frame, missing-camera blocks - stay holes and
take no disk space; the filesystem may still allocate small gaps. Copy the cache
with a sparse-aware tool (``cp --sparse=always``, ``rsync -S``) or it inflates
to the dense size.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

FORMAT_VERSION = 1
METADATA_FILE = "metadata.json"
_DTYPE = ml_dtypes.bfloat16

# Keys whose mismatch means the cache was built from a different VLA or dataset.
IDENTITY_KEYS = ("vla_config", "params_fingerprint", "norm_stats_fingerprint", "repo_id", "frame_stride")


def directory_fingerprint(path: pathlib.Path | str) -> str:
    """Cheap content identity of a checkpoint directory.

    Hashes every file's relative path and size, plus the full bytes of files
    under 1 MiB (orbax metadata, norm stats). Survives the directory being
    moved; changes whenever the checkpoint is rewritten.
    """
    root = pathlib.Path(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        size = file.stat().st_size
        digest.update(f"{file.relative_to(root)}:{size}\n".encode())
        if size < 2**20:
            digest.update(file.read_bytes())
    return digest.hexdigest()


def load_metadata(root: pathlib.Path | str) -> dict[str, Any] | None:
    path = pathlib.Path(root) / METADATA_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def _write_metadata(root: pathlib.Path, metadata: dict[str, Any]) -> None:
    tmp = root / f"{METADATA_FILE}.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp.replace(root / METADATA_FILE)


def check_identity(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    """Raise if the cache was built from a different VLA checkpoint, norm stats or dataset."""
    mismatched = {
        key: (metadata.get(key), expected[key]) for key in IDENTITY_KEYS if metadata.get(key) != expected[key]
    }
    if mismatched:
        details = "; ".join(f"{k}: cache={a!r}, expected={b!r}" for k, (a, b) in mismatched.items())
        raise ValueError(f"Prefix cache does not match this config ({details}). Rebuild it with precompute_prefix.py.")


def dense_bytes(num_frames: int, seq_len: int, hidden_dim: int) -> int:
    """Upper bound on the cache size (every prefix slot valid)."""
    return num_frames * seq_len * hidden_dim * np.dtype(_DTYPE).itemsize


class CacheWriter:
    """Creates or reopens a cache and writes rows by position; see module docstring."""

    def __init__(
        self,
        root: pathlib.Path | str,
        *,
        identity: dict[str, Any],
        frame_index: np.ndarray,
        seq_len: int,
        hidden_dim: int,
    ):
        self.root = pathlib.Path(root)
        shape = (len(frame_index), seq_len, hidden_dim)
        existing = load_metadata(self.root)
        if existing is None:
            self.root.mkdir(parents=True, exist_ok=True)
            np.save(self.root / "frame_index.npy", frame_index.astype(np.int64))
            np.save(self.root / "done.npy", np.zeros(len(frame_index), dtype=bool))
            np.lib.format.open_memmap(self.root / "mask.npy", mode="w+", dtype=bool, shape=shape[:2]).flush()
            with open(self.root / "hidden.bin", "wb") as f:
                f.truncate(int(np.prod(shape)) * np.dtype(_DTYPE).itemsize)  # sparse until written
            self.metadata = {
                "format_version": FORMAT_VERSION,
                **identity,
                "num_frames": shape[0],
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "dtype": "bfloat16",
                "complete": False,
            }
            _write_metadata(self.root, self.metadata)
        else:
            check_identity(existing, identity)
            if (existing["num_frames"], existing["seq_len"], existing["hidden_dim"]) != shape:
                raise ValueError(f"Existing cache at {self.root} has a different shape; rebuild with --overwrite.")
            self.metadata = existing
        self.done = np.load(self.root / "done.npy")
        self._hidden = np.memmap(self.root / "hidden.bin", dtype=_DTYPE, mode="r+", shape=shape)
        self._mask = np.load(self.root / "mask.npy", mmap_mode="r+")

    def write(self, rows: np.ndarray, hidden: np.ndarray, mask: np.ndarray) -> None:
        """Store a batch. ``rows`` are cache positions; rewriting a row is idempotent."""
        for row, h, m in zip(rows, hidden, mask, strict=True):
            valid = np.flatnonzero(m)
            self._hidden[row, valid] = h[valid].astype(_DTYPE)
            self._mask[row] = m
            self.done[row] = True

    def flush(self) -> None:
        self._hidden.flush()
        self._mask.flush()
        np.save(self.root / "done.tmp.npy", self.done)
        (self.root / "done.tmp.npy").replace(self.root / "done.npy")

    def finalize(self) -> dict[str, Any]:
        self.flush()
        if not self.done.all():
            raise RuntimeError(f"{int((~self.done).sum())} cache rows were never written.")
        valid_cols = np.flatnonzero(np.asarray(self._mask).any(axis=0))
        self.metadata.update(complete=True, max_valid_len=int(valid_cols[-1]) + 1)
        _write_metadata(self.root, self.metadata)
        return self.metadata


class PrefixCacheDataset:
    """Map-style dataset over a complete cache, trimmed to ``max_valid_len``.

    Items are ``{"hidden": uint16 (seq_len, D) bfloat16 bits, "mask": bool (seq_len,)}``.

    Trailing slots that are padding in every frame are dropped; the token model
    gives identical results with or without them (see ``token_model``).
    """

    def __init__(self, root: pathlib.Path | str):
        self.root = pathlib.Path(root)
        metadata = load_metadata(self.root)
        if metadata is None or not metadata.get("complete"):
            raise FileNotFoundError(f"No complete prefix cache at {self.root}; run scripts/rlt/precompute_prefix.py.")
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError(f"Prefix cache format {metadata['format_version']} != {FORMAT_VERSION}; rebuild it.")
        self.metadata = metadata
        self.seq_len = metadata["max_valid_len"]
        self._shape = (metadata["num_frames"], metadata["seq_len"], metadata["hidden_dim"])
        self._hidden = None  # opened lazily so each data-loader worker maps its own view
        self._mask = None

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, index) -> dict[str, np.ndarray]:
        if self._hidden is None:
            self._hidden = np.memmap(self.root / "hidden.bin", dtype=_DTYPE, mode="r", shape=self._shape)
            self._mask = np.load(self.root / "mask.npy", mmap_mode="r")
        return {
            # The bfloat16 bit pattern as uint16: the training data loader ships samples as torch
            # tensors, which cannot hold numpy bfloat16. `as_bfloat16` restores it on device.
            "hidden": np.array(self._hidden[index, : self.seq_len]).view(np.uint16),
            "mask": np.array(self._mask[index, : self.seq_len]),
        }


def as_bfloat16(bits):
    """Reinterpret the uint16 ``hidden`` bits from ``PrefixCacheDataset`` as bfloat16 (no copy in jit)."""
    return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)


def delete(root: pathlib.Path | str) -> int:
    """Remove a cache directory and return the disk bytes it held. Refuses non-cache dirs."""
    root = pathlib.Path(root)
    if load_metadata(root) is None:
        raise ValueError(f"{root} has no {METADATA_FILE}; refusing to delete it as a prefix cache.")
    freed = sum(p.stat().st_blocks * 512 for p in root.rglob("*") if p.is_file())
    shutil.rmtree(root)
    return freed
