import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from openpi.rlt import prefix_cache
import openpi.training.data_loader as _data_loader

_IDENTITY = {
    "vla_config": "cfg",
    "vla_checkpoint": "/ckpt",
    "params_fingerprint": "p",
    "norm_stats_fingerprint": "n",
    "repo_id": "repo",
    "frame_stride": 2,
}
# 1024 bf16 values = 2 KiB per token row; two rows per 4 KiB page.
_SEQ, _DIM = 16, 1024


def _batch(rows, rng):
    hidden = rng.standard_normal((len(rows), _SEQ, _DIM)).astype(np.float32)
    mask = np.ones((len(rows), _SEQ), dtype=bool)
    mask[:, 10:] = False  # prompt padding, every frame
    mask[0, 2:4] = False  # a missing camera block
    return hidden, mask


def _writer(root, num_frames=6, identity=_IDENTITY):
    return prefix_cache.CacheWriter(
        root, identity=identity, frame_index=np.arange(num_frames) * 2, seq_len=_SEQ, hidden_dim=_DIM
    )


def test_roundtrip_resume_and_trim(tmp_path):
    rng = np.random.default_rng(0)
    writer = _writer(tmp_path)
    first, second = np.array([0, 1, 2]), np.array([3, 4, 5])
    h1, m1 = _batch(first, rng)
    writer.write(first, h1, m1)
    writer.flush()
    del writer

    with pytest.raises(FileNotFoundError):
        prefix_cache.PrefixCacheDataset(tmp_path)  # not complete yet

    writer = _writer(tmp_path)  # resume
    assert writer.done[first].all()
    assert not writer.done[second].any()
    h2, m2 = _batch(second, rng)
    writer.write(second, h2, m2)
    metadata = writer.finalize()
    assert metadata["max_valid_len"] == 10

    dataset = prefix_cache.PrefixCacheDataset(tmp_path)
    assert len(dataset) == 6
    item = dataset[0]
    assert item["hidden"].shape == (10, _DIM)
    np.testing.assert_array_equal(item["mask"], m1[0, :10])
    expected = np.where(m1[0, :10, None], h1[0, :10], 0).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(item["hidden"].view(ml_dtypes.bfloat16), expected)

    # Through the training loader's collate (torch shared-memory tensors) and back to bfloat16.
    batch = _data_loader._collate_fn([dataset[0], dataset[1]])
    hidden = np.asarray(prefix_cache.as_bfloat16(jnp.asarray(batch["hidden"].numpy())))
    assert hidden.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(hidden[0], expected)


def test_trailing_padding_is_not_stored(tmp_path):
    # pi05-like geometry: 4 KiB token rows and a long all-padding tail per frame.
    seq, dim, valid = 256, 2048, 64
    writer = prefix_cache.CacheWriter(
        tmp_path, identity=_IDENTITY, frame_index=np.arange(16), seq_len=seq, hidden_dim=dim
    )
    mask = np.zeros((16, seq), dtype=bool)
    mask[:, :valid] = True
    writer.write(np.arange(16), np.ones((16, seq, dim), np.float32), mask)
    writer.finalize()
    used = (tmp_path / "hidden.bin").stat().st_blocks * 512
    assert used < 0.5 * prefix_cache.dense_bytes(16, seq, dim)


def test_identity_mismatch_is_refused(tmp_path):
    _writer(tmp_path).flush()
    with pytest.raises(ValueError, match="params_fingerprint"):
        _writer(tmp_path, identity={**_IDENTITY, "params_fingerprint": "other"})


def test_delete_refuses_non_cache_dir(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="refusing"):
        prefix_cache.delete(tmp_path)
    cache = tmp_path / "cache"
    _writer(cache).flush()
    assert prefix_cache.delete(cache) > 0
    assert not cache.exists()
    assert (tmp_path / "keep.txt").exists()
