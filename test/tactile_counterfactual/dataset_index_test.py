"""Tests for the (episode, frame) -> row index and paired noise."""

from __future__ import annotations

import numpy as np
import pytest

from test.tactile_counterfactual.dataset_index import build_ep_frame_to_row
from test.tactile_counterfactual.runner import paired_noise


def test_index_maps_episode_frame_to_row():
    # 3 episodes: ep0 frames 0-1 (global 0,1), ep1 frames 0-2 (global 2,3,4),
    # ep2 frames 0-1 (global 5,6).
    episodes = [0, 0, 1, 1, 1, 2, 2]
    index = [0, 1, 2, 3, 4, 5, 6]
    mapping, ep_start, ep_len = build_ep_frame_to_row(episodes, index)

    assert mapping == {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
        (1, 2): 4,
        (2, 0): 5,
        (2, 1): 6,
    }
    assert ep_start == {0: 0, 1: 2, 2: 5}
    assert ep_len == {0: 2, 1: 3, 2: 2}


def test_index_with_nonzero_global_start():
    # Global index column may start anywhere (filtered dataset).
    episodes = [3, 3, 5, 5]
    index = [100, 101, 200, 201]
    mapping, ep_start, ep_len = build_ep_frame_to_row(episodes, index)
    assert mapping == {(3, 0): 0, (3, 1): 1, (5, 0): 2, (5, 1): 3}
    assert ep_start == {3: 100, 5: 200}
    assert ep_len == {3: 2, 5: 2}


def test_index_empty_rejected():
    with pytest.raises(RuntimeError):
        build_ep_frame_to_row([], [])


def test_paired_noise_deterministic_and_paired():
    noise_a1 = paired_noise(42, 1, 50, 32)
    noise_a2 = paired_noise(42, 1, 50, 32)
    noise_b = paired_noise(43, 1, 50, 32)

    assert noise_a1.dtype == np.float32
    assert noise_a1.shape == (1, 50, 32)
    np.testing.assert_array_equal(noise_a1, noise_a2)  # same seed -> identical
    assert not np.array_equal(noise_a1, noise_b)  # different seed -> different

    # Counterfactual pairing rule: seeds for pair i are base+2i and base+2i+1.
    for i in range(3):
        seed_full = 12345 + 2 * i
        seed_empty = 12345 + 2 * i + 1
        assert np.array_equal(paired_noise(seed_full, 1, 50, 32), paired_noise(seed_full, 1, 50, 32))
        assert np.array_equal(paired_noise(seed_empty, 1, 50, 32), paired_noise(seed_empty, 1, 50, 32))
        assert not np.array_equal(paired_noise(seed_full, 1, 50, 32), paired_noise(seed_empty, 1, 50, 32))
