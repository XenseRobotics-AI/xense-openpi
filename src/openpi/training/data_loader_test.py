import dataclasses
import itertools
import unittest.mock

import jax
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug_pi05")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("debug_pi05")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def _bi_flexiv_data_config(*, tactile: bool) -> _config.DataConfig:
    factory = _config.LeRobotBiFlexivDataConfig(
        repo_id="Xense/optical-module-insertion-0731",
        base_config=_config.DataConfig(prompt_from_task=True, tactile=tactile),
    )
    return dataclasses.replace(
        _config.DataConfig(tactile=tactile),
        repack_transforms=factory.repack_transforms,
    )


_XENSE_VIDEO_KEYS = [
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.left_tactile_0",
    "observation.images.left_tactile_1",
    "observation.images.right_tactile_0",
    "observation.images.right_tactile_1",
]


def test_resolve_decode_video_keys_skips_tactile_by_default():
    decode_keys = _data_loader._resolve_decode_video_keys(_bi_flexiv_data_config(tactile=False), _XENSE_VIDEO_KEYS)

    assert decode_keys == {
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }


def test_resolve_decode_video_keys_keeps_tactile_when_enabled():
    decode_keys = _data_loader._resolve_decode_video_keys(_bi_flexiv_data_config(tactile=True), _XENSE_VIDEO_KEYS)

    assert decode_keys == set(_XENSE_VIDEO_KEYS)


def test_resolve_decode_video_keys_rejects_repack_that_needs_tactile():
    data_config = _config.DataConfig(
        tactile=False,
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "head": "observation.images.head",
                            "left_tactile": "observation.images.left_tactile_0",
                        },
                        "state": "observation.state",
                    }
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="observation.images.left_tactile_0"):
        _data_loader._resolve_decode_video_keys(data_config, _XENSE_VIDEO_KEYS)


def test_selective_video_dataset_filters_query_timestamps():
    dataset = _data_loader.SelectiveVideoLeRobotDataset.__new__(_data_loader.SelectiveVideoLeRobotDataset)
    dataset.decode_video_keys = frozenset({"observation.images.head"})

    with unittest.mock.patch.object(
        lerobot_dataset.LeRobotDataset,
        "_get_query_timestamps",
        return_value={key: [0.0] for key in _XENSE_VIDEO_KEYS},
    ):
        assert dataset._get_query_timestamps(0.0) == {"observation.images.head": [0.0]}


def test_infinite_sampler_never_stops_and_permutes_each_pass():
    sampler = _data_loader.InfiniteSampler(10, shuffle=True, seed=0)

    it = iter(sampler)
    drawn = [next(it) for _ in range(35)]

    # Never raises StopIteration, and every pass is a permutation of the full index set,
    # so each sample is still seen exactly once per epoch.
    assert len(drawn) == 35
    for start in (0, 10, 20):
        assert sorted(drawn[start : start + 10]) == list(range(10))
    assert len(sampler) == 10


def test_infinite_sampler_without_shuffle_is_sequential():
    sampler = _data_loader.InfiniteSampler(4, shuffle=False, seed=0)

    it = iter(sampler)

    assert [next(it) for _ in range(10)] == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]


def test_infinite_sampler_is_deterministic_for_a_seed():
    a = list(itertools.islice(iter(_data_loader.InfiniteSampler(16, shuffle=True, seed=7)), 40))
    b = list(itertools.islice(iter(_data_loader.InfiniteSampler(16, shuffle=True, seed=7)), 40))
    c = list(itertools.islice(iter(_data_loader.InfiniteSampler(16, shuffle=True, seed=8)), 40))

    assert a == b
    assert a != c


def test_torch_data_loader_defaults_to_out_of_order_delivery():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2)

    # in_order=False is what removes the head-of-line stall every `num_workers` steps.
    assert loader.torch_loader.in_order is False
    assert isinstance(loader.torch_loader.sampler, _data_loader.InfiniteSampler)


def test_torch_data_loader_strict_batch_order_is_opt_in():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, strict_batch_order=True)

    assert loader.torch_loader.in_order is True


def test_torch_data_loader_keeps_an_explicit_sampler():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)
    sampler = torch.utils.data.SequentialSampler(range(16))

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, sampler=sampler, num_batches=2)

    # The PyTorch DDP path passes its own DistributedSampler; we must not override it.
    assert loader.torch_loader.sampler is sampler
