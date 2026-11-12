import dataclasses
import time

import jax
import numpy as np
import pytest

from openpi.rlt import action_space_test
from openpi.rlt import config as _rlt_config
from openpi.rlt import learner as _learner
from openpi.rlt import replay as _replay

Z, C, R = 8, 4, 6
_CONFIG = _rlt_config.RLConfig(
    num_action_chunks=C,
    ref_num_action_chunks=R,
    actor_hidden_dims=(16, 16),
    critic_hidden_dims=(16, 16),
    replay_stride=2,
    warm_up=4,
    utd=3,
    batch_size=8,
    buffer_size=64,
)


def _state():
    state = np.zeros(20, np.float32)
    state[3], state[7], state[12], state[16] = 1, 1, 1, 1  # identity rot6d on both arms
    return state


def _features(rng, space):
    state = _state()
    return {
        "z_rl": rng.normal(size=Z).astype(np.float32),
        "state": state,
        "proprio": np.asarray(space.normalize_state(state)),
        "ref_chunk": rng.uniform(-0.5, 0.5, (R, 20)).astype(np.float32),
    }


def _rows(n, space, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        source = np.full(C, _replay.SOURCE_VLA)
        rows.append(
            {
                "curr_obs": _features(rng, space),
                "next_obs": _features(rng, space),
                "actions": rng.uniform(-0.5, 0.5, (C, 20)).astype(np.float32),
                "chunk_rewards": np.zeros(C, np.float32),
                "intervention_mask": np.zeros(C, bool),
                "action_source": source,
                "terminated": i == n - 1,
                "success": True,
                "actor_enabled": False,
                "episode_id": 1,
                "round_id": 1,
                "source": _replay.chunk_source(source),
                "timestamp": 1790000000.0 + i,
            }
        )
    return rows


def _make(seed=0):
    space = action_space_test.bi_flexiv_space()
    return _learner.Learner(_CONFIG, space, z_dim=Z, seed=seed), space


def _params(learner):
    return jax.device_get({"actor": learner.actor, "critic": learner.critic, "target": learner.target_critic})


def _wait(condition, timeout_s=120.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_train_spends_the_earned_budget():
    learner, space = _make()
    learner.commit(_rows(3, space))
    assert not learner.warmed_up
    assert learner.train() == []  # nothing is earned before warm_up

    learner.commit(_rows(3, space, seed=1))
    infos = learner.train()
    assert len(infos) == 6 * _CONFIG.utd
    assert learner.counters.critic_updates == 6 * _CONFIG.utd
    assert learner.counters.actor_updates == 6 * _CONFIG.utd // _CONFIG.critic_actor_ratio
    assert sum("actor_loss" in info for info in infos) == learner.counters.actor_updates
    assert learner.pending_updates() == 0


def test_background_trainer_matches_synchronous_training():
    sync, space = _make()
    rows = _rows(6, space)
    sync.commit(rows)
    sync_infos = sync.train()

    learner, _ = _make()
    trainer = _learner.BackgroundTrainer(learner, idle_poll_s=0.01).start()
    learner.commit(rows)
    trainer.notify()
    trainer.stop(finish_backlog=True)
    updates = trainer.drain()

    assert [index for index, _ in updates] == list(range(1, len(sync_infos) + 1))
    assert dataclasses.asdict(learner.counters) == dataclasses.asdict(sync.counters)
    np.testing.assert_allclose(updates[-1][1]["critic_loss"], sync_infos[-1]["critic_loss"], rtol=1e-5)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6), _params(learner), _params(sync))


def test_acting_and_committing_while_training():
    learner, space = _make()
    trainer = _learner.BackgroundTrainer(learner, idle_poll_s=0.01).start()
    learner.commit(_rows(6, space))
    trainer.notify()
    features = _features(np.random.default_rng(5), space)
    # Collection keeps acting (and the next round commits) while the budget is spent.
    while learner.counters.critic_updates < 6:
        normalized, absolute = learner.act(features)
        assert np.isfinite(normalized).all()
        assert np.isfinite(absolute).all()
        assert np.isfinite(learner.mean(features)).all()
    learner.commit(_rows(2, space, seed=2))
    trainer.notify()
    trainer.stop(finish_backlog=True)
    updates = trainer.drain()
    assert [index for index, _ in updates] == list(range(1, 8 * _CONFIG.utd + 1))
    assert learner.pending_updates() == 0


def test_interrupted_trainer_leaves_the_backlog_for_resume(tmp_path):
    learner, space = _make()
    trainer = _learner.BackgroundTrainer(learner, idle_poll_s=0.01).start()
    learner.commit(_rows(64, space))
    trainer.notify()
    _wait(lambda: learner.counters.critic_updates > 0)
    trainer.stop(finish_backlog=False)
    done = learner.counters.critic_updates
    assert len(trainer.drain()) == done
    assert done + learner.pending_updates() == 64 * _CONFIG.utd

    learner.save(tmp_path / "ckpt")
    resumed, _ = _make(seed=7)
    resumed.restore(tmp_path / "ckpt")
    assert resumed.pending_updates() == learner.pending_updates()
    jax.tree.map(np.testing.assert_array_equal, _params(resumed), _params(learner))


def test_trainer_failure_is_reraised(monkeypatch):
    learner, space = _make()

    def broken(*args):
        raise FloatingPointError("boom")

    monkeypatch.setattr(learner, "_critic_step", broken)
    trainer = _learner.BackgroundTrainer(learner, idle_poll_s=0.01).start()
    learner.commit(_rows(6, space))
    trainer.notify()
    _wait(lambda: learner._background_error is not None)
    features = _features(np.random.default_rng(0), space)
    with pytest.raises(RuntimeError, match="background trainer failed"):
        learner.act(features)
    with pytest.raises(RuntimeError, match="background trainer failed"):
        trainer.drain()
    trainer.stop(finish_backlog=True)  # returns: the thread has already exited


def test_async_training_may_change_on_resume(tmp_path):
    learner, space = _make()
    learner.save(tmp_path / "ckpt")
    toggled = dataclasses.replace(_CONFIG, async_training=not _CONFIG.async_training)
    _learner.Learner(toggled, space, z_dim=Z).restore(tmp_path / "ckpt")


def test_default_chunk_length_follows_the_paper():
    assert _rlt_config.RLConfig().num_action_chunks == 10


def test_first_eviction_is_logged(caplog):
    learner, space = _make()
    learner.replay = _learner._replay.ReplayBuffer(
        6, z_dim=Z, state_dim=20, action_dim=20, num_action_chunks=C, ref_num_action_chunks=R
    )
    learner.commit(_rows(6, space))
    assert "Replay is full" not in caplog.text
    learner.commit(_rows(2, space, seed=1))
    assert "Replay is full" in caplog.text
    assert learner.replay.evicted == 2


def test_default_replay_keeps_hours_of_data():
    config = _rlt_config.RLConfig()
    # 30 Hz, one window every replay_stride steps.
    assert config.buffer_size / (30 / config.replay_stride) / 3600 > 1.5
