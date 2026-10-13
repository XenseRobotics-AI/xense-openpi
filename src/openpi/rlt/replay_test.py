import numpy as np
import pytest

from openpi.rlt import critical_trace
from openpi.rlt import replay as _replay
from openpi.rlt import tacxense_reference

Z, S, A, C, R = 4, 3, 2, 3, 5


def _features(i):
    return {
        "z_rl": np.full(Z, i, np.float32),
        "state": np.full(S, i, np.float32),
        "proprio": np.zeros(S, np.float32),
        "ref_chunk": np.zeros((R, A), np.float32),
    }


def _buffer(capacity=4, seed=0):
    return _replay.ReplayBuffer(
        capacity, z_dim=Z, state_dim=S, action_dim=A, num_action_chunks=C, ref_num_action_chunks=R, seed=seed
    )


def _row(i, human=(False, True, False)):
    human = np.asarray(human)
    action_source = np.where(human, _replay.SOURCE_HUMAN, _replay.SOURCE_VLA)
    return {
        "curr_obs": _features(i),
        "next_obs": _features(i + 1),
        "actions": np.full((C, A), i, np.float32),
        "chunk_rewards": np.zeros(C, np.float32),
        "mc_return": float(i),
        "intervention_mask": human,
        "action_source": action_source,
        "terminated": False,
        "success": True,
        "actor_enabled": False,
        "episode_id": 0,
        "round_id": 0,
        "source": _replay.chunk_source(action_source),
        "timestamp": 1790000000.0 + i,
    }


def test_fifo_sampling_and_checkpoint_roundtrip():
    buffer = _buffer()
    for i in range(6):
        buffer.add(_row(i))
    assert len(buffer) == 4
    batch = buffer.sample(64)
    assert set(np.unique(batch["actions"][:, 0, 0])) <= {2.0, 3.0, 4.0, 5.0}
    np.testing.assert_array_equal(batch["next_obs"]["z_rl"][:, 0], batch["curr_obs"]["z_rl"][:, 0] + 1)
    assert buffer.sample(64)["actions"].shape == (4, C, A)

    restored = _buffer(seed=123)
    restored.load_state_dict(buffer.state_dict())
    np.testing.assert_equal(restored.sample(8), buffer.sample(8))


def test_inconsistent_provenance_is_refused():
    row = _row(0)
    row["action_source"] = np.zeros(C, np.int8)
    with pytest.raises(ValueError, match="disagree"):
        _buffer().prepare(row)


@pytest.mark.parametrize(
    ("action_source", "expected"),
    [
        ([0, 0, 0], _replay.SOURCE_VLA),
        ([1, 1, 1], _replay.SOURCE_ACTOR),
        ([2, 2, 2], _replay.SOURCE_HUMAN),
        ([0, 2, 0], _replay.SOURCE_MIXED),
        ([1, 2, 2], _replay.SOURCE_MIXED),
    ],
)
def test_chunk_source(action_source, expected):
    assert _replay.chunk_source(np.asarray(action_source, np.int8)) == expected


def test_chunk_source_disagreeing_with_the_steps_is_refused():
    row = _row(0)  # VLA and human steps: MIXED
    row["source"] = _replay.SOURCE_VLA
    with pytest.raises(ValueError, match="source and action_source disagree"):
        _buffer().prepare(row)


def test_checkpoint_roundtrip_keeps_source_and_timestamp():
    buffer = _buffer()
    for i in range(3):
        buffer.add(_row(i, human=(False, False, False) if i == 1 else (False, True, False)))
    restored = _buffer()
    restored.load_state_dict(buffer.state_dict())
    storage = restored.state_dict()["storage"]
    assert storage["source"].tolist() == [_replay.SOURCE_MIXED, _replay.SOURCE_VLA, _replay.SOURCE_MIXED]
    assert storage["timestamp"].tolist() == [1790000000.0, 1790000001.0, 1790000002.0]


def _trace(length, stride, human_at=()):
    trace = critical_trace.CriticalTrace(C, stride)
    source = np.asarray([_replay.SOURCE_HUMAN if i in human_at else _replay.SOURCE_VLA for i in range(length)])
    trace.extend(
        np.arange(length, dtype=np.float32)[:, None].repeat(A, 1), np.zeros(length), source, actor_enabled=True
    )
    return trace


def test_sliding_windows():
    trace = _trace(10, stride=2, human_at=(8,))
    assert trace.anchors() == [0, 2, 4, 6, 7]
    assert trace.missing_feature_indices() == [0, 2, 3, 4, 5, 6, 7, 9, 10]
    for index in trace.missing_feature_indices():
        trace.add_features(index, _features(index))
    trace.set_terminal_reward(1.0)
    windows = trace.windows()
    assert [w.terminal for w in windows] == [False] * 4 + [True]
    last = windows[-1]
    np.testing.assert_array_equal(last.executed[:, 0], [7, 8, 9])
    np.testing.assert_array_equal(last.rewards, [0, 0, 1])
    np.testing.assert_array_equal(last.human, [False, True, False])
    assert last.next_features["z_rl"][0] == 10


def test_mc_return_is_the_exact_discounted_return():
    trace = _trace(5, stride=1)
    trace.set_terminal_reward(1.0)
    # gamma=0.5: window at start a sees rewards[a:], so the terminal 1.0 lands as 0.5**(4-a).
    expected = {0: 0.5**4, 1: 0.5**3, 2: 0.5**2, 3: 0.5, 4: 1.0}
    for anchor, value in expected.items():
        assert trace.mc_return(anchor, gamma=0.5) == pytest.approx(value)


def test_short_phase_has_no_windows():
    assert _trace(C - 1, stride=2).windows() == []


@pytest.mark.parametrize(("length", "stride"), [(10, 2), (11, 3), (3, 1), (20, 4)])
def test_anchors_match_tacxense(length, stride):
    tacxense_reference.import_tacxense()
    from tacxense.rlt.critical_trace import CriticalTrace

    reference = CriticalTrace(C, stride=stride)
    reference.extend(np.zeros((length, A)), np.zeros(length), np.zeros(length, bool), np.zeros(length, np.int8))
    assert _trace(length, stride).anchors() == reference._sliding_anchor_indices(terminal=True)
