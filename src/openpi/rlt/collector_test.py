import dataclasses

import numpy as np
import pytest

from openpi.rlt import action_space_test
from openpi.rlt import collector as _collector
from openpi.rlt import config as _rlt_config
from openpi.rlt import learner as _learner
from openpi.rlt import replay as _replay

Z, C, R, STRIDE = 8, 4, 6, 2
_CONFIG = _rlt_config.RLConfig(
    num_action_chunks=C,
    ref_num_action_chunks=R,
    actor_hidden_dims=(16, 16),
    critic_hidden_dims=(16, 16),
    replay_stride=STRIDE,
    warm_up=4,
    utd=2,
    batch_size=8,
    buffer_size=64,
    replay_feature_batch_size=3,
)


class FakeExtractor:
    def __init__(self, space):
        self.space = space
        self.rng = np.random.default_rng(0)
        self.calls = 0

    def extract(self, obs):
        return self.extract_batch([obs])[0]

    def extract_batch(self, observations, *, pad_to=None):
        self.calls += len(observations)
        out = []
        for obs in observations:
            state = np.asarray(obs["state"], np.float32)
            ref = np.asarray(self.space.decode(self.rng.uniform(-0.5, 0.5, (R, 20)).astype(np.float32), state))
            out.append(
                {
                    "z_rl": np.full(Z, obs["t"] if "t" in obs else obs["images"]["t"], np.float32),
                    "state": state,
                    "proprio": np.asarray(self.space.normalize_state(state)),
                    "ref_chunk": np.asarray(self.space.encode(ref, state)),
                    "ref_exec": ref,
                }
            )
        return out


class FakeRobot:
    """Scripted robot side: `plan` lists, per chunk, the segments (length, recording, label, human steps)."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.t = 0
        self.window_steps = 0
        self.recording = False
        self.requests = []

    def _obs(self):
        state = np.zeros(20, np.float32)
        state[3], state[7], state[12], state[16] = 1, 1, 1, 1  # identity rot6d on both arms
        return {"state": state, "images": {}, "t": self.t}

    def request(self, message):
        self.requests.append(message)
        if message["op"] == "reset":
            self.stride = message["capture_stride"]
            return {"obs": self._obs(), "recording": False}
        segments, captures = [], []
        for spec in self.plan.pop(0):
            length, recording, label, human = (
                spec.get("n", C),
                spec.get("rec", False),
                spec.get("label"),
                spec.get("human", ()),
            )
            if recording and not self.recording:
                self.window_steps = 0
            self.recording = recording
            for _ in range(length):
                self.t += 1
                if recording:
                    self.window_steps += 1
                    if self.window_steps % self.stride == 0:
                        captures.append({"step": self.window_steps, "obs": self._obs()})
            segments.append(
                {
                    "obs": self._obs(),
                    "executed": np.asarray(message["actions"])[:length] if length else np.zeros((0, 20)),
                    "human": np.isin(np.arange(length), human),
                    "recording": recording,
                    "label": label,
                    "round_end": spec.get("end", False),
                    "discard": spec.get("discard", False),
                }
            )
        last = spec
        return {
            "segments": segments,
            "captures": captures,
            "recording_next": last.get("next", recording and label is None),
        }

    def status(self, text):
        pass


def _setup(plan):
    space = action_space_test.bi_flexiv_space()
    learner = _learner.Learner(_CONFIG, space, z_dim=Z)
    robot = FakeRobot(plan)
    return _collector.Collector(robot, FakeExtractor(space), learner), robot, learner


# Chunk 1 runs outside a window and opens one for the next chunk; then a 10-step
# phase (with a takeover continuation) is labeled success; the round then ends.
_PLAN = [
    [{"rec": False, "next": True}],
    [{"rec": True, "human": (2, 3)}, {"rec": True, "n": 2, "human": (0, 1)}],
    [{"rec": True, "label": "success"}],
    [{"rec": False, "end": True}],
]


def test_round_turns_a_labeled_phase_into_sliding_windows():
    collector, _, _ = _setup(_PLAN)
    result = collector.run_round()
    rows = result["rows"]
    # 10 phase steps, C=4, stride 2: anchors 0, 2, 4, 6.
    assert len(rows) == 4
    assert [row["terminated"] for row in rows] == [False, False, False, True]
    np.testing.assert_array_equal(rows[-1]["chunk_rewards"], [0, 0, 0, 1])
    np.testing.assert_array_equal(rows[0]["intervention_mask"], [False, False, True, True])
    np.testing.assert_array_equal(rows[1]["action_source"], [_replay.SOURCE_HUMAN] * 4)
    # Window starts at phase steps 0/2/4/6 = robot time 4/6/8/10; each next_obs is C steps later.
    assert [row["curr_obs"]["z_rl"][0] for row in rows] == [4, 6, 8, 10]
    assert [row["next_obs"]["z_rl"][0] for row in rows] == [8, 10, 12, 14]
    assert all(np.abs(row["actions"]).max() <= 1 for row in rows)
    assert result["metrics"]["success"] == 1


def test_unlabeled_and_discarded_windows_leave_no_rows():
    plan = [
        [{"rec": True, "next": False}],  # window closes without a label
        [{"rec": True, "label": "failure"}],
        [{"rec": False, "discard": True, "end": True}],  # ...and the operator discards the round
    ]
    collector, _, _ = _setup(plan)
    assert collector.run_round()["rows"] == []


def test_actor_drives_only_open_windows_after_warm_up(tmp_path):
    collector, robot, learner = _setup(_PLAN * 1)
    learner.commit(collector.run_round()["rows"])
    assert learner.warmed_up
    assert learner.pending_updates() == 4 * _CONFIG.utd
    infos = learner.train()
    assert len(infos) == 8
    assert sum("actor_loss" in info for info in infos) == 4
    assert all(np.isfinite(list(info.values())).all() for info in infos)

    robot.plan = list(_PLAN)
    acted = []
    act = learner.act
    learner.act = lambda features: acted.append(act(features)) or acted[-1]
    before = len(robot.requests)
    collector.run_round()
    learner.act = act
    # The two in-window chunks execute the actor's decoded chunk; the others the VLA reference.
    chunks = [m["actions"] for m in robot.requests[before:] if m["op"] == "chunk"]
    assert len(acted) == 2
    np.testing.assert_array_equal(chunks[1], acted[0])
    np.testing.assert_array_equal(chunks[2], acted[1])

    learner.counters.rounds = 3
    learner.save(tmp_path / "3")
    restored = _learner.Learner(_CONFIG, collector.learner.space, z_dim=Z, seed=9)
    restored.restore(tmp_path / "3")
    assert restored.counters == learner.counters
    assert len(restored.replay) == len(learner.replay)
    features = collector.extractor.extract(robot._obs())
    learner._rng = restored._rng
    np.testing.assert_allclose(restored.act(features), learner.act(features), atol=1e-6)
    np.testing.assert_equal(restored.replay.sample(4), learner.replay.sample(4))


def test_restore_refuses_another_config(tmp_path):
    _, _, learner = _setup([])
    learner.save(tmp_path / "0")
    other = _learner.Learner(dataclasses.replace(_CONFIG, q_weight=0.3), learner.space, z_dim=Z)
    with pytest.raises(ValueError, match="different"):
        other.restore(tmp_path / "0")
