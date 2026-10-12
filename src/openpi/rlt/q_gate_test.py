import dataclasses

import numpy as np
import pytest

from openpi.rlt import action_space_test
from openpi.rlt import config as _config
from openpi.rlt import learner as _learner
from openpi.rlt import q_gate as _q_gate
from openpi.rlt import replay as _replay

Z, S, A, C, R = 8, 20, 20, 4, 6


def _metrics(acc=0.9, margin=1.0, clean=0.95, success=30):
    return {"rank_accuracy": acc, "margin_std": margin, "clean_accuracy": clean, "success_windows": success}


def _gate(**kwargs):
    return _config.QGateConfig(**kwargs)


class TestStateMachine:
    def test_opens_after_sustained_passes(self):
        gate = _q_gate.QGate(_gate(sustain_evals=3))
        assert gate.observe(_metrics(), actor_update=0) is None
        assert gate.observe(_metrics(), actor_update=0) is None
        assert gate.observe(_metrics(), actor_update=10) == "opened"
        assert gate.open and gate.opened_at_update == 10

    def test_a_failed_pass_streak_resets(self):
        gate = _q_gate.QGate(_gate(sustain_evals=2))
        gate.observe(_metrics(), 0)
        gate.observe(_metrics(acc=0.5), 0)  # below the open threshold: streak resets
        assert gate.observe(_metrics(), 5) is None
        assert gate.observe(_metrics(), 6) == "opened"

    def test_hysteresis_holds_between_thresholds(self):
        gate = _q_gate.QGate(_gate(sustain_evals=1, close_evals=2))
        assert gate.observe(_metrics(), 0) == "opened"
        # Between the close and open thresholds: not a failure, the gate stays open.
        assert gate.observe(_metrics(acc=0.7, margin=0.3, clean=0.8), 1) is None
        assert gate.open

    def test_closes_after_consecutive_failures(self):
        gate = _q_gate.QGate(_gate(sustain_evals=1, close_evals=2))
        gate.observe(_metrics(), 0)
        bad = _metrics(acc=0.5)  # below close_rank_accuracy
        assert gate.observe(bad, 1) is None
        assert gate.observe(bad, 2) == "closed"
        assert not gate.open and gate.opened_at_update is None

    def test_failure_streak_resets_on_recovery(self):
        gate = _q_gate.QGate(_gate(sustain_evals=1, close_evals=2))
        gate.observe(_metrics(), 0)
        gate.observe(_metrics(acc=0.5), 1)
        gate.observe(_metrics(), 2)  # healthy again: streak resets
        assert gate.observe(_metrics(acc=0.5), 3) is None
        assert gate.open

    def test_margin_alone_can_close(self):
        gate = _q_gate.QGate(_gate(sustain_evals=1, close_evals=1))
        gate.observe(_metrics(), 0)
        assert gate.observe(_metrics(margin=0.1), 1) == "closed"  # below close_margin_std

    def test_success_window_prerequisite(self):
        gate = _q_gate.QGate(_gate(sustain_evals=1, min_success_windows=50))
        assert gate.observe(_metrics(success=10), 0) is None
        assert not gate.open

    def test_state_dict_roundtrip(self):
        gate = _q_gate.QGate(_gate(sustain_evals=1))
        gate.observe(_metrics(), 7)
        clone = _q_gate.QGate(_gate())
        clone.load_state_dict(gate.state_dict())
        assert clone.open and clone.opened_at_update == 7 and clone.evals == 1


def _learner_with_gate(**gate_overrides):
    cfg = _config.RLConfig(
        num_action_chunks=C,
        ref_num_action_chunks=R,
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        warm_up=4,
        batch_size=8,
        buffer_size=64,
        q_gate=_gate(enable=True, **gate_overrides),
    )
    return _learner.Learner(cfg, action_space_test.bi_flexiv_space(), z_dim=Z)


class TestActorWeights:
    def test_disabled_gate_keeps_config_weights(self):
        cfg = _config.RLConfig(num_action_chunks=C, ref_num_action_chunks=R, q_weight=0.1, bc_weight=5.0)
        learner = _learner.Learner(cfg, action_space_test.bi_flexiv_space(), z_dim=Z)
        assert learner._actor_weights() == (5.0, 0.1)

    def test_closed_gate_holds_warmup_weights(self):
        learner = _learner_with_gate()
        assert learner._actor_weights() == (7.0, 0.05)

    def test_open_gate_ramps_to_online(self):
        learner = _learner_with_gate(ramp_updates=10)
        learner.gate.open = True
        learner.gate.opened_at_update = 0
        learner.counters.actor_updates = 4  # progress (4 + 1) / 10 = 0.5
        bc, q = learner._actor_weights()
        assert bc == pytest.approx(7.0 + 0.5 * (2.5 - 7.0))
        assert q == pytest.approx(0.05 + 0.5 * (0.45 - 0.05))
        learner.counters.actor_updates = 100  # ramp saturated
        assert learner._actor_weights() == (2.5, 0.45)


def _row(rng, space, *, intervened: bool, success: bool, round_id: int, episode_id: int):
    state = np.zeros(S, np.float32)
    state[[3, 7, 12, 16]] = 1.0  # identity rot6d on both arms
    obs = {
        "z_rl": rng.normal(size=Z).astype(np.float32),
        "state": state,
        "proprio": np.asarray(space.normalize_state(state)),
        "ref_chunk": rng.uniform(-0.5, 0.5, (R, A)).astype(np.float32),
    }
    human = np.zeros(C, bool)
    human[:2] = intervened
    action_source = np.where(human, _replay.SOURCE_HUMAN, _replay.SOURCE_VLA).astype(np.int8)
    return {
        "curr_obs": obs,
        "next_obs": obs,
        "actions": rng.uniform(-0.5, 0.5, (C, A)).astype(np.float32),
        "chunk_rewards": np.zeros(C, np.float32),
        "intervention_mask": human,
        "action_source": action_source,
        "terminated": success,
        "success": success,
        "actor_enabled": False,
        "episode_id": episode_id,
        "round_id": round_id,
        "source": _replay.chunk_source(action_source),
        "timestamp": 0.0,
    }


def _fill_replay(learner, *, intervened=8, clean=8, successes=4):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(intervened):
        rows.append(_row(rng, learner.space, intervened=True, success=i < successes, round_id=1, episode_id=i))
    for i in range(clean):
        rows.append(
            _row(rng, learner.space, intervened=False, success=False, round_id=1, episode_id=intervened + i)
        )
    learner.commit(rows)


class TestEvaluateGate:
    def test_skips_when_too_few_windows(self):
        learner = _learner_with_gate(min_windows=8)
        _fill_replay(learner, intervened=4, clean=4)
        metrics, event = learner.evaluate_gate()
        assert event is None and metrics["q_gate/skipped"] == 1.0

    def test_metrics_and_opening_with_a_healthy_critic(self):
        learner = _learner_with_gate(sustain_evals=1, min_windows=2, min_success_windows=1, eval_windows=4)
        _fill_replay(learner)
        # Stub the critic eval: executed always scores above the proposal/random.
        learner._gate_q_raw = lambda critic, obs, a, b: (np.ones(len(a)), np.zeros(len(a)))
        learner._gate_q_canon = lambda critic, obs, a, b: (np.ones(len(a)), np.zeros(len(a)))
        metrics, event = learner.evaluate_gate()
        assert event == "opened"
        assert metrics["q_gate/rank_accuracy"] == 1.0
        assert metrics["q_gate/margin_std"] > 0
        assert metrics["q_gate/clean_accuracy"] == 1.0
        assert metrics["q_gate/open"] == 1.0

    def test_disabled_gate_reports_but_never_opens(self):
        cfg = _config.RLConfig(
            num_action_chunks=C, ref_num_action_chunks=R, buffer_size=64, batch_size=8, warm_up=4
        )
        learner = _learner.Learner(cfg, action_space_test.bi_flexiv_space(), z_dim=Z)
        _fill_replay(learner)
        learner._gate_q_raw = lambda critic, obs, a, b: (np.ones(len(a)), np.zeros(len(a)))
        learner._gate_q_canon = lambda critic, obs, a, b: (np.ones(len(a)), np.zeros(len(a)))
        metrics, event = learner.evaluate_gate()
        assert event is None and not learner.gate.open
        assert metrics["q_gate/rank_accuracy"] == 1.0  # metrics still computed and logged

    def test_real_critic_eval_is_finite(self):
        learner = _learner_with_gate(min_windows=2, min_success_windows=1)
        _fill_replay(learner)
        metrics, _ = learner.evaluate_gate()
        for key in ("rank_accuracy", "margin_std", "clean_accuracy"):
            assert np.isfinite(metrics[f"q_gate/{key}"])
