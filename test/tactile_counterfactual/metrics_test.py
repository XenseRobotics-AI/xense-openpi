"""Tests for pair sensitivity metrics."""

from __future__ import annotations

import numpy as np

from test.tactile_counterfactual.metrics import compute_pair_metrics

AH, AD, EMB, STEPS = 4, 3, 8, 2


def _seq(value: float) -> list[np.ndarray]:
    return [np.full((AH, EMB), value, dtype=np.float32) for _ in range(STEPS)]


def _x(value: float) -> list[np.ndarray]:
    return [np.full((AH, AD), value, dtype=np.float32) for _ in range(STEPS)]


def test_metrics_zero_delta():
    m = compute_pair_metrics(
        pair_id=0,
        final_action_FF=np.ones((AH, AD)),
        final_action_FE=np.ones((AH, AD)),
        final_action_EE=np.zeros((AH, AD)),
        final_action_EF=np.zeros((AH, AD)),
        action_hidden_FF=_seq(1.0),
        action_hidden_FE=_seq(1.0),
        v_t_FF=_x(1.0),
        v_t_FE=_x(1.0),
        x_t_FF=_x(1.0),
        x_t_FE=_x(1.0),
    )
    d = m.to_dict()
    assert d["final_action_rms_F"] == 0.0
    assert d["final_action_l2_F"] == 0.0
    assert d["final_action_rms_E"] == 0.0
    assert d["per_step_l2_F"] == [0.0] * AH
    assert d["per_action_dim_rms_F"] == [0.0] * AD
    assert d["action_hidden_rms_difference"] == [0.0] * STEPS
    assert d["v_t_rms_difference"] == [0.0] * STEPS
    assert d["x_t_rms_difference"] == [0.0] * STEPS


def test_metrics_constant_delta():
    m = compute_pair_metrics(
        pair_id=1,
        final_action_FF=np.ones((AH, AD)) * 2.0,
        final_action_FE=np.ones((AH, AD)),
        final_action_EE=np.zeros((AH, AD)),
        final_action_EF=np.ones((AH, AD)) * 0.5,
        action_hidden_FF=_seq(2.0),
        action_hidden_FE=_seq(1.0),
        v_t_FF=_x(2.0),
        v_t_FE=_x(1.0),
        x_t_FF=_x(2.0),
        x_t_FE=_x(1.0),
    )
    d = m.to_dict()
    # delta_F = ones -> rms = 1, l2 = sqrt(AH*AD).
    assert d["final_action_rms_F"] == 1.0
    assert d["final_action_l2_F"] == float(np.sqrt(AH * AD))
    # delta_E = -0.5 ones -> rms = 0.5.
    assert d["final_action_rms_E"] == 0.5
    assert d["per_step_l2_F"] == [float(np.sqrt(AD))] * AH
    assert d["per_action_dim_rms_F"] == [1.0] * AD
    # hidden diff = ones -> rms 1 per step.
    assert d["action_hidden_rms_difference"] == [1.0] * STEPS
    assert d["v_t_rms_difference"] == [1.0] * STEPS
    assert d["x_t_rms_difference"] == [1.0] * STEPS
