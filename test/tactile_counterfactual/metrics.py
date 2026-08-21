"""Basic sensitivity metrics for one counterfactual pair.

For a pair ``i`` with the four conditions:

    delta_F = final_action(F_F) - final_action(F_E)
    delta_E = final_action(E_E) - final_action(E_F)

we report aggregate RMS / L2 of the final-action difference plus per-step and
per-action-dim breakdowns, and the per-denoising-step RMS differences of
action_hidden / v_t / x_t between the paired conditions.
"""

# F_F/F_E/E_E/E_F condition and delta_F/delta_E names are experiment terms of
# art (see todolist section 14), not PEP8 naming violations.
# ruff: noqa: N803, N806, N815

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import numpy as np


@dataclasses.dataclass
class PairMetrics:
    pair_id: int

    # final-action deltas (ah, ad)
    delta_F: np.ndarray
    delta_E: np.ndarray

    # Per-step (action_horizon,) L2 norms of the deltas.
    per_step_l2_F: np.ndarray
    per_step_l2_E: np.ndarray

    # Per-action-dim (ad,) RMS of the deltas.
    per_action_dim_rms_F: np.ndarray
    per_action_dim_rms_E: np.ndarray

    # Per denoising step RMS of the hidden/v/x differences (num_steps,).
    action_hidden_rms_difference: np.ndarray
    v_t_rms_difference: np.ndarray
    x_t_rms_difference: np.ndarray

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "final_action_rms_F": float(_rms(self.delta_F)),
            "final_action_l2_F": float(_l2(self.delta_F)),
            "final_action_rms_E": float(_rms(self.delta_E)),
            "final_action_l2_E": float(_l2(self.delta_E)),
            "per_step_l2_F": _as_list(self.per_step_l2_F),
            "per_step_l2_E": _as_list(self.per_step_l2_E),
            "per_action_dim_rms_F": _as_list(self.per_action_dim_rms_F),
            "per_action_dim_rms_E": _as_list(self.per_action_dim_rms_E),
            "action_hidden_rms_difference": _as_list(self.action_hidden_rms_difference),
            "v_t_rms_difference": _as_list(self.v_t_rms_difference),
            "x_t_rms_difference": _as_list(self.x_t_rms_difference),
        }


def compute_pair_metrics(
    pair_id: int,
    final_action_FF: np.ndarray,
    final_action_FE: np.ndarray,
    final_action_EE: np.ndarray,
    final_action_EF: np.ndarray,
    action_hidden_FF: Sequence[np.ndarray],
    action_hidden_FE: Sequence[np.ndarray],
    v_t_FF: Sequence[np.ndarray],
    v_t_FE: Sequence[np.ndarray],
    x_t_FF: Sequence[np.ndarray],
    x_t_FE: Sequence[np.ndarray],
) -> PairMetrics:
    """Compute pair metrics from the four traced conditions.

    All arrays are (ah, ad) for final actions and (ah, emb) / (ah, ad) for the
    per-step sequences (already squeezed to batch=1).
    """
    delta_F = np.asarray(final_action_FF) - np.asarray(final_action_FE)
    delta_E = np.asarray(final_action_EE) - np.asarray(final_action_EF)

    per_step_l2_F = np.linalg.norm(delta_F, axis=-1)
    per_step_l2_E = np.linalg.norm(delta_E, axis=-1)
    per_action_dim_rms_F = np.sqrt(np.mean(delta_F**2, axis=0))
    per_action_dim_rms_E = np.sqrt(np.mean(delta_E**2, axis=0))

    def _per_step_rms(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([_rms(np.asarray(x) - np.asarray(y)) for x, y in zip(a, b)])

    return PairMetrics(
        pair_id=pair_id,
        delta_F=delta_F,
        delta_E=delta_E,
        per_step_l2_F=per_step_l2_F,
        per_step_l2_E=per_step_l2_E,
        per_action_dim_rms_F=per_action_dim_rms_F,
        per_action_dim_rms_E=per_action_dim_rms_E,
        action_hidden_rms_difference=_per_step_rms(action_hidden_FF, action_hidden_FE),
        v_t_rms_difference=_per_step_rms(v_t_FF, v_t_FE),
        x_t_rms_difference=_per_step_rms(x_t_FF, x_t_FE),
    )


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def _l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x)))


def _as_list(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(x).reshape(-1)]
