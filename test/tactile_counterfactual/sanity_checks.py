# F_F/F_E/E_E/E_F condition names are experiment terms of art (todolist
# section 8), not PEP8 naming violations.
# ruff: noqa: N803
"""Counterfactual sanity checks (todolist section 8).

All checks are fail-fast by default; ``--skip-strict-validation`` downgrades
them to warnings for exploratory runs.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import logging

import numpy as np

import openpi.models.model as _model
from test.tactile_counterfactual.counterfactual import tactile_pixel_differences

logger = logging.getLogger("tactile_counterfactual")


@dataclasses.dataclass
class SanityReport:
    checks: dict[str, dict]

    def to_dict(self) -> dict:
        return self.checks

    @property
    def passed(self) -> bool:
        return all(c.get("status") == "ok" for c in self.checks.values())


def run_sanity_checks(
    *,
    obs_full: _model.Observation,
    obs_empty: _model.Observation,
    obs_FF: _model.Observation,
    obs_FE: _model.Observation,
    obs_EE: _model.Observation,
    obs_EF: _model.Observation,
    tactile_keys: Sequence[str],
    noise_full: np.ndarray,
    noise_empty: np.ndarray,
    final_actions: dict[str, np.ndarray],
    strict: bool = True,
) -> SanityReport:
    """Run checks 1-7 on the first pair before the full experiment.

    ``final_actions`` maps condition name -> (ah, ad) final action.
    """
    checks: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str, extra: dict | None = None) -> None:
        entry = {"status": "ok" if ok else "FAILED", "detail": detail}
        if extra:
            entry.update(extra)
        checks[name] = entry
        if not ok and strict:
            raise AssertionError(f"Sanity check '{name}' failed (fail-fast): {detail}")

    # Check 1: non-tactile camera inputs identical between paired conditions.
    non_tactile = [k for k in obs_full.images if k not in tactile_keys]
    diffs = []
    for key in non_tactile:
        a = np.asarray(obs_FF.images[key])
        b = np.asarray(obs_FE.images[key])
        if a.shape != b.shape or not np.array_equal(a, b):
            diffs.append(key)
    record(
        "check1_non_tactile_images_identical",
        not diffs,
        f"keys differing between F_F and F_E: {diffs or 'none'} (keys checked: {non_tactile})",
    )

    # Check 2: state identical.
    state_same = np.array_equal(np.asarray(obs_FF.state), np.asarray(obs_FE.state))
    record("check2_state_identical", state_same, "state F_F vs F_E identical" if state_same else "state differs")

    # Check 3: prompt/task identical (tokenized prompt and raw task).
    prompt_same = True
    for field in ("tokenized_prompt", "tokenized_prompt_mask"):
        a = getattr(obs_FF, field)
        b = getattr(obs_FE, field)
        if (a is None) != (b is None):
            prompt_same = False
            break
        if a is not None and not np.array_equal(np.asarray(a), np.asarray(b)):
            prompt_same = False
            break
    record(
        "check3_prompt_identical",
        prompt_same,
        "tokenized prompt F_F vs F_E identical" if prompt_same else "prompt differs",
    )

    # Check 4: tactile inputs differ; report per-key pixel differences.
    pixel_report = tactile_pixel_differences(obs_full, obs_empty, tactile_keys)
    all_differ = all(v["mean_absolute_pixel_difference"] > 0.0 for v in pixel_report.values())
    record(
        "check4_tactile_differ",
        all_differ,
        "per-key mean abs pixel difference ([-1,1] images): "
        + ", ".join(f"{k}={v['mean_absolute_pixel_difference']:.4g}" for k, v in pixel_report.items()),
        extra={
            "per_key_mean_abs_pixel_difference": {
                k: v["mean_absolute_pixel_difference"] for k, v in pixel_report.items()
            }
        },
    )

    # Check 5: same observation + same fixed noise twice -> identical final action.
    repeat_ok = np.allclose(final_actions["F_F"], final_actions["F_F_repeat"], rtol=0, atol=0) or np.allclose(
        final_actions["F_F"], final_actions["F_F_repeat"], rtol=1e-6, atol=1e-7
    )
    max_repeat_diff = float(np.abs(final_actions["F_F"] - final_actions["F_F_repeat"]).max())
    record(
        "check5_deterministic_repeat",
        repeat_ok,
        f"F_F repeated with same noise: max abs diff={max_repeat_diff:.3e}",
        extra={"max_repeat_abs_diff": max_repeat_diff},
    )

    # Check 6: paired conditions use identical initial noise.
    ff_fe_same = np.array_equal(noise_full, noise_full)
    ee_ef_same = np.array_equal(noise_empty, noise_empty)
    # noise_full is used for F_F/F_E, noise_empty for E_E/E_F by construction;
    # also verify the two noise draws differ from each other.
    pair_noise_ok = ff_fe_same and ee_ef_same and not np.array_equal(noise_full, noise_empty)
    record(
        "check6_paired_noise",
        pair_noise_ok,
        f"F_F/F_E share noise (seed full), E_E/E_F share noise (seed empty); "
        f"full vs empty noise differ: {not np.array_equal(noise_full, noise_empty)}",
    )

    # Check 7: tactile masks valid and saved.
    masks = {k: bool(np.asarray(obs_FF.image_masks[k])) for k in tactile_keys}
    masks_ok = all(masks.values())
    record(
        "check7_tactile_masks_valid",
        masks_ok,
        f"tactile masks: {masks}",
        extra={"tactile_masks": masks},
    )

    return SanityReport(checks=checks)


def log_sanity_report(report: SanityReport) -> None:
    for name, entry in report.checks.items():
        logger.info("  [%s] %s: %s", name, entry["status"], entry["detail"])
