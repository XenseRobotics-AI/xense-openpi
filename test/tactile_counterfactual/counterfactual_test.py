"""Tests for the tactile-only counterfactual swap."""

from __future__ import annotations

import numpy as np
import pytest

import openpi.models.model as _model
from test.tactile_counterfactual.counterfactual import assert_swap_clean
from test.tactile_counterfactual.counterfactual import make_counterfactual_observation
from test.tactile_counterfactual.counterfactual import tactile_pixel_differences

TACTILE_KEYS = ("tactile_0_rgb", "tactile_1_rgb", "tactile_2_rgb", "tactile_3_rgb")
RGB_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _make_obs(seed: int, tactile_value: float) -> _model.Observation:
    from openpi.shared import array_typing as at

    rng = np.random.default_rng(seed)
    images = {k: rng.uniform(-1, 1, (1, 32, 32, 3)).astype(np.float32) for k in RGB_KEYS}
    for k in TACTILE_KEYS:
        images[k] = np.full((1, 32, 32, 3), tactile_value, dtype=np.float32)
    masks = {k: np.ones((1,), dtype=bool) for k in (*RGB_KEYS, *TACTILE_KEYS)}
    with at.disable_typechecking():
        return _model.Observation(
            images=images,
            image_masks=masks,
            state=np.zeros((1, 20), dtype=np.float32),
            tokenized_prompt=np.arange(10, dtype=np.int32)[None, :],
            tokenized_prompt_mask=np.ones((1, 10), dtype=bool),
        )


def test_swap_only_changes_tactile_fields():
    base = _make_obs(seed=1, tactile_value=0.1)
    donor = _make_obs(seed=2, tactile_value=-0.7)

    swapped = make_counterfactual_observation(base, donor, TACTILE_KEYS)

    # Tactile fields come from the donor.
    for k in TACTILE_KEYS:
        assert np.array_equal(np.asarray(swapped.images[k]), np.asarray(donor.images[k]))
    # Non-tactile fields identical to base.
    for k in RGB_KEYS:
        assert np.array_equal(np.asarray(swapped.images[k]), np.asarray(base.images[k]))
    assert np.array_equal(np.asarray(swapped.state), np.asarray(base.state))
    assert np.array_equal(np.asarray(swapped.tokenized_prompt), np.asarray(base.tokenized_prompt))
    assert np.array_equal(np.asarray(swapped.tokenized_prompt_mask), np.asarray(base.tokenized_prompt_mask))


def test_swap_clean_assertions_pass():
    base = _make_obs(seed=1, tactile_value=0.1)
    donor = _make_obs(seed=2, tactile_value=-0.7)
    swapped = make_counterfactual_observation(base, donor, TACTILE_KEYS)
    assert_swap_clean(base, swapped, TACTILE_KEYS)  # must not raise


def test_swap_clean_assertions_catch_state_leak():
    base = _make_obs(seed=1, tactile_value=0.1)
    donor = _make_obs(seed=2, tactile_value=-0.7)
    swapped = make_counterfactual_observation(base, donor, TACTILE_KEYS)
    # Tamper with the state: the assertion must fail.
    import dataclasses

    swapped = dataclasses.replace(swapped, state=base.state + 1.0)
    with pytest.raises(AssertionError, match="state"):
        assert_swap_clean(base, swapped, TACTILE_KEYS)


def test_swap_clean_assertions_catch_rgb_leak():
    base = _make_obs(seed=1, tactile_value=0.1)
    donor = _make_obs(seed=2, tactile_value=-0.7)
    swapped = make_counterfactual_observation(base, donor, TACTILE_KEYS)
    import dataclasses

    images = dict(swapped.images)
    images["base_0_rgb"] = images["base_0_rgb"] + 1.0
    swapped = dataclasses.replace(swapped, images=images)
    with pytest.raises(AssertionError, match="base_0_rgb"):
        assert_swap_clean(base, swapped, TACTILE_KEYS)


def test_swap_missing_key_raises():
    base = _make_obs(seed=1, tactile_value=0.1)
    donor = _make_obs(seed=2, tactile_value=-0.7)
    with pytest.raises(ValueError, match="tactile_9_rgb"):
        make_counterfactual_observation(base, donor, ("tactile_0_rgb", "tactile_9_rgb"))


def test_pixel_differences_report():
    base = _make_obs(seed=1, tactile_value=0.1)
    donor = _make_obs(seed=2, tactile_value=-0.7)
    report = tactile_pixel_differences(base, donor, TACTILE_KEYS)
    assert set(report) == set(TACTILE_KEYS)
    for key in TACTILE_KEYS:
        assert report[key]["mean_absolute_pixel_difference"] == pytest_approx(0.8)
        assert report[key]["max_absolute_pixel_difference"] == pytest_approx(0.8)


def pytest_approx(x: float):
    import pytest

    return pytest.approx(x)
