"""Counterfactual observation construction: tactile-only swap.

``make_counterfactual_observation`` takes a base observation (already run
through the training data pipeline) and a tactile donor observation, and
returns a new observation in which ONLY the configured tactile image keys
(and their masks) come from the donor. Everything else — non-tactile RGB,
state, tokenized prompt — is guaranteed bit-identical to the base.

The swap happens at the model-input level (after the training transforms), so
the non-tactile fields are literally the same tensors. The donor tactile
images have been through the same per-image preprocessing as in training, so
they re-enter the model through the identical tactile preprocessing chain
(resize is deterministic at eval time) and then FastViT -> tactile_proj.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import numpy as np

import openpi.models.model as _model


def make_counterfactual_observation(
    base_observation: _model.Observation,
    tactile_donor_observation: _model.Observation,
    tactile_keys: Sequence[str],
) -> _model.Observation:
    """Return ``base_observation`` with the tactile images/masks from the donor.

    Args:
        base_observation: The observation that provides every non-tactile field.
        tactile_donor_observation: The observation that provides the tactile
            images and masks.
        tactile_keys: The exact set of tactile image keys to swap (e.g. the
            model config's ``tactile_image_keys``).

    Raises:
        ValueError: If a tactile key is missing from either observation, or the
            base/donor tactile images are not elementwise different (a swap
            that changes nothing would silently invalidate the experiment).
    """
    missing = [k for k in tactile_keys if k not in base_observation.images or k not in tactile_donor_observation.images]
    if missing:
        raise ValueError(
            f"Tactile keys {missing} missing from base/donor observations "
            f"(base keys={sorted(base_observation.images)}, donor keys={sorted(tactile_donor_observation.images)})"
        )

    swapped_images = dict(base_observation.images)
    swapped_masks = dict(base_observation.image_masks)
    for key in tactile_keys:
        swapped_images[key] = tactile_donor_observation.images[key]
        if key in tactile_donor_observation.image_masks:
            swapped_masks[key] = tactile_donor_observation.image_masks[key]

    return dataclasses.replace(
        base_observation,
        images=swapped_images,
        image_masks=swapped_masks,
    )


def assert_swap_clean(
    base_observation: _model.Observation,
    swapped: _model.Observation,
    tactile_keys: Sequence[str],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Debug assertions that the swap touched exactly the tactile fields.

    Compares, elementwise, every field of the swapped observation against the
    base. Non-tactile fields must be identical (default tolerance 0 — they are
    the same tensors by construction); tactile fields may differ (they come
    from the donor).
    """
    tactile_set = set(tactile_keys)

    # Images.
    if set(swapped.images) != set(base_observation.images):
        raise AssertionError(
            f"image key set changed: base={sorted(base_observation.images)} swapped={sorted(swapped.images)}"
        )
    for key in base_observation.images:
        a = np.asarray(base_observation.images[key])
        b = np.asarray(swapped.images[key])
        if key in tactile_set:
            if a.shape != b.shape:
                raise AssertionError(f"tactile image {key} shape changed: {a.shape} vs {b.shape}")
            continue  # may differ — that is the point of the swap
        if a.shape != b.shape:
            raise AssertionError(f"non-tactile image {key} shape changed: {a.shape} vs {b.shape}")
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(f"non-tactile image {key} changed by the swap (max abs diff={np.abs(a - b).max()})")

    # Image masks.
    for key in base_observation.image_masks:
        a = np.asarray(base_observation.image_masks[key])
        b = np.asarray(swapped.image_masks.get(key))
        if b is None:
            raise AssertionError(f"image mask {key} disappeared in swapped observation")
        if key not in tactile_set and not np.array_equal(a, b):
            raise AssertionError(f"non-tactile image mask {key} changed by the swap")

    # State.
    if not np.allclose(np.asarray(base_observation.state), np.asarray(swapped.state), rtol=rtol, atol=atol):
        raise AssertionError("state changed by the swap")

    # Tokenized prompt.
    for field in ("tokenized_prompt", "tokenized_prompt_mask"):
        a = getattr(base_observation, field)
        b = getattr(swapped, field)
        if (a is None) != (b is None):
            raise AssertionError(f"{field} presence changed by the swap")
        if a is not None and not np.array_equal(np.asarray(a), np.asarray(b)):
            raise AssertionError(f"{field} changed by the swap")


def tactile_pixel_differences(
    obs_a: _model.Observation,
    obs_b: _model.Observation,
    tactile_keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-key mean absolute pixel difference (and max) between two observations.

    Values are computed on the model-input images (already in [-1, 1] float).
    """
    report: dict[str, dict[str, float]] = {}
    for key in tactile_keys:
        a = np.asarray(obs_a.images[key]).astype(np.float64)
        b = np.asarray(obs_b.images[key]).astype(np.float64)
        diff = np.abs(a - b)
        report[key] = {
            "mean_absolute_pixel_difference": float(diff.mean()),
            "max_absolute_pixel_difference": float(diff.max()),
            "shape": list(diff.shape),
        }
    return report
