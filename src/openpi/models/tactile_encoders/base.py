"""Base interface for tactile-image encoders.

A tactile encoder maps a (B, H, W, 3) image tensor into a (B, feature_dim) feature vector.
Implementations live alongside this file (e.g. `fastvit.py`) and are registered in
`__init__.py` so that the model can be swapped via a simple string name in the
training config without touching `Pi0TactileFastVit` or `pi0.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class TactileEncoder(Protocol):
    """Protocol describing the contract every tactile encoder must satisfy.

    Concrete implementations are usually `nnx.Module` subclasses, but at type-check
    time we only require the two attributes below.

    Attributes:
        feature_dim: Number of channels in the per-image embedding produced by ``__call__``.

    Methods:
        __call__: ``(B, H, W, 3)`` (float in [-1, 1]) -> ``(B, feature_dim)``.
    """

    feature_dim: int

    def __call__(self, images: jax.Array) -> jax.Array: ...
