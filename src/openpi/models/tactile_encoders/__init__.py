"""Registry for tactile-image encoders.

To add a new encoder, drop a new module next to ``fastvit.py`` that exposes a
class satisfying :class:`TactileEncoder`, and register it in
:func:`build_tactile_encoder`. Downstream code (``Pi0TactileFastVit``) selects
the encoder by name via the training config — it does NOT import a concrete
encoder, so swapping encoders does not require touching the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flax.nnx as nnx

from openpi.models.tactile_encoders.base import TactileEncoder


def build_tactile_encoder(
    name: str,
    *,
    rngs: nnx.Rngs,
    pretrained_path: str | Path | None = None,
    **kwargs: Any,
) -> TactileEncoder:
    """Construct a tactile encoder by registered name.

    Args:
        name: Registered encoder name. Currently supported: ``"fastvit_t12"``.
        rngs: NNX rngs used to lazy-init the encoder.
        pretrained_path: Optional path to a Flax-format weight file (.safetensors).
        **kwargs: Encoder-specific overrides forwarded to the constructor.

    Returns:
        A ``TactileEncoder`` instance exposing ``feature_dim`` and
        ``__call__((B, H, W, 3)) -> (B, feature_dim)``.
    """
    if name == "fastvit_t12":
        from openpi.models.tactile_encoders.fastvit import TactileFastVitEncoder

        return TactileFastVitEncoder(
            rngs=rngs,
            pretrained_path=pretrained_path,
            **kwargs,
        )
    raise ValueError(f"Unknown tactile encoder: {name!r}")


__all__ = ["TactileEncoder", "build_tactile_encoder"]
