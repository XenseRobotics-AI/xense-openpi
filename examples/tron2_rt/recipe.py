"""Bench recipes for the TRON2 RT OpenPI example.

The recipe contains physical bench settings (network endpoints, startup pose,
Bridge camera and gripper).  Per-run policy/runtime settings stay in
``runs/`` and are applied by :mod:`examples.tron2_rt.main`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.robots.tron2_rt.config_tron2_rt import Tron2RTConfig

import examples.flexiv_recipe as _shared

RECIPES_DIR = Path(__file__).parent / "recipes"


def available_recipes() -> list[str]:
    """Return the names of recipes shipped with this example."""

    return _shared.available_recipes(RECIPES_DIR)


def resolve_recipe_path(recipe: str | Path) -> Path:
    """Resolve a recipe name or an explicit YAML path."""

    return _shared.resolve_recipe_path(RECIPES_DIR, recipe)


def load_robot_config(recipe: str | Path, **overrides: Any) -> Tron2RTConfig:
    """Decode a recipe and apply non-``None`` run-time overrides."""

    return _shared.load_robot_config(RECIPES_DIR, Tron2RTConfig, recipe, **overrides)
