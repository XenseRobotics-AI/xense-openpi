"""Bench recipes for the bimanual Elite CS66 RT example.

Binds the shared loader (``examples/flexiv_recipe.py``) to this example's
``recipes/`` directory and config class. See ``recipes/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig

import examples.flexiv_recipe as _shared

RECIPES_DIR = Path(__file__).parent / "recipes"


def available_recipes() -> list[str]:
    """Recipe names shipped alongside this example, sorted."""
    return _shared.available_recipes(RECIPES_DIR)


def resolve_recipe_path(recipe: str | Path) -> Path:
    """Resolve a recipe name (under ``recipes/``) or path to a file."""
    return _shared.resolve_recipe_path(RECIPES_DIR, recipe)


def load_robot_config(recipe: str | Path, **overrides: Any) -> BiEliteCS66RTConfig:
    """Decode a recipe's ``robot:`` block, then apply non-None CLI overrides."""
    return _shared.load_robot_config(RECIPES_DIR, BiEliteCS66RTConfig, recipe, **overrides)
