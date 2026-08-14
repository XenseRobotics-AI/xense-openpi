"""Load a bench recipe into a ``BiFlexivRizon4RTConfig``.

Bench hardware (arm SNs, start/home poses, camera SNs, gripper backend) is no
longer carried by the lerobot config dataclass — the ``stations/`` layer that
``bi_mount_type`` used to index was removed upstream at ``3b964bc6``. Whoever
builds the config supplies the hardware, and recipes are how lerobot's own CLIs
supply it. This module does the same thing for the OpenPI client: read the
``robot:`` block out of a YAML recipe and decode it through draccus, so a typo
or a knob belonging to the other gripper backend fails the parse instead of
being silently ignored.

Run tuning stays on the OpenPI CLI and is applied on top of the decoded config:

    dataclass default  <  recipe YAML  <  --args.* flag

See ``recipes/README.md`` for what belongs on which side of that line.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import draccus

# Importing each config class is what registers it with the draccus choice
# registries the recipe's `type:` discriminators resolve against — the same
# unused-looking import block lerobot_record.py and lerobot_teleoperate.py carry.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.xense.configuration_xense import XenseTactileCameraConfig  # noqa: F401
from lerobot.robots import RobotConfig
from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import ROBOT_TYPE
from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
import yaml

RECIPES_DIR = Path(__file__).parent / "recipes"


def available_recipes() -> list[str]:
    """Recipe names shipped alongside this example, sorted."""
    return sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))


def resolve_recipe_path(recipe: str | Path) -> Path:
    """Turn a ``--args.robot-recipe`` value into a file path.

    A bare name (``forward-05``) resolves against this example's ``recipes/``
    directory. Anything with a separator or a ``.yaml`` suffix is taken as a
    path, so a recipe living in the lerobot-xense tree can be passed directly.
    """
    text = str(recipe)
    candidate = Path(text).expanduser()
    if "/" in text or candidate.suffix in (".yaml", ".yml"):
        if not candidate.is_file():
            raise FileNotFoundError(f"Robot recipe not found: {candidate}")
        return candidate

    path = RECIPES_DIR / f"{text}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Unknown robot recipe {text!r}. Available: {', '.join(available_recipes())}. "
            f"Pass a path instead to use a recipe from outside {RECIPES_DIR}."
        )
    return path


def load_robot_config(recipe: str | Path, **overrides: Any) -> BiFlexivRizon4RTConfig:
    """Decode a recipe's ``robot:`` block, then apply non-None CLI overrides.

    Args:
        recipe: Recipe name (resolved in ``recipes/``) or a path to a YAML file.
        **overrides: Config fields to force. ``None`` values are dropped, so a
            CLI flag left at its "unset" default leaves the recipe's value alone.

    Raises:
        FileNotFoundError: The recipe does not exist.
        ValueError: The file has no ``robot:`` block, or it describes a different
            robot type.
        draccus.utils.DecodingError: The block has an unknown or mistyped field.
    """
    path = resolve_recipe_path(recipe)
    raw = yaml.safe_load(path.read_text()) or {}

    block = raw.get("robot")
    if not isinstance(block, dict):
        raise ValueError(f"{path} has no `robot:` block — see recipes/README.md.")

    declared = block.get("type")
    if declared != ROBOT_TYPE:
        raise ValueError(f"{path} declares `type: {declared}`, but this example drives {ROBOT_TYPE!r}.")

    config = draccus.decode(RobotConfig, block)

    applied = {k: v for k, v in overrides.items() if v is not None}
    if applied:
        # replace() re-runs __post_init__, so the validators fire on the merged
        # values and the per-side gripper configs are rebuilt.
        config = dataclasses.replace(config, **applied)

    return config
