"""Decode a lerobot bench recipe into a robot config.

Shared by the Flexiv examples; each binds its own ``recipes/`` directory and
config class in a thin ``recipe.py`` beside it.

Bench hardware (arm SNs, start/home poses, camera SNs, gripper backend) is no
longer carried by the lerobot config dataclasses. The ``stations/`` layer that
``bi_mount_type`` indexed was folded into recipes upstream at ``3b964bc6``, and
the flat ``gripper_*`` knobs became one typed ``gripper:`` block at the same
commit. Whoever builds the config supplies the hardware, and recipes are how
lerobot's own CLIs supply it — so this reads a recipe's ``robot:`` block and
decodes it through draccus, the same path ``lerobot_record.py`` takes. A typo,
or a knob belonging to the other gripper backend, fails the parse instead of
being silently ignored.

Run tuning stays on the OpenPI CLI, which owns those knobs outright: every
tuning flag has a concrete default, so it is always applied on top of the
decoded config. A recipe written for ``lerobot-teleoperate`` may carry tuning
keys of its own (``use_force:``, ``enable_tactile_sensors:``, ``log_level:``);
those lose to the flag, and ``load_robot_config`` logs each one it overrides so
the swap is visible rather than silent.
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
from lerobot.utils.robot_utils import get_logger
import yaml

import examples.run_config as _run_config

logger = get_logger("FlexivRecipe")


def available_recipes(recipes_dir: Path) -> list[str]:
    """Recipe names in ``recipes_dir``, sorted."""
    return _run_config.available(recipes_dir)


def resolve_recipe_path(recipes_dir: Path, recipe: str | Path) -> Path:
    """Turn a ``--args.robot-recipe`` value into a file path.

    A bare name (``forward-05``) resolves against ``recipes_dir``. Anything with
    a separator or a YAML suffix is taken as a path, so a recipe living in the
    lerobot-xense tree can be passed directly.
    """
    return _run_config.resolve_path(recipes_dir, recipe, kind="robot recipe")


def load_robot_config(recipes_dir: Path, config_cls: type, recipe: str | Path, **overrides: Any):
    """Decode a recipe's ``robot:`` block, then apply non-None CLI overrides.

    Args:
        recipes_dir: Where a bare recipe name resolves.
        config_cls: The robot config this example drives. Its registered choice
            name is what the recipe's ``type:`` must declare.
        recipe: Recipe name or path to a YAML file.
        **overrides: Config fields to force. ``None`` values are dropped, so a
            CLI flag left at its "unset" default leaves the recipe's value alone.

    Raises:
        FileNotFoundError: The recipe does not exist.
        ValueError: The file is not a mapping, has no ``robot:`` block, or
            describes a different robot type.
        draccus.utils.DecodingError: The block has an unknown or mistyped field.
    """
    robot_type = RobotConfig.get_choice_name(config_cls)
    path = resolve_recipe_path(recipes_dir, recipe)
    raw = yaml.safe_load(path.read_text())

    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a YAML mapping — see recipes/README.md.")

    block = raw.get("robot")
    if not isinstance(block, dict):
        raise ValueError(f"{path} has no `robot:` block — see recipes/README.md.")

    declared = block.get("type")
    if declared != robot_type:
        raise ValueError(f"{path} declares `type: {declared}`, but this example drives {robot_type!r}.")

    config = draccus.decode(RobotConfig, block)

    applied = {k: v for k, v in overrides.items() if v is not None}
    if applied:
        # A recipe written for lerobot-teleoperate may set tuning keys the CLI
        # owns here. The flag wins, but say so — a silently ignored line in a
        # recipe is exactly the failure the typed block exists to prevent.
        shadowed = {k: block[k] for k in applied if k in block and block[k] != applied[k]}
        if shadowed:
            swaps = ", ".join(f"{k}: {v!r} -> {applied[k]!r}" for k, v in sorted(shadowed.items()))
            logger.warn(f"CLI flags override {path.name}: {swaps}")

        # replace() re-runs __post_init__, so the validators fire on the merged
        # values and any per-side gripper configs are rebuilt.
        config = dataclasses.replace(config, **applied)

    return config
