"""Load a bench recipe into a ``FlexivRizon4RTConfig``.

The single-arm config lost its flat gripper knobs (``gripper_type``,
``gripper_mac_addr``, ``gripper_cam_size``, ``gripper_rectify_size``,
``gripper_max_pos``) when lerobot promoted grippers to a typed device family at
``3b964bc6``: there are now two backends (``serial``, ``taccap_follower``), both
configured through a single ``gripper:`` block that draccus decodes, so a knob
belonging to the other backend fails the parse instead of being ignored.

Neither backend supplies a camera. The wrist camera that used to arrive with the
Flare gripper is now an ordinary entry in ``cameras:`` — and unlike the bimanual
driver, this one does no USB-hub auto-discovery, so every camera the policy needs
must be pinned in the recipe.

Run tuning stays on the OpenPI CLI and is applied on top of the decoded config:

    dataclass default  <  recipe YAML  <  --args.* flag
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import draccus

# Importing each config class registers it with the draccus choice registries
# the recipe's `type:` discriminators resolve against.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.xense.configuration_xense import XenseTactileCameraConfig  # noqa: F401
from lerobot.robots import RobotConfig
from lerobot.robots.flexiv_rizon4_rt.config_flexiv_rizon4_rt import FlexivRizon4RTConfig
import yaml

RECIPES_DIR = Path(__file__).parent / "recipes"
ROBOT_TYPE = "flexiv_rizon4_rt"


def available_recipes() -> list[str]:
    """Recipe names shipped alongside this example, sorted."""
    return sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))


def resolve_recipe_path(recipe: str | Path) -> Path:
    """Turn a ``--args.robot-recipe`` value into a file path.

    A bare name resolves against this example's ``recipes/`` directory; anything
    with a separator or a ``.yaml`` suffix is taken as a path.
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


def load_robot_config(recipe: str | Path, **overrides: Any) -> FlexivRizon4RTConfig:
    """Decode a recipe's ``robot:`` block, then apply non-None CLI overrides."""
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
        # replace() re-runs __post_init__ so validators fire on the merged values.
        config = dataclasses.replace(config, **applied)

    return config
