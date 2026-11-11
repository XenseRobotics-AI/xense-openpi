"""Load nested Rebot arm configs from an inference, record, or teleop recipe."""

import dataclasses
from pathlib import Path

import draccus
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.xense.configuration_xense import XenseTactileCameraConfig  # noqa: F401
from lerobot.robots.bi_rebot_b601_rt.config_bi_rebot_b601_rt import BiRebotB601RTConfig
from lerobot.robots.config import RobotConfig
from lerobot.utils.robot_utils import get_logger
import yaml

import examples.run_config as run_config

RECIPES_DIR = Path(__file__).parent / "recipes"
logger = get_logger("BiRebotRecipe")


def resolve_recipe_path(recipe: str | Path) -> Path:
    return run_config.resolve_path(RECIPES_DIR, recipe, kind="robot recipe")


def validate_robot_config(config: BiRebotB601RTConfig, *, go_to_start: bool = True) -> None:
    """Reject incompatible state/action layouts before connecting to hardware."""
    for side in ("left", "right"):
        arm = getattr(config, f"{side}_arm_config")
        if arm.use_joint_observation:
            raise ValueError(f"{side}: use_joint_observation must be false for the 20D Cartesian policy")
        if arm.gripper is None and not arm.has_gripper:
            raise ValueError(f"{side}: a gripper is required for the 20D policy")
        if go_to_start and arm.reset_position_rad is None:
            raise ValueError(f"{side}: reset_position_rad is required when go_to_start is true")


def load_robot_config(
    recipe: str | Path,
    *,
    enable_tactile: bool = False,
    max_lin_speed: float | None = None,
    max_ang_speed: float | None = None,
    reset_on_exit: bool | None = None,
) -> BiRebotB601RTConfig:
    path = resolve_recipe_path(recipe).resolve()
    raw = yaml.safe_load(path.read_text())
    block = raw.get("robot") if isinstance(raw, dict) else None
    if not isinstance(block, dict) or block.get("type") != "bi_rebot_b601_rt":
        raise ValueError(f"{path} must contain a robot block with type: bi_rebot_b601_rt")

    # Upstream record recipes refer to third_party/... from the lerobot repo
    # root. Resolve against recipe ancestors so launching from openpi works too.
    for side in ("left", "right"):
        arm = block.get(f"{side}_arm_config", {})
        if arm.get("urdf_path"):
            urdf = Path(arm["urdf_path"]).expanduser()
            candidates = [urdf] if urdf.is_absolute() else [parent / urdf for parent in path.parents]
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f"{side} URDF {urdf} not found relative to {path} or its ancestors")
            arm["urdf_path"] = str(resolved.resolve())

    config = draccus.decode(RobotConfig, block)
    if not isinstance(config, BiRebotB601RTConfig):
        raise TypeError(f"Expected BiRebotB601RTConfig, got {type(config).__name__}")
    updated = {}
    for side in ("left", "right"):
        arm = getattr(config, f"{side}_arm_config")
        overrides = {}
        for name, value in (
            ("max_cartesian_linear_speed", max_lin_speed),
            ("max_cartesian_angular_speed", max_ang_speed),
        ):
            if value is not None:
                logger.info(f"{side}.{name}: {getattr(arm, name)} -> {value}")
                overrides[name] = value
        if arm.gripper is not None:
            logger.info(f"{side}.gripper.enable_tactile: {arm.gripper.enable_tactile} -> {enable_tactile}")
            overrides["gripper"] = dataclasses.replace(arm.gripper, enable_tactile=enable_tactile)
        updated[f"{side}_arm_config"] = dataclasses.replace(arm, **overrides)
    if reset_on_exit is not None:
        updated["reset_on_exit"] = reset_on_exit
    return dataclasses.replace(config, **updated)
