"""Decode Tianji recipes without importing Flexiv or teleoperator modules."""

# Robot configuration imports require the hardware fork.

from pathlib import Path

import yaml

from examples import run_config
from examples.bi_tianji_wuji.schema import CAMERAS

RECIPES_DIR = Path(__file__).parent / "recipes"


def load_robot_config(recipe: str):
    import draccus
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.tianji_arm_wuji import TianjiArmWujiConfig

    path = run_config.resolve_path(RECIPES_DIR, recipe, kind="robot recipe")
    raw = yaml.safe_load(path.read_text())
    block = raw.get("robot") if isinstance(raw, dict) else None
    if not isinstance(block, dict) or block.get("type") != "tianji_arm_wuji":
        raise ValueError(f"{path} must contain robot.type: tianji_arm_wuji")
    block = dict(block)
    if block.get("wuji") is not None:
        raise ValueError("Use flat wuji_* fields, not a nested wuji block")
    # Lifecycle belongs to the deployment script. Never auto-home on connect
    # or open the hands on teardown (including partial connection failure).
    block.update(go_home_on_connect=False, wuji_return_to_zero_on_disconnect=False)
    config = draccus.decode(RobotConfig, block)
    if not isinstance(config, TianjiArmWujiConfig):
        raise TypeError("Expected TianjiArmWujiConfig")
    if not config.connect_wuji or config.wuji_hand_type != "both":
        raise ValueError("The 58D policy requires both Wuji hands connected")
    if set(config.cameras) != set(CAMERAS):
        raise ValueError(f"Configure exactly the three RGB cameras: {CAMERAS}")
    for camera in config.cameras.values():
        if hasattr(camera, "use_depth"):
            camera.use_depth = False
        color_mode = getattr(camera, "color_mode", "rgb")
        if getattr(color_mode, "value", color_mode) != "rgb":
            raise ValueError("Policy cameras must use RGB color mode")
    return config
