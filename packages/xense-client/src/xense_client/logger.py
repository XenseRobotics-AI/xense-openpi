# Re-export get_logger from lerobot for unified logging
from lerobot.utils.robot_utils import get_logger
import spdlog

# spdlog loggers only have warn(), but stdlib-style warning() keeps coming back
# (ruff G010 "fixes" warn -> warning). Alias it so both spellings work.
if not hasattr(spdlog.Logger, "warning"):
    spdlog.Logger.warning = spdlog.Logger.warn

__all__ = ["get_logger"]
