"""Chunk value checks only: no temporal/velocity gates or target rewriting."""

from importlib.util import find_spec
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from examples.bi_tianji_wuji.schema import ACTION_KEYS
from examples.bi_tianji_wuji.schema import validate_actions


def load_finger_limits() -> dict[str, tuple[float, float]]:
    """Read position bounds from the installed Hand 2 description, by name."""
    spec = find_spec("wuji_retargeting")
    if spec is None or spec.origin is None:
        raise RuntimeError("Chunk checks require wuji_retargeting with the Hand 2 URDF files installed")
    directory = Path(spec.origin).parent / "wuji-description/hand2/body/urdf"
    limits = {}
    for side in ("left", "right"):
        path = directory / f"{side}.urdf"
        for joint in ET.parse(path).getroot().findall("joint"):
            if joint.get("type") != "revolute":
                continue
            name = joint.attrib["name"]
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"{path}: missing position limits for {name}")
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            if name in limits or not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(f"{path}: duplicate joint or invalid position limits for {name}")
            limits[name] = (lower, upper)
    expected = {key.removesuffix(".pos") for key in ACTION_KEYS[18:]}
    if set(limits) != expected:
        raise ValueError(
            f"Hand 2 URDF joint names mismatch: missing={expected - set(limits)}, extra={set(limits) - expected}"
        )
    return limits


class ChunkLimits:
    def __init__(self):
        limits = load_finger_limits()
        self.finger_bounds = np.asarray([limits[key.removesuffix(".pos")][:2] for key in ACTION_KEYS[18:]])

    def validate(self, actions):
        actions = validate_actions(actions)
        lower, upper = self.finger_bounds.T
        invalid = (actions[:, 18:] < lower - 1e-6) | (actions[:, 18:] > upper + 1e-6)
        if invalid.any():
            step, joint = np.argwhere(invalid)[0]
            raise ValueError(
                f"Chunk rejected: step={step}, {ACTION_KEYS[18 + joint]}={actions[step, 18 + joint]:.6g} rad, "
                f"allowed=[{lower[joint]:.6g}, {upper[joint]:.6g}] rad"
            )
        return actions
