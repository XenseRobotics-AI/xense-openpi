"""Explicit training-order schema, independent of hardware SDK imports."""

import numpy as np

CAMERAS = ("head", "left_wrist", "right_wrist")
FINGER_NAMES = (
    "index_finger_mcp_flex",
    "index_finger_mcp_abd",
    "index_finger_pip",
    "index_finger_dip",
    "middle_finger_mcp_flex",
    "middle_finger_mcp_abd",
    "middle_finger_pip",
    "middle_finger_dip",
    "pinky_mcp_flex",
    "pinky_mcp_abd",
    "pinky_pip",
    "pinky_dip",
    "ring_finger_mcp_flex",
    "ring_finger_mcp_abd",
    "ring_finger_pip",
    "ring_finger_dip",
    "thumb_cmc_flex",
    "thumb_cmc_abd",
    "thumb_mcp",
    "thumb_ip",
)
ACTION_KEYS = tuple(
    f"{side}_tcp.{axis}" for side in ("left", "right") for axis in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")
) + tuple(f"{side}_{joint}.pos" for side in ("l", "r") for joint in FINGER_NAMES)


def state_from_observation(observation: dict) -> np.ndarray:
    state = np.asarray([observation[key] for key in ACTION_KEYS], dtype=np.float32)
    if state.shape != (58,) or not np.isfinite(state).all():
        raise ValueError("Robot state must contain 58 finite scalar values")
    return state


def validate_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 58 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite absolute actions with shape (horizon, 58), got {actions.shape}")
    for start in (3, 12):
        first, second = actions[:, start : start + 3], actions[:, start + 3 : start + 6]
        if np.any(np.linalg.norm(first, axis=-1) < 1e-6) or np.any(
            np.linalg.norm(np.cross(first, second), axis=-1) < 1e-6
        ):
            raise ValueError("Degenerate TCP rotation in action chunk")
    return actions


def action_to_dict(action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action)
    if action.shape != (58,):
        raise ValueError(f"Expected one 58D action, got {action.shape}")
    action = validate_actions(action[None])[0]
    # Neither delta conversion nor finger clipping belongs here. The server
    # returns absolute targets; the hand driver owns device joint ordering.
    return dict(zip(ACTION_KEYS, map(float, action), strict=True))
