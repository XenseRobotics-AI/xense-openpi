"""Pluggable keypoint detector for the dewu video-switch demo.

Contract
--------
A detector receives one forwarded frame and returns a *scene id* (a short
string) or None if it cannot decide this frame. The scene id is what drives the
front-end video switch (see web/index.html SCENES).

frame = {
    "step":   int,
    "state":  np.ndarray shape (20,)            # robot TCP state, optional
    "images": {"head": np.ndarray (H, W, 3) uint8}  # raw head camera, optional
    "action": np.ndarray (20,)                  # model action, optional
}

The 20-D state is the bi-arm proprioception (see the dataset feature names):
    [0:9]  left_tcp  (x, y, z, r1..r6)
    [9:18] right_tcp (x, y, z, r1..r6)
    [18]   left_gripper.pos
    [19]   right_gripper.pos

Two detectors ship here:
  - GripperSceneDetector: the *real* default. Switches scene on the robot's
    grasp state (gripper open/closed), the single most visually-tied event in a
    pick-and-place task. Pure state math, no ML deps — keeps the playback PC
    light (websockets + msgpack + numpy only).
  - StubDetector: a brightness placeholder kept only for the synthetic link test
    (sim_laptop). Not for real footage.

A heavier product detector (e.g. a keypoint model over the head image) plugs in
the same way: subclass Detector, implement detect(), register in make_detector().
"""

from __future__ import annotations

import abc

import numpy as np


class Detector(abc.ABC):
    """Maps a forwarded frame to a scene id (or None = undecided)."""

    @abc.abstractmethod
    def detect(self, frame: dict) -> str | None: ...


# State layout constants (see module docstring / dataset feature names).
LEFT_GRIPPER_IDX = 18
RIGHT_GRIPPER_IDX = 19


class GripperSceneDetector(Detector):
    """Real detector: switch scene on the robot's grasp state.

    The clearest, most visually-tied event in a pick-and-place task is the
    gripper opening/closing. We read one gripper position from the 20-D state and
    emit ``scene_low`` while it sits in its lower plateau and ``scene_high`` while
    in its upper plateau — so the screen flips exactly when the robot grabs /
    releases.

    The absolute open/closed values differ per robot and per dataset, so the
    split self-calibrates: we track the running min/max of the gripper signal and
    threshold at the midpoint with a hysteresis band (no chatter at the
    boundary). SceneController adds temporal debounce on top.

    Which physical state ("grabbing" vs "empty") should show which clip is a
    product choice — flip ``scene_low`` / ``scene_high`` (or pick the other
    gripper via ``grip_index``) to change it. The offline harness
    (replay_offline.py) prints which plateau each scene corresponds to.
    """

    def __init__(
        self,
        *,
        grip_index: int = RIGHT_GRIPPER_IDX,
        scene_low: str = "scene_a",
        scene_high: str = "scene_b",
        hysteresis: float = 0.1,
        min_range: float = 0.1,
    ) -> None:
        self._grip_index = int(grip_index)
        self._scene_low = scene_low
        self._scene_high = scene_high
        self._hysteresis = float(hysteresis)  # half-band as a fraction of range
        self._min_range = float(min_range)

        self._lo: float | None = None
        self._hi: float | None = None
        self._plateau: str | None = None  # "low" / "high", last decided side

    def detect(self, frame: dict) -> str | None:
        state = frame.get("state")
        if not (isinstance(state, np.ndarray) and state.shape[-1] > self._grip_index):
            return None
        pos = float(state[..., self._grip_index])

        # Self-calibrate the open/closed range from what we've observed so far.
        self._lo = pos if self._lo is None else min(self._lo, pos)
        self._hi = pos if self._hi is None else max(self._hi, pos)
        rng = self._hi - self._lo
        if rng < self._min_range:
            return None  # not enough spread yet to tell open from closed apart

        mid = 0.5 * (self._lo + self._hi)
        band = self._hysteresis * rng
        if pos >= mid + band:
            self._plateau = "high"
        elif pos <= mid - band:
            self._plateau = "low"
        # inside the band: hold the previous plateau (hysteresis)

        if self._plateau is None:
            return None
        return self._scene_high if self._plateau == "high" else self._scene_low


class StubDetector(Detector):
    """Placeholder: derive a scene from image brightness, no ML deps.

    Kept only for the synthetic link test (sim_laptop ramps brightness): it
    proves the transport + switching loop without any real robot data. On real
    footage it is meaningless — use GripperSceneDetector. The scene ids it emits
    match web/index.html's SCENES keys.
    """

    def __init__(self, scene_ids: tuple[str, ...] = ("scene_a", "scene_b")) -> None:
        if not scene_ids:
            raise ValueError("scene_ids must be non-empty")
        self._scenes = scene_ids

    def detect(self, frame: dict) -> str | None:
        imgs = frame.get("images") or {}
        head = imgs.get("head")
        if isinstance(head, np.ndarray) and head.size > 0:
            brightness = float(head.mean()) / 255.0  # 0..1
            idx = min(int(brightness * len(self._scenes)), len(self._scenes) - 1)
            return self._scenes[idx]

        state = frame.get("state")
        if isinstance(state, np.ndarray) and state.shape[-1] >= 2:
            grip = float(state[..., -1])  # right gripper as a crude proxy
            idx = min(int(abs(grip) * len(self._scenes)), len(self._scenes) - 1)
            return self._scenes[idx]

        return None


def make_detector(name: str = "gripper", *, config_path: str | None = None, **kwargs) -> Detector:
    """Factory so app.py can select a detector by name / config.

    config_path is only used by detectors that take a JSON config (shoe_sm); it is
    not forwarded to the simple detectors.
    """
    if name == "gripper":
        return GripperSceneDetector(**kwargs)
    if name == "stub":
        return StubDetector(**kwargs)
    if name == "shoe_sm":
        # Per-shoe state machine (pose+gripper pick event + OpenCV blue insole).
        # Lazy import: pulls in pose_events/vision (cv2) only when selected.
        try:
            from examples.dewu_video_switch.shoe_state_machine import ShoeStateMachineDetector
        except ImportError:  # standalone copy on the video-playback laptop
            from shoe_state_machine import ShoeStateMachineDetector  # type: ignore
        return ShoeStateMachineDetector(config_path=config_path, **kwargs)
    raise ValueError(f"unknown detector: {name!r}")
