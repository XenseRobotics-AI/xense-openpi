"""State-only (proprioception) sub-detectors for the shoe state machine.

These operate on the 20-D robot state vector forwarded from the robot laptop:
    [0:9]  left_tcp  (x, y, z, r1..r6)
    [9:18] right_tcp (x, y, z, r1..r6)
    [18]   left_gripper.pos     (continuous; nominally closed≈low, open≈high)
    [19]   right_gripper.pos
TCP xyz are in meters in the robot base frame. The gripper signal is continuous
(NOT 0/1), so the open/closed split self-calibrates (running min/max + hysteresis,
same idea as detector.GripperSceneDetector).

No heavy deps — numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT_TCP_XYZ = (0, 1, 2)
RIGHT_TCP_XYZ = (9, 10, 11)
LEFT_GRIPPER_IDX = 18
RIGHT_GRIPPER_IDX = 19


@dataclass
class BoundingBox3D:
    """Axis-aligned box in robot-base meters (e.g. the shoebox pick region)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, xyz) -> bool:
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max and self.z_min <= z <= self.z_max


class GraspCalibrator:
    """Self-calibrating open/closed decision for a continuous gripper signal.

    Tracks running min/max and thresholds at the midpoint with a hysteresis band.
    "closed" is the lower plateau when grasp_is_low (the common convention here),
    else the upper plateau. Returns None until enough spread is observed.
    """

    def __init__(
        self, grasp_is_low: bool = True, threshold: float | None = None, hysteresis: float = 0.1, min_range: float = 0.1
    ) -> None:
        self._grasp_is_low = grasp_is_low
        self._fixed = threshold
        self._hyst = hysteresis
        self._min_range = min_range
        self._lo: float | None = None
        self._hi: float | None = None
        self._closed = False

    def update(self, pos: float) -> bool | None:
        if self._fixed is not None:
            self._closed = (pos <= self._fixed) if self._grasp_is_low else (pos >= self._fixed)
            return self._closed
        self._lo = pos if self._lo is None else min(self._lo, pos)
        self._hi = pos if self._hi is None else max(self._hi, pos)
        rng = self._hi - self._lo
        if rng < self._min_range:
            return None
        mid = 0.5 * (self._lo + self._hi)
        band = self._hyst * rng
        if pos <= mid - band:
            self._closed = self._grasp_is_low
        elif pos >= mid + band:
            self._closed = not self._grasp_is_low
        return self._closed


class BoxGraspEdgeEvent:
    """One pick: ENTER box -> GRASP inside -> EXIT box while still grasping.

    Rising/falling-edge latch over a TCP position and a gripper:
        idle    -> inside : TCP enters the box (debounced confirm_inside frames)
        inside  -> grasped: gripper closes while inside (debounced confirm_grasp)
        grasped -> FIRE   : TCP leaves the box while still grasped
    Premature gripper-open inside drops grasped->inside; leaving the box without a
    grasp drops inside->idle. update() returns True exactly once, on the firing
    frame, then re-arms (resets to idle).
    """

    def __init__(
        self,
        *,
        box: BoundingBox3D,
        tcp_idx=RIGHT_TCP_XYZ,
        grip_idx: int = RIGHT_GRIPPER_IDX,
        grasp_is_low: bool = True,
        grasp_threshold: float | None = None,
        confirm_inside: int = 2,
        confirm_grasp: int = 2,
    ) -> None:
        self.box = box
        self._tcp_idx = tuple(tcp_idx)
        self._grip_idx = grip_idx
        self._grasp = GraspCalibrator(grasp_is_low, grasp_threshold)
        self._confirm_inside = max(1, confirm_inside)
        self._confirm_grasp = max(1, confirm_grasp)
        # Per-frame signal snapshot for debug/visualization (no effect on logic);
        # update() refreshes it every frame.
        self.last: dict = {"phase": "idle", "xyz": None, "in_box": False, "grasp_closed": False}
        self.reset()

    def reset(self) -> None:
        self._phase = "idle"
        self._inside_count = 0
        self._outside_count = 0
        self._grasp_count = 0
        # NOTE: do not touch self.last here. reset() is called *inside* update() on
        # the firing frame (TCP exits the box), and clobbering the snapshot there
        # would hide the very signals that caused the pick. self.last is owned by
        # update() (set fresh each frame) and initialized in __init__.

    @property
    def phase(self) -> str:
        return self._phase

    def update(self, state: np.ndarray) -> bool:
        xyz = (state[self._tcp_idx[0]], state[self._tcp_idx[1]], state[self._tcp_idx[2]])
        inside = self.box.contains(xyz)
        is_closed = bool(self._grasp.update(float(state[self._grip_idx])))
        self.last = {
            "phase": self._phase,
            "xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            "in_box": bool(inside),
            "grasp_closed": is_closed,
        }

        if self._phase == "idle":
            self._inside_count = self._inside_count + 1 if inside else 0
            if self._inside_count >= self._confirm_inside:
                self._phase = "inside"
                self._grasp_count = 0
                self._outside_count = 0
        elif self._phase == "inside":
            if not inside:
                self._outside_count += 1
                if self._outside_count >= self._confirm_inside:
                    self.reset()
                return False
            self._outside_count = 0
            self._grasp_count = self._grasp_count + 1 if is_closed else 0
            if self._grasp_count >= self._confirm_grasp:
                self._phase = "grasped"
        elif self._phase == "grasped":
            if not is_closed:
                self._phase = "inside"
                self._grasp_count = 0
                return False
            if not inside:
                self.reset()
                return True
        return False


@dataclass
class HomePose:
    """Reference 'init' pose for both TCPs; episode end = arm returned home."""

    left_xyz: tuple
    right_xyz: tuple
    tol: float = 0.05  # meters (L2 distance, per arm)

    def at_home(self, state: np.ndarray) -> bool:
        lx = np.array([state[i] for i in LEFT_TCP_XYZ], dtype=float)
        rx = np.array([state[i] for i in RIGHT_TCP_XYZ], dtype=float)
        return (
            np.linalg.norm(lx - np.asarray(self.left_xyz, dtype=float)) <= self.tol
            and np.linalg.norm(rx - np.asarray(self.right_xyz, dtype=float)) <= self.tol
        )
