"""Per-shoe state-machine detector — the production detection logic.

States 0..4: 0 = standby / reset (arm at init), 1..4 = currently working shoe N.

  - Pick event (advances state N-1 -> N): the RIGHT TCP enters a 3D bounding box
    (the shoebox), grasps inside, then exits the box while still grasping
    (pose_events.BoxGraspEdgeEvent). Each pick -> scene "detecting".
  - Blue event (within states 1..4): an OpenCV check on the head image runs about
    once per second (vision_stride frames); when the blue insole bottom appears
    -> scene "next". Re-armed when a new shoe starts.
  - Reset (4 -> 0): in the final state, when BOTH TCPs return to the init/home
    pose (the 4-shoe episode is done and the arm homes) -> scene "standby".

Emission model (Option A): detect() returns the LEVEL scene id that should be
showing *now*, every frame. It is internally idempotent and only changes the
level when an event fires, so app.py's SceneController can debounce/min-dwell on
top unchanged. The three scene ids must match web/index.html SCENES.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# Support `python -m examples.dewu_video_switch.app` and standalone `python app.py`.
try:
    from examples.dewu_video_switch.detector import Detector
    from examples.dewu_video_switch.pose_events import (
        LEFT_TCP_XYZ,
        RIGHT_TCP_XYZ,
        BoundingBox3D,
        BoxGraspEdgeEvent,
        HomePose,
    )
    from examples.dewu_video_switch.vision import BlueInsoleConfig
except ImportError:  # standalone copy on the video-playback laptop
    from detector import Detector  # type: ignore
    from pose_events import (  # type: ignore
        LEFT_TCP_XYZ,
        RIGHT_TCP_XYZ,
        BoundingBox3D,
        BoxGraspEdgeEvent,
        HomePose,
    )
    from vision import BlueInsoleConfig  # type: ignore

SCENE_STANDBY = "standby"
SCENE_DETECTING = "detecting"
SCENE_NEXT = "next"


@dataclass
class TransitionConfig:
    """One pick transition: which box/arm/gripper defines it."""

    box: BoundingBox3D
    tcp_idx: tuple = RIGHT_TCP_XYZ  # right TCP by default (data-driven per shoe later)
    grip_idx: int = 19
    grasp_is_low: bool = True
    grasp_threshold: float | None = None  # None => self-calibrate


@dataclass
class ShoeSMConfig:
    n_shoes: int = 4
    # Scenes alternate by shoe group. The shoes cycle through `scene_groups` visual
    # groups (e.g. white / red shoes -> 2 groups), so state N uses group
    # ((N-1) % scene_groups)+1. The emitted scene id is "<group>-1" while detecting
    # (after the pick) and "<group>-2" after the blue insole -> so the flow is
    # standby -> 1-1 -> 1-2 -> 2-1 -> 2-2 -> 1-1 -> 1-2 -> 2-1 -> 2-2 -> standby.
    scene_groups: int = 2
    transitions: list = field(default_factory=list)  # len == n_shoes (one box per pick)
    blue: BlueInsoleConfig = field(default_factory=BlueInsoleConfig)
    vision_stride: int = 30   # run the blue check every N frames (~1 Hz @ 30 fps)
    vision_confirm: int = 2   # consecutive positive blue checks before firing
    home_tol: float = 0.05    # meters; both TCPs within tol of init => home
    home_pose: HomePose | None = None  # None => auto-capture from initial state-0 frames
    # Insole-flip pose gate. The presenting arm rotates the insole to show its blue
    # underside to the head cam, then HOLDS it. So blue is only accepted while that
    # arm's 6D orientation is far (> flip_min_delta) from the pose it had at the pick,
    # AND has stayed there for flip_confirm frames. This rejects the blue box-wall / mat
    # false positives (which fire ~1 s after the pick, when the arm is still in its carry
    # pose, delta ~ 0) and — being proprioceptive — is invariant to scene and lighting.
    # Measured: carry pose delta ~ 0.0, flip-and-hold delta ~ 1.9-2.8 across both scenes.
    # Set use_flip_gate false to fall back to vision-only.
    use_flip_gate: bool = True
    flip_min_delta: float = 1.0
    flip_confirm: int = 8     # frames the flipped pose must hold before blue is allowed

    @classmethod
    def from_json(cls, path: str) -> "ShoeSMConfig":
        """Overlay a JSON config onto the defaults. See shoe_sm.example.json.

        Schema (all optional except a box source):
            n_shoes, vision_stride, vision_confirm, home_tol
            pick_box:   {x_min,x_max,y_min,y_max,z_min,z_max}   # reused for all picks
            pick_boxes: [ {..6..}, ... ]                        # one per pick (overrides pick_box)
            grip_idx, grasp_is_low, grasp_threshold            # applied to every transition
            blue: {h_lo,h_hi,s_lo,s_hi,v_lo,v_hi,min_area_frac,roi,open_ksize,close_ksize}
            home_pose: {left_xyz:[x,y,z], right_xyz:[x,y,z]}
        """
        with open(path) as f:
            d = json.load(f)
        cfg = cls()
        cfg.n_shoes = int(d.get("n_shoes", cfg.n_shoes))
        cfg.scene_groups = int(d.get("scene_groups", cfg.scene_groups))
        cfg.vision_stride = int(d.get("vision_stride", cfg.vision_stride))
        cfg.vision_confirm = int(d.get("vision_confirm", cfg.vision_confirm))
        cfg.home_tol = float(d.get("home_tol", cfg.home_tol))
        cfg.use_flip_gate = bool(d.get("use_flip_gate", cfg.use_flip_gate))
        cfg.flip_min_delta = float(d.get("flip_min_delta", cfg.flip_min_delta))
        cfg.flip_confirm = int(d.get("flip_confirm", cfg.flip_confirm))

        def _box(b):
            return BoundingBox3D(b["x_min"], b["x_max"], b["y_min"], b["y_max"], b["z_min"], b["z_max"])

        grip_idx = int(d.get("grip_idx", 19))
        grasp_is_low = bool(d.get("grasp_is_low", True))
        grasp_threshold = d.get("grasp_threshold")
        tcp_idx = tuple(d.get("tcp_idx", RIGHT_TCP_XYZ))
        if "transitions" in d:
            # Full per-pick spec: each entry sets its own box and (optionally) which
            # arm detects it via tcp_idx/grip_idx — e.g. picks 1-2 on the RIGHT arm
            # and picks 3-4 on the LEFT arm when the two arms alternate take-outs.
            # Top-level tcp_idx/grip_idx/grasp_* are the per-entry defaults.
            cfg.transitions = [
                TransitionConfig(
                    box=_box(e["box"]),
                    tcp_idx=tuple(e.get("tcp_idx", tcp_idx)),
                    grip_idx=int(e.get("grip_idx", grip_idx)),
                    grasp_is_low=bool(e.get("grasp_is_low", grasp_is_low)),
                    grasp_threshold=e.get("grasp_threshold", grasp_threshold),
                )
                for e in d["transitions"]
            ]
            cfg.n_shoes = len(cfg.transitions)
        else:
            if "pick_boxes" in d:
                boxes = [_box(b) for b in d["pick_boxes"]]
            elif "pick_box" in d:
                boxes = [_box(d["pick_box"])] * cfg.n_shoes
            else:
                boxes = []  # falls back to placeholder default below
            cfg.transitions = [
                TransitionConfig(box=b, tcp_idx=tcp_idx, grip_idx=grip_idx,
                                 grasp_is_low=grasp_is_low, grasp_threshold=grasp_threshold)
                for b in boxes
            ]

        if "blue" in d:
            cfg.blue = BlueInsoleConfig(**{k: (tuple(v) if k == "roi" and v is not None else v)
                                           for k, v in d["blue"].items()})
        if "home_pose" in d:
            hp = d["home_pose"]
            cfg.home_pose = HomePose(tuple(hp["left_xyz"]), tuple(hp["right_xyz"]), cfg.home_tol)
        return cfg


def _placeholder_config() -> ShoeSMConfig:
    """Default with a PLACEHOLDER pick box — runs, but must be calibrated.

    The box below is a loose guess in robot-base meters; replace via a tuned
    shoe_sm.json (see replay_offline.py --calibrate-tcp).
    """
    cfg = ShoeSMConfig()
    box = BoundingBox3D(x_min=0.45, x_max=0.95, y_min=-0.35, y_max=0.35, z_min=-0.40, z_max=0.05)
    cfg.transitions = [TransitionConfig(box=box) for _ in range(cfg.n_shoes)]
    return cfg


class ShoeStateMachineDetector(Detector):
    def __init__(self, cfg: ShoeSMConfig | None = None, config_path: str | None = None) -> None:
        if config_path:
            cfg = ShoeSMConfig.from_json(config_path)
        cfg = cfg or _placeholder_config()
        if not cfg.transitions:
            cfg.transitions = _placeholder_config().transitions
        self.cfg = cfg

        self._picks = [
            BoxGraspEdgeEvent(box=t.box, tcp_idx=t.tcp_idx, grip_idx=t.grip_idx,
                              grasp_is_low=t.grasp_is_low, grasp_threshold=t.grasp_threshold)
            for t in cfg.transitions
        ]
        self._blue = None  # lazy (needs cv2)
        self._home = cfg.home_pose
        self._home_capture: list = []

        self._state = 0
        self._blue_fired = False
        self._blue_pos_count = 0
        self._frame_i = 0
        self._last_blue_area: float | None = None
        # Insole-flip gate state (see ShoeSMConfig): reference orientation captured at
        # the pick, which arm's 6D orientation to watch, and the hold counter.
        self._flip_ref: np.ndarray | None = None
        self._flip_ori_slice: slice | None = None
        self._flip_count = 0
        self._last_flip: dict = {"delta": None, "ready": False}
        # Per-frame debug snapshot (filled by detect(); read by replay_debug).
        self.last: dict = {}

    # ----- introspection (used by replay_offline timeline) -----
    @property
    def state(self) -> int:
        return self._state

    def _record(self, event, from_state, to_state, pick_dbg, blue_dbg) -> str:
        """Stash a per-frame debug snapshot and return the current scene level."""
        scene = self._level()
        self.last = {
            "frame": self._frame_i,
            "state": self._state,
            "n_shoes": self.cfg.n_shoes,
            "scene": scene,
            "event": event,            # None | "pick" | "blue" | "reset"
            "from_state": from_state,
            "to_state": to_state,
            "pick": pick_dbg,          # {phase, xyz, in_box, grasp_closed, target_state} | None
            "blue": blue_dbg,          # {checked, area_frac, present}
            "blue_fired": self._blue_fired,
            "last_blue_area": self._last_blue_area,
            "flip": dict(self._last_flip),   # {delta, ready} — insole-flip pose gate
        }
        return scene

    def _flip_gate(self, state) -> bool:
        """Return True once the presenting arm has rotated the insole far from its pick
        pose and held there for flip_confirm frames. Called every frame while a shoe is
        active so the hold counter stays continuous. Rejects blue false positives on the
        boxes/mat, which fire at the carry pose (delta ~ 0)."""
        if not self.cfg.use_flip_gate:
            self._last_flip = {"delta": None, "ready": True}
            return True
        if self._flip_ref is None or self._flip_ori_slice is None:
            self._last_flip = {"delta": None, "ready": False}
            return False
        ori = np.asarray(state[self._flip_ori_slice], dtype=float)
        delta = float(np.linalg.norm(ori - self._flip_ref))
        self._flip_count = self._flip_count + 1 if delta > self.cfg.flip_min_delta else 0
        ready = self._flip_count >= self.cfg.flip_confirm
        self._last_flip = {"delta": delta, "ready": ready}
        return ready

    def _ensure_blue(self):
        if self._blue is None:
            try:
                from examples.dewu_video_switch.vision import BlueInsoleDetector
            except ImportError:
                from vision import BlueInsoleDetector  # type: ignore
            self._blue = BlueInsoleDetector(self.cfg.blue)
        return self._blue

    def _level(self) -> str:
        if self._state == 0:
            return SCENE_STANDBY
        group = ((self._state - 1) % max(1, self.cfg.scene_groups)) + 1  # 1,2,1,2,... by shoe
        phase = 2 if self._blue_fired else 1                             # 1=detecting, 2=after blue
        return f"{group}-{phase}"

    def _reset_cycle(self) -> None:
        self._state = 0
        self._blue_fired = False
        self._blue_pos_count = 0
        self._flip_ref = None
        self._flip_ori_slice = None
        self._flip_count = 0
        for p in self._picks:
            p.reset()

    def detect(self, frame: dict) -> str | None:
        self._frame_i += 1
        blue_dbg = {"checked": False, "area_frac": None, "present": None}
        pick_dbg = None

        state = frame.get("state")
        if not isinstance(state, np.ndarray) or state.shape[-1] < 20:
            return self._record(None, self._state, self._state, pick_dbg, blue_dbg)

        # Auto-capture the home pose from the first state-0 frames (arm at init).
        if self._home is None and self._state == 0 and len(self._home_capture) < 5:
            self._home_capture.append(np.asarray(state, dtype=float))
            if len(self._home_capture) == 5:
                med = np.median(np.stack(self._home_capture), axis=0)
                self._home = HomePose(
                    left_xyz=tuple(float(med[i]) for i in LEFT_TCP_XYZ),
                    right_xyz=tuple(float(med[i]) for i in RIGHT_TCP_XYZ),
                    tol=self.cfg.home_tol,
                )

        # Pick event: advance to the next shoe state.
        if self._state < self.cfg.n_shoes:
            ev = self._picks[self._state]
            fired = ev.update(state)
            pick_dbg = dict(ev.last)
            pick_dbg["phase"] = ev.phase
            pick_dbg["target_state"] = self._state + 1
            pick_dbg["arm"] = "L" if tuple(self.cfg.transitions[self._state].tcp_idx) == LEFT_TCP_XYZ else "R"
            if fired:
                from_state = self._state
                self._state += 1
                self._blue_fired = False
                self._blue_pos_count = 0
                # Capture the presenting arm's 6D orientation as the flip reference.
                # Layout per arm: x,y,z, r1..r6 -> orientation starts 3 past the xyz.
                tcp0 = int(self.cfg.transitions[from_state].tcp_idx[0])
                self._flip_ori_slice = slice(tcp0 + 3, tcp0 + 9)
                self._flip_ref = np.asarray(state[self._flip_ori_slice], dtype=float)
                self._flip_count = 0
                return self._record("pick", from_state, self._state, pick_dbg, blue_dbg)

        # Blue event: the insole-flip pose gate AND the vision check, once each shoe.
        if 1 <= self._state <= self.cfg.n_shoes and not self._blue_fired:
            flip_ready = self._flip_gate(state)  # every frame -> keeps the hold counter live
            if flip_ready and self._frame_i % max(1, self.cfg.vision_stride) == 0:
                head = (frame.get("images") or {}).get("head")
                if isinstance(head, np.ndarray) and head.ndim == 3:
                    present, _dbg = self._ensure_blue().detect(head)
                    self._last_blue_area = _dbg.get("area_frac")
                    blue_dbg = {"checked": True, "area_frac": self._last_blue_area, "present": bool(present)}
                    self._blue_pos_count = self._blue_pos_count + 1 if present else 0
                    if self._blue_pos_count >= self.cfg.vision_confirm:
                        self._blue_fired = True
                        return self._record("blue", self._state, self._state, pick_dbg, blue_dbg)

        # Reset 4 -> 0 when the arm returns to the init/home pose (episode done).
        if self._state == self.cfg.n_shoes and self._home is not None and self._home.at_home(state):
            from_state = self._state
            self._reset_cycle()
            return self._record("reset", from_state, 0, pick_dbg, blue_dbg)

        return self._record(None, self._state, self._state, pick_dbg, blue_dbg)
