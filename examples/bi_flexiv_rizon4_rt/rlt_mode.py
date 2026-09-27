"""RLT mode: serve this bench to an online RLT training server (``openpi.rlt.env_protocol``).

Entered from ``main.py`` with ``--args.rlt``. The training server
(``scripts/rlt/train_rl.py``) listens; this process dials ``--args.host/--args.port``
and executes the chunks it is sent. Pico4 is required: the grips take over the
arms (a held grip arms the takeover; it starts once a controller moves past the
server's threshold), and the face buttons are the operator's controls:

- ``A``: at the reset gate, start the round; during a round, end it (the arms home).
- ``B``: open a recording window for the next chunk; pressed while one is open, label it success.
- ``Y``: label the open window failure.
- ``X``: discard the round's labeled data and close the window.

A label pressed mid-chunk lets the chunk run out: it is reported by the first
segment that completes all its steps, so a labeled phase always ends on a whole unit.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from examples.bi_flexiv_rizon4_rt.main import Args

PROTOCOL = "openpi-rlt/1"  # openpi.rlt.env_protocol.PROTOCOL; not imported to keep the robot host light
STATE_DIM = ACTION_DIM = 20
# Takeover continuations buffered in one reply before refusing to go on.
_MAX_SEGMENTS = 256


class Operator:
    """Face-button state machine; the robot loop feeds it presses and reads the controls."""

    def __init__(self) -> None:
        self.recording_requested = False
        self.label: str | None = None
        self.round_end = False
        self.discard = False

    def press(self, button: str, *, window_open: bool) -> None:
        if button == "A":
            self.round_end = True
        elif button == "B":
            if window_open:
                self.label = self.label or "success"
            else:
                self.recording_requested = True
        elif button == "Y" and window_open:
            self.label = self.label or "failure"
        elif button == "X":
            self.discard = True
            self.recording_requested = False
            self.label = None
        logging.info("Pico4 %s: recording_requested=%s label=%s", button, self.recording_requested, self.label)

    def new_round(self) -> None:
        self.__init__()


class Session:
    """Executes server requests on the bench. ``env`` and ``controller`` are the hardware seams."""

    def __init__(self, env, controller, *, step_dt: float | None, takeover_motion: type | None = None) -> None:
        if takeover_motion is None:
            from examples.bi_flexiv_rizon4_rt.intervention import TakeoverMotion as takeover_motion  # noqa: N813
        self.env = env
        self.controller = controller
        self._takeover_motion = takeover_motion
        self.step_dt = step_dt
        self.operator = Operator()
        self.recording = False  # latched window state the next segment runs under
        self._window_steps = 0
        self._capture_stride = 1
        self._last_gripper: np.ndarray | None = None
        self._homed = False

    def observe(self) -> dict[str, Any]:
        obs = self.env.get_observation()
        return {"state": np.asarray(obs["state"], np.float32), "images": dict(obs["images"])}

    def _drain_buttons(self) -> None:
        for button in self.controller.consume_button_events():
            self.operator.press(button, window_open=self.recording)

    def home(self) -> None:
        self.env.reset()
        self.controller.reset_for_new_episode()
        self._last_gripper = None
        self._homed = True

    def reset(self, request: dict) -> dict:
        self.controller.set_takeover_motion(
            self._takeover_motion(request["takeover_position_m"], math.radians(request["takeover_rotation_deg"]))
        )
        self._capture_stride = int(request["capture_stride"])
        if not self._homed:
            self.home()
        logging.info("Arms homed. Press Pico4 A to start the round.")
        self.operator.round_end = False
        while not self.operator.round_end:  # A at the gate starts the round
            self.controller.poll_buttons()
            self._drain_buttons()
            time.sleep(0.05)
        self.operator.new_round()
        self.recording = False
        self._homed = False
        return {"obs": self.observe(), "recording": False}

    def _latch(self) -> None:
        """Apply a requested window at a segment boundary, never mid-segment."""
        self._drain_buttons()
        if self.operator.recording_requested and not self.recording:
            self.recording = True
            self._window_steps = 0

    def chunk(self, request: dict) -> dict:
        actions = np.asarray(request["actions"], np.float32)
        steps = len(actions)
        segments, captures = [], []
        for index in range(_MAX_SEGMENTS):
            self._latch()
            recording = self.recording
            executed, human = [], []
            released = False
            op = self.operator
            for step in range(steps):
                started = time.monotonic()
                active = self.controller.poll_and_decide(gripper_command=self._last_gripper)
                # A release hands control back: the server replans from the current observation.
                # Continuation segments only ever run human steps.
                released = self.controller.consume_release_event() or (index > 0 and not active)
                self._drain_buttons()
                if op.round_end or op.discard or released:
                    break
                action = self.controller.get_override_action() if active else actions[step]
                self.env.apply_action({"actions": action})
                self._last_gripper = np.asarray(action[-2:], np.float32).copy()
                executed.append(np.asarray(action, np.float32))
                human.append(bool(active))
                if recording:
                    self._window_steps += 1
                    if self._window_steps % self._capture_stride == 0:
                        captures.append({"step": self._window_steps, "obs": self.observe()})
                if self.step_dt is not None:
                    time.sleep(max(0.0, self.step_dt - (time.monotonic() - started)))
            complete = len(executed) == steps and not (op.round_end or op.discard)
            label = op.label if recording and complete else None
            if label is not None:
                op.label = None
                op.recording_requested = False
                self.recording = False
            if op.discard:
                self.recording = False
            segments.append(
                {
                    "obs": self.observe(),
                    "executed": np.asarray(executed, np.float32).reshape(-1, ACTION_DIM),
                    "human": np.asarray(human, bool),
                    "recording": recording,
                    "label": label,
                    "round_end": op.round_end,
                    "discard": op.discard,
                }
            )
            op.discard = False
            if op.round_end or released or label is not None or not human or not human[-1]:
                break
        else:
            raise RuntimeError(
                f"Takeover exceeded {_MAX_SEGMENTS} segments; aborting rather than resuming a stale chunk."
            )
        self._latch()
        if segments[-1]["round_end"]:
            self.home()  # home right away; the server trains meanwhile
        return {"segments": segments, "captures": captures, "recording_next": self.recording}


def serve(conn, session: Session, msgpack_numpy) -> None:
    """Answer server requests until it closes the session."""
    import traceback

    packer = msgpack_numpy.Packer()
    while True:
        request = msgpack_numpy.unpackb(conn.recv())
        op = request["op"]
        if op == "status":
            logging.info("Server: %s", request["text"])
            continue
        try:
            if op == "reset":
                reply = session.reset(request)
            elif op == "chunk":
                reply = session.chunk(request)
            elif op == "close":
                conn.send(packer.pack({}))
                return
            else:
                raise ValueError(f"Unknown op {op!r}")
        except Exception:
            conn.send(packer.pack({"error": traceback.format_exc()}))
            raise
        conn.send(packer.pack(reply))


def run(args: Args, env, controller) -> None:
    """Dial the training server and serve it, reconnecting across drops."""
    import websockets.exceptions
    import websockets.sync.client
    from xense_client import msgpack_numpy

    if controller is None:
        raise SystemExit("RLT mode is driven from the Pico4 controllers; relaunch with --args.pico4-intervention.")
    session = Session(env, controller, step_dt=1.0 / args.runtime_hz)
    hello = {"protocol": PROTOCOL, "state_dim": STATE_DIM, "action_dim": ACTION_DIM}
    uri = f"ws://{args.host}:{args.port}"
    while True:
        try:
            with websockets.sync.client.connect(uri, compression=None, max_size=None) as conn:
                conn.send(msgpack_numpy.Packer().pack(hello))
                logging.info("Connected to the RLT training server at %s.", uri)
                serve(conn, session, msgpack_numpy)
                return
        except (ConnectionRefusedError, OSError):
            logging.info("Waiting for the RLT training server at %s...", uri)
            time.sleep(5)
        except websockets.exceptions.ConnectionClosed as exc:
            logging.warning("Connection to the training server lost (%s); reconnecting.", exc)
