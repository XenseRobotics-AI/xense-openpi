"""Wire protocol between the RLT training server and the robot host (synchronous chunks).

The training server listens; the robot host dials in (the direction the policy
client already uses) and sends a hello::

    {"protocol": PROTOCOL, "state_dim": int, "action_dim": int}

Then the server drives, one msgpack frame per request and one per reply:

``{"op": "reset", "takeover_position_m", "takeover_rotation_deg", "capture_stride"}``
    Home, wait for the operator to start the round, reply ``{"obs", "recording"}``.

``{"op": "chunk", "actions": (C, A) absolute}``
    Execute the chunk. A human takeover continues as further segments of up to C
    human steps each, executed locally without a round trip; releasing the
    takeover ends the reply. Reply::

        {"segments": [segment, ...], "captures": [{"step": int, "obs": obs}, ...], "recording_next": bool}

    where each segment is::

        {"obs": obs after the segment, "executed": (n, A) absolute actions, "human": (n,) bool,
         "recording": bool (window open while it ran), "label": None | "success" | "failure",
         "round_end": bool, "discard": bool}

    A label is reported by the first segment that completes all C steps after
    the press, so the terminal window is always a whole unit. A window requested
    mid-segment opens at the next segment boundary; ``recording_next`` is the
    window state latched at the end of the reply, which the next chunk runs under. ``captures`` are observations taken while a window is open, after
    every ``capture_stride``-th step counted from the step the window opened;
    the server computes window features from them after a label.

``{"op": "status", "text": str}`` (no reply) and ``{"op": "close"}`` (reply ``{}``).

An observation is ``{"state": (S,), "images": {camera: (3, H, W) uint8}}``. A
reply ``{"error": str}`` aborts the request on the server.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from typing import Any

import websockets.exceptions
import websockets.sync.server
from xense_client import msgpack_numpy

PROTOCOL = "openpi-rlt/1"


class EnvConnectionLostError(RuntimeError):
    """The robot host disconnected; the current round cannot continue."""


class RemoteEnv:
    """Server end: listens for one robot host at a time and sends it requests."""

    def __init__(self, host: str, port: int, *, state_dim: int, action_dim: int):
        self._dims = {"state_dim": state_dim, "action_dim": action_dim}
        self._connections: queue.Queue = queue.Queue()
        self._conn = None
        self._done: threading.Event | None = None
        self._packer = msgpack_numpy.Packer()
        self._server = websockets.sync.server.serve(self._handle, host, port, compression=None, max_size=None)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        logging.info("RLT env server listening on %s:%d", host, port)

    def _handle(self, conn) -> None:
        hello = msgpack_numpy.unpackb(conn.recv())
        expected = {"protocol": PROTOCOL, **self._dims}
        if (got := {key: hello.get(key) for key in expected}) != expected:
            conn.send(self._packer.pack({"error": f"expected {expected}, got {got}"}))
            logging.error("Refused robot host %s: hello %s, expected %s", conn.remote_address, got, expected)
            return
        done = threading.Event()
        self._connections.put((conn, done))
        done.wait()  # the connection closes when this handler returns

    def wait_for_robot(self) -> None:
        if self._conn is not None:
            return
        logging.info("Waiting for the robot host to connect...")
        self._conn, self._done = self._connections.get()
        logging.info("Robot host %s connected.", self._conn.remote_address)

    def _drop(self) -> None:
        if self._done is not None:
            self._done.set()
        self._conn = self._done = None

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.wait_for_robot()
        try:
            self._conn.send(self._packer.pack(message))
            reply = msgpack_numpy.unpackb(self._conn.recv())
        except websockets.exceptions.ConnectionClosed as exc:
            self._drop()
            raise EnvConnectionLostError(str(exc)) from exc
        if "error" in reply:
            self._drop()
            raise EnvConnectionLostError(f"robot host error: {reply['error']}")
        return reply

    def status(self, text: str) -> None:
        """Show a line on the operator's terminal; best effort, no reply."""
        logging.info("%s", text)
        if self._conn is None:
            return
        try:
            self._conn.send(self._packer.pack({"op": "status", "text": text}))
        except websockets.exceptions.ConnectionClosed:
            self._drop()

    def close(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(EnvConnectionLostError):
                self.request({"op": "close"})
            self._drop()
        self._server.shutdown()
