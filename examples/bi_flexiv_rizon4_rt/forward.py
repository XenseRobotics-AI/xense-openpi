"""Forward selected observation channels to a downstream video-playback laptop.

Plan B: the detection + seamless video-switching runs on a SEPARATE machine
(the video-playback computer). This subscriber lives in the robot host laptop's
control process and forwards only the two channels that machine needs — the
**head camera stream** and the **robot state** — over a websocket, at a
configurable rate (subscribe_hz). All detection/render cost stays off the laptop,
and the per-step cost on the laptop is kept ~zero (see on_step).

Design constraints (see xense_client.runtime.runtime.Runtime._step, which calls
subscriber.on_step INSIDE the control loop):
- on_step must NEVER block. It only drops the latest frame into a single-slot
  queue (discarding any stale frame) and returns immediately.
- A daemon thread owns the websocket: it connects, sends, and reconnects on
  failure. Every exception is swallowed so a dead video-playback laptop, a slow
  link, or a network hiccup can never disturb the robot's 30 Hz control loop or
  its home-on-exit motion.
- Optional STARTUP handshake (require_handshake): before the control loop starts,
  block until the video-playback laptop's obs server is reachable and greets us —
  mirroring the VLA policy client (WebsocketClientPolicy._wait_for_server), which
  waits for the inference server before running. This is the ONE place forwarding
  may block, and only at construction — never in on_step. Once running, the
  swallow-all-errors / never-block contract above still holds.

The wire format is msgpack (xense_client.msgpack_numpy), the same codec the
inference channel uses. The video-playback laptop can decode it with a vendored
copy of msgpack_numpy.py (no xense_client dependency required).
"""

import contextlib
import queue
import threading
import time

from typing_extensions import override
import websockets.sync.client
from xense_client import msgpack_numpy
from xense_client.logger import get_logger
from xense_client.runtime import subscriber as _subscriber

logger = get_logger("ForwardSubscriber")


class ForwardSubscriber(_subscriber.Subscriber):
    """Push a slim per-step payload to a downstream websocket receiver.

    Connects OUT to the video-playback laptop (which runs the websocket server), so
    the robot laptop needs no inbound ports. One-way: obs out, nothing read back.
    """

    def __init__(
        self,
        uri: str,
        *,
        cameras: tuple[str, ...] = ("head",),
        send_state: bool = True,
        send_action: bool = False,
        subscribe_hz: float = 0.0,
        send_stride: int = 1,
        connect_timeout_s: float = 2.0,
        require_handshake: bool = False,
        handshake_timeout_s: float = 0.0,
    ) -> None:
        """
        The detector needs exactly two channels: the **head camera stream** and
        the **robot state**. Those are the defaults; everything else is opt-in or
        dropped to keep the per-step cost ~zero.

        Args:
            uri: ws URI of the video-playback laptop, e.g. "ws://192.168.1.50:9100".
            cameras: which raw cameras to forward from observation["images_raw"].
                Defaults to the head camera only — head video + state is all the
                detector needs; wrists are dropped to save bandwidth.
            send_state: forward the 20-D robot state (observation["state"]).
            send_action: also forward the model action (observation/debug only —
                the detector does not use it). Off by default to stay lean.
            subscribe_hz: cap the forward rate to this many frames/sec via a
                wall-clock throttle, independent of the control loop's runtime_hz.
                0 = forward on every step. e.g. 10 streams the head at 10 Hz while
                control still runs at 30 Hz.
            send_stride: forward every Nth step (1 = every step). Integer
                subsample; prefer subscribe_hz to target an actual rate.
            connect_timeout_s: per-attempt websocket connect timeout.
            require_handshake: if True, block in __init__ until the receiver is up
                and greets us (see _wait_for_receiver) before returning — so the
                caller never starts inference while the video-playback laptop is
                unreachable. Mirrors WebsocketClientPolicy._wait_for_server.
            handshake_timeout_s: with require_handshake, give up and raise
                ConnectionError after this many seconds. 0 = wait forever (retry),
                exactly like the VLA client.
        """
        self._uri = uri
        self._cameras = cameras
        self._send_state = send_state
        self._send_action = send_action
        self._send_stride = max(1, int(send_stride))
        self._min_period = (1.0 / subscribe_hz) if subscribe_hz and subscribe_hz > 0 else 0.0
        self._last_enqueue_t = 0.0
        self._connect_timeout_s = connect_timeout_s

        self._packer = msgpack_numpy.Packer()
        # Single-slot: on_step always overwrites, so the sender thread only ever
        # ships the freshest frame and never builds a backlog.
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ws = None
        self._step_idx = 0
        self._dropped = 0
        self._sent = 0

        # Startup handshake: block until the video-playback laptop is up and greets
        # us, so the caller (main.py) never proceeds to inference with the screen
        # unreachable. Raises ConnectionError on timeout. The open connection is
        # reused by the sender thread below.
        if require_handshake:
            self._ws = self._wait_for_receiver(handshake_timeout_s)

        self._worker = threading.Thread(target=self._run, name="forward-sender", daemon=True)
        self._worker.start()
        rate = f"{subscribe_hz:g} Hz" if self._min_period > 0.0 else f"every {self._send_stride} step(s)"
        logger.info(
            f"ForwardSubscriber → {uri} (cameras={cameras}, state={send_state}, action={send_action}, rate={rate})"
        )

    # ----- Subscriber API (runs in the control loop; keep it cheap) -----

    @override
    def on_episode_start(self) -> None:
        self._step_idx = 0

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        # This runs INSIDE the 30 Hz control loop, so keep it ~zero-cost: cheap
        # gates first, then stash *references* to the two key channels (no array
        # copy, no serialization) and a single-slot enqueue. All real work —
        # msgpack of the head frame, socket send, reconnect — is done off-loop on
        # the daemon thread below. The head image is taken by reference straight
        # from the obs buffer; latest-wins semantics (drop-oldest) mean a torn
        # frame is at worst one stale head image, which the detector tolerates.
        self._step_idx += 1
        if self._step_idx % self._send_stride != 0:
            return
        if self._min_period > 0.0:
            now = time.monotonic()
            if (now - self._last_enqueue_t) < self._min_period:
                return  # not due yet at the requested subscribe_hz
            self._last_enqueue_t = now

        payload: dict = {"step": self._step_idx}
        if self._send_state:
            state = observation.get("state")
            if state is not None:
                payload["state"] = state  # 20-D, by reference

        raw = observation.get("images_raw")
        if raw is not None:
            imgs = {cam: raw[cam] for cam in self._cameras if cam in raw}  # head, by reference
            if imgs:
                payload["images"] = imgs

        if self._send_action:
            act = action.get("actions") if isinstance(action, dict) else None
            if act is not None:
                payload["action"] = act

        # Non-blocking enqueue, drop-oldest. Never blocks the control loop.
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(payload)

    @override
    def on_episode_end(self) -> None:
        # Don't tear down the socket between episodes; the detector keeps running.
        logger.info(f"ForwardSubscriber episode end: sent={self._sent}, dropped(stale)={self._dropped}")

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)
        self._safe_close_ws()

    # ----- Background sender thread (owns the socket; swallows all errors) -----

    def _run(self) -> None:
        # Reuse the connection opened by the startup handshake if there is one;
        # otherwise open it now (require_handshake=False / lazy path).
        if self._ws is None:
            self._connect()
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._ws is None and not self._connect():
                continue
            try:
                self._ws.send(self._packer.pack(payload))
                self._sent += 1
            except Exception as e:
                logger.warning(f"forward send failed ({e}); reconnecting")
                self._reconnect()
        self._safe_close_ws()

    # ----- Startup handshake (mirrors WebsocketClientPolicy._wait_for_server) -----

    def _wait_for_receiver(self, timeout_s: float):
        """Block until the video-playback laptop's obs server is up and greets us,
        then return the open connection. Retries on failure; raises ConnectionError
        once timeout_s (> 0) elapses. timeout_s <= 0 waits forever, exactly like the
        VLA policy client waiting for the inference server."""
        logger.info(f"Waiting for video-playback laptop at {self._uri} ...")
        deadline = (time.monotonic() + timeout_s) if timeout_s and timeout_s > 0 else None
        attempt = 0
        while not self._stop.is_set():
            ws = self._connect_and_greet()
            if ws is not None:
                logger.info(f"ForwardSubscriber handshake OK with {self._uri}")
                return ws
            attempt += 1
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise ConnectionError(
                    f"No video-playback laptop obs server at {self._uri} after {timeout_s:g}s. "
                    f"Start `python -m examples.dewu_video_switch.app` on the screen PC, or drop "
                    f"--subscribe to run without it."
                )
            logger.info(f"Still waiting for video-playback laptop at {self._uri} (attempt {attempt})...")
            backoff = 3.0 if deadline is None else min(3.0, max(0.0, deadline - now))
            self._stop.wait(backoff)  # interruptible backoff (Ctrl-C / close() breaks out)
        raise ConnectionError(f"ForwardSubscriber handshake to {self._uri} aborted before connecting")

    def _connect_and_greet(self):
        """One handshake attempt: open the ws and wait for the receiver's hello frame.
        Returns the open connection on success, or None (closing any half-open socket)
        so _wait_for_receiver retries. Requiring the greeting confirms the detection
        app is really there, not just some listener on the port."""
        try:
            ws = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=self._connect_timeout_s,
            )
        except Exception as e:
            logger.debug(f"forward connect failed ({e})")
            return None
        try:
            msg = ws.recv(timeout=self._connect_timeout_s)
            hello = msgpack_numpy.unpackb(msg) if isinstance(msg, (bytes, bytearray)) else msg
            logger.info(f"ForwardSubscriber greeted by receiver: {hello}")
            return ws
        except Exception as e:
            logger.debug(f"connected to {self._uri} but got no handshake hello ({e}); retrying")
            with contextlib.suppress(Exception):
                ws.close()
            return None

    def _connect(self) -> bool:
        try:
            self._ws = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=self._connect_timeout_s,
            )
            logger.info(f"ForwardSubscriber connected to {self._uri}")
            return True
        except Exception as e:
            self._ws = None
            logger.debug(f"forward connect failed ({e}); will retry")
            return False

    def _reconnect(self) -> None:
        self._safe_close_ws()
        # Brief backoff without blocking shutdown responsiveness.
        self._stop.wait(0.5)
        if not self._stop.is_set():
            self._connect()

    def _safe_close_ws(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None


def make_forward_subscriber(
    uri: str,
    *,
    cameras: tuple[str, ...] = ("head",),
    send_state: bool = True,
    send_action: bool = False,
    subscribe_hz: float = 0.0,
    send_stride: int = 1,
    require_handshake: bool = False,
    handshake_timeout_s: float = 0.0,
) -> ForwardSubscriber:
    """Convenience factory mirroring make_recorder_subscriber's style."""
    return ForwardSubscriber(
        uri,
        cameras=cameras,
        send_state=send_state,
        send_action=send_action,
        subscribe_hz=subscribe_hz,
        send_stride=send_stride,
        require_handshake=require_handshake,
        handshake_timeout_s=handshake_timeout_s,
    )
