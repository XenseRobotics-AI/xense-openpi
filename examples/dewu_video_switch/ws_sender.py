"""Dependency-free websocket sender for local testing (no xense_client).

`replay_lerobot.py` and `sim_laptop.py` are meant to run the *exact* production
path via examples.bi_flexiv_rizon4_rt.subscribe.ObsSubscriber — but that
imports `xense_client`, which only exists on the robot laptop. On a dev / display
machine (which by design carries only the detection-app deps in
requirements.txt) that import fails.

This module provides a drop-in fallback that speaks the identical wire format
(msgpack payload ``{step, state, images:{cam:arr}, action}``) using only the
vendored `msgpack_numpy` + `websockets`. `make_obs_subscriber_auto` returns
the real ObsSubscriber when available and this vendored sender otherwise, so
the same script works on the robot laptop and on the playback PC unchanged.

It is synchronous (sends inside on_step) — fine for a standalone replay/sim
script, which is not the latency-critical robot control loop the real subscriber
must never block.
"""

from __future__ import annotations

import contextlib
import time

import websockets.sync.client

try:
    from examples.dewu_video_switch import msgpack_numpy
except ImportError:  # standalone copy on the dev machine
    import msgpack_numpy  # type: ignore


class VendoredObsSender:
    """Mirror of ObsSubscriber's payload, minus the robot-SDK dependency."""

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
    ) -> None:
        self._uri = uri
        self._cameras = tuple(cameras)
        self._send_state = send_state
        self._send_action = send_action
        self._send_stride = max(1, int(send_stride))
        self._min_period = (1.0 / subscribe_hz) if subscribe_hz and subscribe_hz > 0 else 0.0
        self._last_enqueue_t = 0.0
        self._timeout = connect_timeout_s
        self._ws = None
        self._step_idx = 0
        self._sent = 0

    def on_episode_start(self) -> None:
        self._step_idx = 0

    def _connect(self) -> bool:
        try:
            self._ws = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None, open_timeout=self._timeout
            )
            print(f"[sender] connected to {self._uri}")
            return True
        except Exception as e:  # app not up yet / network: retry on the next frame
            self._ws = None
            print(f"[sender] connect failed ({e}); will retry")
            return False

    def on_step(self, observation: dict, action: dict) -> None:
        self._step_idx += 1
        if self._step_idx % self._send_stride != 0:
            return
        if self._min_period > 0.0:
            now = time.monotonic()
            if (now - self._last_enqueue_t) < self._min_period:
                return  # not due yet at the requested subscribe_hz
            self._last_enqueue_t = now

        payload: dict = {"step": self._step_idx}
        if self._send_state and "state" in observation:
            payload["state"] = observation["state"]
        raw = observation.get("images_raw", {})
        imgs = {cam: raw[cam] for cam in self._cameras if cam in raw}
        if imgs:
            payload["images"] = imgs
        if self._send_action:
            act = action.get("actions") if isinstance(action, dict) else None
            if act is not None:
                payload["action"] = act

        if self._ws is None and not self._connect():
            return
        try:
            self._ws.send(msgpack_numpy.packb(payload))
            self._sent += 1
        except Exception as e:
            print(f"[sender] send failed ({e}); reconnecting")
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None

    def on_episode_end(self) -> None:
        print(f"[sender] episode end: sent={self._sent}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._ws is not None:
                self._ws.close()
        self._ws = None


def make_sender(uri: str, **kwargs) -> VendoredObsSender:
    return VendoredObsSender(uri, **kwargs)


def make_obs_subscriber_auto(uri: str, **kwargs):
    """Real ObsSubscriber if xense_client is importable, else vendored sender.

    Both expose the same on_episode_start / on_step / on_episode_end / close API
    and the same msgpack wire format, so callers don't care which they get.
    """
    try:
        from examples.bi_flexiv_rizon4_rt.subscribe import make_obs_subscriber

        return make_obs_subscriber(uri, **kwargs)
    except Exception as e:
        print(f"[sender] production ObsSubscriber unavailable ({e}); using vendored sender")
        return make_sender(uri, **kwargs)
