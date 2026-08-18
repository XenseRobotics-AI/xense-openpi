"""Pace-decoupled wrapper around any chunked-policy broker.

Wraps a BasePolicy (typically ActionChunkBroker or RTCActionChunkBroker) so
that action consumption is paced by a dedicated thread rather than by the
outer obs loop. The wrapped broker's `.infer(obs)` is called from an internal
producer thread using the latest observation submitted via `submit_obs(obs)`;
returned actions are queued for the action thread to pop via `pop_action()`.

Use this when:
  * camera FPS pins the outer obs loop (e.g. 30 Hz) but you want to consume
    chunk actions faster (e.g. 60–90 Hz) to speed up robot execution.
  * the underlying broker's `.infer()` can occasionally block on the policy
    server (~hundreds of ms) — the queue absorbs the stall.

Semantics:
  * `submit_obs(obs)` is non-blocking. The latest obs replaces any earlier one
    that the producer thread hasn't picked up yet (single-slot, not a queue):
    we always want the freshest obs for the next inference, not stale backlog.
  * `pop_action()` blocks the action thread until an action is available.
  * The producer thread continuously calls `inner.infer(obs)` and pushes the
    result to `_action_queue`. If the queue is full (consumer is slow), the
    producer blocks on `put()` — natural back-pressure.

RTC note: RTCActionChunkBroker already has an internal inference thread + its
own action queue. Wrapping it adds a second queue layer (slight redundancy)
but the semantics remain correct as long as its `frequency_hz` is configured
to match the pace at which we pop from it — i.e. equal to action_hz, not the
obs loop rate.
"""

import queue
import threading
import time
from typing import override

from xense_client import base_policy as _base_policy
from xense_client.logger import get_logger

logger = get_logger("PacedBroker")


class PacedBroker(_base_policy.BasePolicy):
    """Decouples action-thread pace from the obs loop pace.

    Args:
        inner: Wrapped policy/broker. Must implement BasePolicy.
        queue_size: Max actions buffered between producer and consumer.
            Should be >= action_hz * worst_case_infer_block_seconds to avoid
            consumer stalls (e.g. 50 covers a 0.8 s websocket stall at 60 Hz).
        target_hz: If > 0, the producer sleeps after each iteration to cap its
            call rate at this value. Set this to the consumer (action thread)
            rate. Without it, the producer runs flat-out: harmless when the
            inner broker has no internal state (queue.put back-pressure
            eventually paces it), but harmful when the inner broker maintains
            its own queue — notably RTCActionChunkBroker. There, an unpaced
            producer drains RTC's 50-action buffer in milliseconds during
            startup, faster than RTC's background inference can refill,
            triggering "Action queue exhausted" warning spam at ~200 Hz.
            Setting target_hz=action_hz keeps RTC's internal queue at a
            healthy fill level.
        producer_idle_sleep: Seconds to sleep when no obs has been submitted
            yet, before retrying. Tiny; just prevents a tight spin at startup.
    """

    def __init__(
        self,
        inner: _base_policy.BasePolicy,
        queue_size: int = 50,
        target_hz: float = 0.0,
        producer_idle_sleep: float = 0.005,
    ) -> None:
        self._inner = inner
        self._action_queue: queue.Queue[dict] = queue.Queue(maxsize=queue_size)
        self._target_hz = target_hz

        # Single-slot latest obs; producer always reads the freshest one.
        self._latest_obs: dict | None = None
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()

        self._producer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._producer_idle_sleep = producer_idle_sleep

    # ------------------------------------------------------------------
    # BasePolicy passthrough for warmup/reset
    # ------------------------------------------------------------------
    @override
    def warmup(self, obs: dict) -> None:
        # JIT-compile + first-call latency happens here on the main thread,
        # before the producer/consumer pair takes over.
        self._inner.warmup(obs)

    @override
    def reset(self) -> None:
        # Caller is expected to have stopped the producer thread before reset.
        self._inner.reset()
        # Drain any stale actions and obs.
        try:
            while True:
                self._action_queue.get_nowait()
        except queue.Empty:
            pass
        with self._obs_lock:
            self._latest_obs = None
        self._obs_event.clear()

    @override
    def infer(self, obs: dict) -> dict:
        # Not used in the decoupled runtime; kept so PacedBroker still
        # satisfies BasePolicy and could fall back to synchronous use.
        # Synchronous fallback: submit obs and pop one action.
        self.submit_obs(obs)
        return self.pop_action()

    # ------------------------------------------------------------------
    # Decoupled-mode API
    # ------------------------------------------------------------------
    def submit_obs(self, obs: dict) -> None:
        """Replace the producer's latest-obs slot. Non-blocking, always overwrites."""
        with self._obs_lock:
            self._latest_obs = obs
        self._obs_event.set()

    def pop_action(self, timeout: float | None = None) -> dict:
        """Block until the next action is available."""
        return self._action_queue.get(timeout=timeout)

    def queue_size(self) -> int:
        return self._action_queue.qsize()

    def start(self) -> None:
        """Spawn the producer thread. Idempotent: no-op if already running."""
        if self._producer_thread is not None and self._producer_thread.is_alive():
            return
        self._stop_event.clear()
        self._producer_thread = threading.Thread(target=self._producer_loop, name="PacedBroker-producer", daemon=True)
        self._producer_thread.start()
        rate_msg = f", target_hz={self._target_hz}" if self._target_hz > 0 else " (unpaced)"
        logger.info(f"PacedBroker producer thread started (queue_size={self._action_queue.maxsize}{rate_msg})")

    def stop(self, join_timeout: float = 2.0) -> None:
        """Signal producer to exit and wait for it. Safe to call multiple times."""
        self._stop_event.set()
        # Wake producer if it's waiting on obs_event.
        self._obs_event.set()
        # Wake consumer if it's blocked on pop (so caller can shutdown cleanly).
        # We push a sentinel only if a consumer is likely waiting; safest is to
        # leave it alone — caller should set its own stop flag before joining.
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=join_timeout)
            if self._producer_thread.is_alive():
                logger.warning("PacedBroker producer did not exit within timeout")
            self._producer_thread = None

    # ------------------------------------------------------------------
    # Producer thread body
    # ------------------------------------------------------------------
    def _producer_loop(self) -> None:
        period = 1.0 / self._target_hz if self._target_hz > 0 else 0.0
        next_tick = time.time()
        while not self._stop_event.is_set():
            # Wait until we have at least one obs to feed the inner broker.
            if not self._obs_event.wait(timeout=0.5):
                continue  # periodic check on stop flag
            if self._stop_event.is_set():
                break

            with self._obs_lock:
                obs = self._latest_obs
            if obs is None:
                time.sleep(self._producer_idle_sleep)
                continue

            try:
                action = self._inner.infer(obs)
            except Exception as e:
                logger.error(f"Inner broker.infer raised: {e}; sleeping briefly")
                time.sleep(0.05)
                continue

            # Back-pressure: block until consumer makes room. This naturally
            # paces the producer to consumer rate when the queue is full,
            # and lets it run ahead (filling the buffer) when consumer is fast.
            while not self._stop_event.is_set():
                try:
                    self._action_queue.put(action, timeout=0.1)
                    break
                except queue.Full:
                    continue

            # Rate cap: keep producer at ≤ target_hz. Critical when the inner
            # broker has its own internal queue (RTC): without this, the
            # producer empties RTC's buffer during startup faster than RTC's
            # bg inference can refill, causing repeated "queue empty" warnings.
            if period > 0:
                next_tick += period
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # We're behind schedule (inner.infer + put took longer
                    # than period). Don't try to catch up by bursting — reset
                    # the schedule from now so future ticks are paced again.
                    next_tick = time.time()
