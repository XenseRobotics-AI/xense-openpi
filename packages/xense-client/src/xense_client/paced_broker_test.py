"""Offline tests for PacedBroker.

Uses fake BasePolicy implementations so the tests can run without any
robot, camera, or policy server.
"""

import queue
import threading
import time

import pytest

from xense_client import base_policy
from xense_client.paced_broker import PacedBroker


class FakePolicy(base_policy.BasePolicy):
    """Returns the latest obs back as the action, after an optional infer delay."""

    def __init__(self, infer_delay_s: float = 0.0):
        self.infer_delay_s = infer_delay_s
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def infer(self, obs: dict) -> dict:
        if self.infer_delay_s > 0:
            time.sleep(self.infer_delay_s)
        with self.lock:
            self.calls.append(obs)
        return {"actions": obs["step"]}


class FlakyPolicy(base_policy.BasePolicy):
    """Raises on the first N calls, then behaves normally."""

    def __init__(self, fail_first: int = 2):
        self.fail_first = fail_first
        self.call_count = 0

    def infer(self, obs: dict) -> dict:
        self.call_count += 1
        if self.call_count <= self.fail_first:
            raise RuntimeError(f"simulated failure #{self.call_count}")
        return {"actions": obs["step"]}


class FakeInternalQueueBroker(base_policy.BasePolicy):
    """Mimics RTCActionChunkBroker: holds an internal queue refilled by a
    background thread; .infer() pops one. If the queue is empty, returns a
    sentinel and increments an "exhaust" counter — that's what the real
    ActionQueue logs the warning for.
    """

    def __init__(self, initial_size: int = 50, refill_period_s: float = 0.4):
        import collections

        self._q: collections.deque = collections.deque()
        for _ in range(initial_size):
            self._q.append({"actions": 0})
        self._lock = threading.Lock()
        self._refill_period_s = refill_period_s
        self._refill_size = initial_size
        self._exhausted_count = 0
        self._stop = threading.Event()
        self._bg = threading.Thread(target=self._refill_loop, daemon=True)
        self._bg.start()

    def stop(self):
        self._stop.set()

    def _refill_loop(self):
        while not self._stop.is_set():
            time.sleep(self._refill_period_s)
            with self._lock:
                if len(self._q) <= self._refill_size // 2:
                    for _ in range(self._refill_size - len(self._q)):
                        self._q.append({"actions": 0})

    def infer(self, obs: dict) -> dict:
        with self._lock:
            if not self._q:
                self._exhausted_count += 1
                # Return a sentinel rather than block, matching the behavior
                # of RTCActionChunkBroker (which busy-polls and warns).
                return {"actions": -1}
            return self._q.popleft()


# ---------------------------------------------------------------------------
# T1: basic produce-consume
# ---------------------------------------------------------------------------
def test_basic_produce_consume():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=10)
    pb.submit_obs({"step": 0})
    pb.start()
    try:
        action = pb.pop_action(timeout=2.0)
        assert action["actions"] == 0
        # Producer keeps spinning on the same obs; we should be able to pop more.
        for _ in range(3):
            a = pb.pop_action(timeout=2.0)
            assert a["actions"] == 0
    finally:
        pb.stop()


# ---------------------------------------------------------------------------
# T2a: submit_obs overwrites in place — only the latest is retained
# ---------------------------------------------------------------------------
def test_submit_obs_overwrites_slot():
    pb = PacedBroker(inner=FakePolicy(), queue_size=10)
    pb.submit_obs({"step": 1})
    pb.submit_obs({"step": 2})
    pb.submit_obs({"step": 3})
    with pb._obs_lock:
        assert pb._latest_obs == {"step": 3}


# ---------------------------------------------------------------------------
# T2b: overwrites during an in-flight infer drop intermediate obs values
# (single-slot, not a queue of pending obs)
# ---------------------------------------------------------------------------
def test_overwrites_during_infer_skip_intermediates():
    # 50 ms inference window; submit 18 overwrites during it. Only the
    # producer's pickup *before* and *after* that window can land in the queue.
    inner = FakePolicy(infer_delay_s=0.05)
    pb = PacedBroker(inner=inner, queue_size=100)

    pb.submit_obs({"step": 1})
    pb.start()
    # Give the producer time to pick up step=1 and enter infer.
    time.sleep(0.02)
    for s in range(2, 20):
        pb.submit_obs({"step": s})
    # Wait long enough for the producer to: finish step=1's infer (~30 ms left),
    # put it, loop, read slot (now step=19), infer (50 ms), put.
    time.sleep(0.2)
    pb.stop()

    drained = []
    while True:
        try:
            drained.append(pb.pop_action(timeout=0.01)["actions"])
        except queue.Empty:
            break

    # Intermediate values 2..18 were overwritten before the producer could
    # pick them up — they must not appear.
    intermediates = [d for d in drained if 2 <= d <= 18]
    assert intermediates == [], f"intermediates leaked: {drained}"
    # We should have at least seen step=1 (initial pickup) and step=19 (post-overwrite).
    assert 1 in drained, f"missing step=1: {drained}"
    assert 19 in drained, f"missing step=19: {drained}"


# ---------------------------------------------------------------------------
# T3: back-pressure — producer blocks when queue is full
# ---------------------------------------------------------------------------
def test_backpressure_when_queue_full():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=3)
    pb.submit_obs({"step": 42})
    pb.start()
    # Give producer a moment to fill the queue and then block.
    time.sleep(0.2)
    assert pb.queue_size() <= 3, f"queue overflowed: {pb.queue_size()}"
    # Once we drain, the producer should refill quickly.
    for _ in range(3):
        pb.pop_action(timeout=1.0)
    time.sleep(0.1)
    assert pb.queue_size() >= 1
    pb.stop()


# ---------------------------------------------------------------------------
# T4: reset() drains queue and clears latest_obs
# ---------------------------------------------------------------------------
def test_reset_drains_queue():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=10)
    pb.submit_obs({"step": 7})
    pb.start()
    # Wait for some items to accumulate.
    time.sleep(0.2)
    assert pb.queue_size() > 0
    # Caller is expected to stop the producer before reset (per docstring).
    pb.stop()
    pb.reset()
    assert pb.queue_size() == 0
    # After reset, popping should not return any stale action immediately —
    # we'd block on an empty queue. queue.Empty must be raised under timeout.
    with pytest.raises(queue.Empty):
        pb.pop_action(timeout=0.1)


# ---------------------------------------------------------------------------
# T5: stop() exits cleanly even with no consumer popping
# ---------------------------------------------------------------------------
def test_stop_is_clean_with_full_queue():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=2)
    pb.submit_obs({"step": 0})
    pb.start()
    # Let producer fully fill the queue and start blocking on put().
    time.sleep(0.2)
    assert pb.queue_size() == 2
    t0 = time.time()
    pb.stop(join_timeout=2.0)
    elapsed = time.time() - t0
    assert elapsed < 1.5, f"stop took too long: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# T6: producer recovers from inner.infer exceptions
# ---------------------------------------------------------------------------
def test_producer_recovers_from_inner_errors():
    inner = FlakyPolicy(fail_first=3)
    pb = PacedBroker(inner=inner, queue_size=10)
    pb.submit_obs({"step": 99})
    pb.start()
    try:
        # After 3 failures, the 4th call succeeds; we should eventually get an action.
        action = pb.pop_action(timeout=3.0)
        assert action["actions"] == 99
        assert inner.call_count >= 4
    finally:
        pb.stop()


# ---------------------------------------------------------------------------
# T7: start() is idempotent
# ---------------------------------------------------------------------------
def test_start_is_idempotent():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=5)
    pb.submit_obs({"step": 1})
    pb.start()
    first_thread = pb._producer_thread
    pb.start()  # second call should be a no-op
    assert pb._producer_thread is first_thread
    pb.stop()


# ---------------------------------------------------------------------------
# T8: stop() is safe before start()
# ---------------------------------------------------------------------------
def test_stop_before_start_is_safe():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=5)
    pb.stop()  # should not raise


# ---------------------------------------------------------------------------
# T9: target_hz caps the producer rate
# ---------------------------------------------------------------------------
def test_target_hz_caps_producer_rate():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=1000, target_hz=30.0)
    pb.submit_obs({"step": 0})
    pb.start()
    # Run for ~1 s. Producer rate cap is 30 Hz, so call count should be
    # ~30 (±25 % for scheduling jitter).
    time.sleep(1.0)
    pb.stop()
    n_calls = len(inner.calls)
    assert 25 <= n_calls <= 40, f"target_hz=30 expected ~30 calls/s, got {n_calls}"


# ---------------------------------------------------------------------------
# T10: target_hz=0 leaves the producer unpaced (consumer-bound by put queue)
# ---------------------------------------------------------------------------
def test_target_hz_zero_means_unpaced():
    inner = FakePolicy()
    pb = PacedBroker(inner=inner, queue_size=5, target_hz=0.0)
    pb.submit_obs({"step": 0})
    pb.start()
    # No rate cap, but queue.put back-pressure (size=5) caps the producer
    # once the queue is full. Without a consumer, producer fills the queue
    # quickly then blocks. We should see ≥ queue_size calls in a short window.
    time.sleep(0.2)
    pb.stop()
    assert len(inner.calls) >= 5


# ---------------------------------------------------------------------------
# T11: regression — target_hz prevents draining an inner broker's queue
# ---------------------------------------------------------------------------
def test_target_hz_protects_inner_queue_from_drain():
    """Replays the bug observed in session_20260517_202348.log.

    Without rate-capping, PacedBroker drained RTCActionChunkBroker's 50-action
    internal queue in milliseconds, faster than its background inference could
    refill, triggering hundreds of 'Action queue exhausted' warnings.

    Here the FakeInternalQueueBroker stands in for RTC. We verify that with
    target_hz set, the inner queue is NOT exhausted; without it, the inner
    queue gets drained and the exhaust counter climbs.
    """
    # --- bad: no rate cap ---
    inner_bad = FakeInternalQueueBroker(initial_size=50, refill_period_s=0.4)
    pb_bad = PacedBroker(inner=inner_bad, queue_size=50, target_hz=0.0)
    pb_bad.submit_obs({"step": 0})
    pb_bad.start()
    time.sleep(0.5)
    pb_bad.stop()
    inner_bad.stop()
    assert (
        inner_bad._exhausted_count > 0
    ), "unpaced producer should drain inner queue (got no exhausts — test setup wrong?)"

    # --- good: rate cap matches consumer ---
    inner_good = FakeInternalQueueBroker(initial_size=50, refill_period_s=0.4)
    pb_good = PacedBroker(inner=inner_good, queue_size=50, target_hz=30.0)
    pb_good.submit_obs({"step": 0})
    pb_good.start()
    time.sleep(0.5)
    pb_good.stop()
    inner_good.stop()
    assert (
        inner_good._exhausted_count == 0
    ), f"paced producer should not exhaust inner queue, got {inner_good._exhausted_count} exhausts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
