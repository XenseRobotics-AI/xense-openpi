"""Optional read-only action subscriptions, independent of robot/action layout.

No subscribers run on inference/action threads. Producers only copy small
snapshots into bounded queues. With no ActionStream attached, brokers retain
their original return values and queue operations.
"""

from collections import OrderedDict
import contextlib
from dataclasses import dataclass
import queue
import threading
import time
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")
ActionRef = tuple[int, int, int]  # episode generation, chunk sequence, source row
ACTION_REF_KEY = "visualization_ref"
_CLOSED = object()


class SubscriptionClosedError(RuntimeError):
    """A closed subscription cannot supply another event."""


def _frozen(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class Subscription(Generic[T]):
    """Bounded latest-data queue. There are no synchronous user callbacks."""

    def __init__(self, capacity=4):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Subscription capacity must be a positive integer")
        self._queue: queue.Queue[T] = queue.Queue(maxsize=capacity)
        self.closed = False
        self.dropped = 0

    def publish(self, value: T):
        if self.closed:
            return
        try:
            self._queue.put_nowait(value)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            self.dropped += 1
            try:
                self._queue.put_nowait(value)
            except queue.Full:
                self.dropped += 1

    def get(self, timeout=None) -> T:
        if self.closed:
            raise SubscriptionClosedError("Subscription is closed")
        value = self._queue.get(timeout=timeout)
        if value is _CLOSED or self.closed:
            # Wake any other consumer already waiting on this subscription.
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(_CLOSED)
            raise SubscriptionClosedError("Subscription is closed")
        return value

    def latest(self) -> T | None:
        if self.closed:
            return None
        value = None
        while True:
            try:
                item = self._queue.get_nowait()
                if item is _CLOSED:
                    return None
                value = item
            except queue.Empty:
                return value

    def close(self):
        self.closed = True
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(_CLOSED)


@dataclass(frozen=True)
class ChunkArrival:
    sequence: int
    generation: int
    actions: np.ndarray
    usable_steps: int
    received_at: float


@dataclass(frozen=True)
class ActionPlan:
    """Complete remaining reference, including actions buffered by PacedBroker."""

    revision: int
    generation: int
    refs: tuple[ActionRef, ...]
    actions: np.ndarray


@dataclass(frozen=True)
class ActionStreamStatus:
    enabled: bool
    latest_chunk_sequence: int
    generation: int
    errors: int


class ActionStream:
    """A display-only side channel; it cannot consume or mutate control queues.

    Pending rows follow broker queue replacement. Rows already popped into a
    runtime/PacedBroker are retained until that runtime reports dispatch. A
    dropped display update is repaired by the next complete plan snapshot.
    """

    def __init__(self):
        self.enabled = True
        self._lock = threading.RLock()
        self._chunks: list[Subscription] = []
        self._plans: list[Subscription] = []
        self._pending = OrderedDict()
        self._buffered = OrderedDict()
        self._chunk_sequence = self._generation = self._revision = self._errors = 0
        self._action_dim = 0

    def subscribe_chunks(self, *, capacity=4):
        with self._lock:
            subscription = Subscription[ChunkArrival](capacity)
            if self.enabled:
                self._chunks.append(subscription)
            else:
                subscription.close()
            return subscription

    def subscribe_plans(self, *, capacity=4):
        with self._lock:
            subscription = Subscription[ActionPlan](capacity)
            if self.enabled:
                self._plans.append(subscription)
            else:
                subscription.close()
            return subscription

    def status(self):
        with self._lock:
            return ActionStreamStatus(self.enabled, self._chunk_sequence, self._generation, self._errors)

    def has_new_chunk(self, since_sequence: int):
        with self._lock:
            return self.enabled and self._chunk_sequence > since_sequence

    def _publish_plan(self):
        self._revision += 1
        rows = OrderedDict(self._buffered)
        rows.update((ref, row) for ref, row in self._pending.items() if ref not in rows)
        actions = np.stack(tuple(rows.values())) if rows else np.empty((0, self._action_dim))
        plan = ActionPlan(self._revision, self._generation, tuple(rows), _frozen(actions))
        for subscription in self._plans:
            subscription.publish(plan)

    def chunk_received(self, actions, *, usable_steps=None):
        """Return source-row IDs; errors only increment display error counters."""
        if not self.enabled:
            return None
        try:
            with self._lock:
                if not self.enabled:
                    return None
                values = _frozen(actions)
                if values.ndim != 2:
                    raise ValueError("Action chunks must have shape [steps, action_dim]")
                self._action_dim = values.shape[1]
                count = len(values) if usable_steps is None else min(usable_steps, len(values))
                self._chunk_sequence += 1
                arrival = ChunkArrival(self._chunk_sequence, self._generation, values, count, time.monotonic())
                for subscription in self._chunks:
                    subscription.publish(arrival)
                return tuple((self._generation, self._chunk_sequence, index) for index in range(count))
        except Exception:
            self._errors += 1
            return None

    def plan_updated(self, actions, refs):
        if not self.enabled or refs is None:
            return
        try:
            with self._lock:
                if not self.enabled:
                    return
                self._pending = OrderedDict(zip(refs, _frozen(actions), strict=True))
                self._publish_plan()
        except Exception:
            self._errors += 1

    def popped(self, ref, action):
        if not self.enabled or ref is None:
            return
        try:
            with self._lock:
                if not self.enabled or ref[0] != self._generation:
                    return
                self._pending.pop(ref, None)
                self._buffered[ref] = _frozen(action)
                self._publish_plan()
        except Exception:
            self._errors += 1

    def dispatched(self, ref):
        if not self.enabled or ref is None:
            return
        try:
            with self._lock:
                if not self.enabled:
                    return
                self._buffered.pop(ref, None)
                self._pending.pop(ref, None)
                self._publish_plan()
        except Exception:
            self._errors += 1

    def reset(self):
        if not self.enabled:
            return
        with self._lock:
            self._generation += 1
            self._pending.clear()
            self._buffered.clear()
            self._publish_plan()

    def close(self):
        with self._lock:
            self.enabled = False
            for subscription in (*self._chunks, *self._plans):
                subscription.close()
            self._chunks.clear()
            self._plans.clear()
            self._pending.clear()
            self._buffered.clear()

    def reopen(self):
        with self._lock:
            self.enabled = True
            self.reset()
