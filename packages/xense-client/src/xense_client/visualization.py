"""Robot-independent lifecycle for optional XMV visualization.

Only a robot's adapter knows its action layout, SDK and URDF. XMV and Rerun
are imported when visualization starts, never by the inference brokers.
"""

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

from xense_client.action_stream import ACTION_REF_KEY
from xense_client.action_stream import ActionStream
from xense_client.action_stream import Subscription

if TYPE_CHECKING:
    from xense_motion_viewer import RobotDescription
    from xense_motion_viewer import RobotState


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = False
    display_hz: float = 30.0
    spawn_viewer: bool = True
    viewer_port: int = 9877
    save_path: str | None = None
    queue_capacity: int = 8

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.spawn_viewer, bool):
            raise ValueError("visualization enabled/spawn_viewer must be booleans")
        if not np.isfinite(self.display_hz) or self.display_hz <= 0:
            raise ValueError("visualization display_hz must be finite and positive")
        if type(self.queue_capacity) is not int or self.queue_capacity < 1:
            raise ValueError("visualization queue_capacity must be positive")
        if type(self.viewer_port) is not int or not 1 <= self.viewer_port <= 65535:
            raise ValueError("visualization viewer_port must be an integer in 1..65535")


def load_visualization_config(recipe, *, enabled=None):
    """Decode the recipe's top-level visualization block, separate from robot."""
    import yaml

    raw = yaml.safe_load(Path(recipe).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Recipe must be a mapping")
    # An explicit CLI off must not make optional display settings a prerequisite
    # for robot inference. Validate the full block only when it is enabled.
    if enabled is False:
        return VisualizationConfig()
    block = raw.get("visualization", {})
    if not isinstance(block, dict):
        raise ValueError("Recipe visualization must be a mapping")
    if enabled is not None:
        block = {**block, "enabled": enabled}
    if block.get("enabled", False) is False:
        return VisualizationConfig()
    return VisualizationConfig(**block)


@dataclass(frozen=True)
class StateSample:
    state: "RobotState | None"
    sample_id: str | None = None
    status: str = ""
    reset_history: bool = False


class RobotVisualizationAdapter(Protocol):
    """Implement this private adapter for a new robot; policy state is unrelated."""

    def describe(self) -> "RobotDescription": ...

    def decode_actions(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        """Return group -> [steps,4,4] world-frame TCP reference poses."""
        ...

    def read_state(self, received_at: float) -> StateSample:
        """Read a cached SDK state, without camera reads or robot commands."""
        ...


class VisualizationSession:
    """Own every optional subscription, state sampler and display resource.

    set_enabled(False) unregisters/drains subscriptions and joins the sampler
    before closing XMV. It leaves inference, action queues and SDK connections
    alone. Starting again uses a new display recording and waits for a new
    broker plan; no stale reference is replayed.
    """

    def __init__(self, config, adapter_factory, *, display_factory=None):
        self.config, self._adapter_factory = config, adapter_factory
        self._display_factory = display_factory
        self.stream = ActionStream()
        self.stream.close()
        self._state_subscriptions = []
        self._stop = threading.Event()
        self._thread = self._display = self._adapter = self._plans = None
        self.errors = 0

    @property
    def enabled(self):
        return self.stream.enabled

    def status(self):
        return self.stream.status()

    def has_new_chunk(self, since_sequence):
        return self.stream.has_new_chunk(since_sequence)

    def subscribe_chunks(self, *, capacity=4):
        return self.stream.subscribe_chunks(capacity=capacity)

    def subscribe_plans(self, *, capacity=4):
        return self.stream.subscribe_plans(capacity=capacity)

    def subscribe_robot_state(self, *, capacity=4):
        subscription = Subscription[StateSample](capacity)
        if self.enabled:
            self._state_subscriptions.append(subscription)
        else:
            subscription.close()
        return subscription

    def _make_display(self, description):
        if self._display_factory is not None:
            return self._display_factory(description)
        import rerun as rr
        from xense_motion_viewer.live import LiveMotionDisplay

        recording = rr.RecordingStream(
            "xense-motion-viewer-live", batcher_config=rr.ChunkBatcherConfig(flush_tick=1 / self.config.display_hz)
        )
        try:
            sinks = []
            if self.config.save_path:
                path = Path(self.config.save_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                sinks.append(rr.FileSink(path))
            if self.config.spawn_viewer:
                recording.spawn(port=self.config.viewer_port, connect=False, memory_limit="512MiB")
                sinks.append(rr.GrpcSink(f"rerun+http://127.0.0.1:{self.config.viewer_port}/proxy"))
            if sinks:
                recording.set_sinks(*sinks)
            return LiveMotionDisplay(recording, description, capacity=self.config.queue_capacity)
        except Exception:
            recording.disconnect()
            raise

    def set_enabled(self, enabled: bool):
        if not enabled:
            self.close()
            return
        if self.enabled:
            return
        if self._thread is not None or self._display is not None:
            # A timed-out SDK read must finish before another sampler starts.
            self.close()
        # Adapter/geometry initialization happens before starting runtime work.
        self._adapter = self._adapter_factory()
        self._display = self._make_display(self._adapter.describe())
        self.stream.reopen()
        self._plans = self.stream.subscribe_plans()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, name="xmv-sdk-state", daemon=True)
        self._thread.start()

    def _sample_loop(self):
        from xense_motion_viewer.live import FutureTrajectory
        from xense_motion_viewer.live import LiveFrame

        adapter, display, plans = self._adapter, self._display, self._plans
        started = time.monotonic()
        period = 1 / self.config.display_hz
        generation = None
        futures = {}
        reset_pending = True
        while not self._stop.is_set():
            tick = time.monotonic()
            status = ""
            plan = plans.latest()
            if plan is not None:
                if plan.generation != generation:
                    generation, reset_pending = plan.generation, True
                try:
                    decoded = adapter.decode_actions(plan.actions) if plan.refs else {}
                    ids = tuple(":".join(map(str, ref)) for ref in plan.refs)
                    futures = {group: FutureTrajectory(ids, poses) for group, poses in decoded.items()}
                except Exception as exc:
                    self.errors += 1
                    futures, status = {}, f"Invalid action reference: {exc}"
            now = time.monotonic() - started
            try:
                sample = adapter.read_state(now)
            except Exception as exc:
                self.errors += 1
                sample = StateSample(None, status=f"SDK state unavailable: {exc}")
            for subscription in tuple(self._state_subscriptions):
                subscription.publish(sample)
            reset_pending = reset_pending or sample.reset_history
            frame = LiveFrame(
                now,
                futures,
                sample.state,
                sample.sample_id,
                reset_pending,
                status + sample.status,
            )
            if display.publish(frame):
                reset_pending = False
            self._stop.wait(max(0, period - (time.monotonic() - tick)))

    def close(self):
        self.stream.close()
        self._stop.set()
        for subscription in self._state_subscriptions:
            subscription.close()
        self._state_subscriptions.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        reader_alive = self._thread is not None and self._thread.is_alive()
        if self._display is not None:
            if not self._display.close():
                raise RuntimeError("XMV display did not stop")
            self._display = None
        if reader_alive:
            raise RuntimeError("Visualization SDK reader did not stop; display closed, re-enable waits for reader exit")
        self._thread = None
        self._adapter = self._plans = None


class ObservedEnvironment:
    """Enabled-only wrapper. It reports dispatch after successful apply_action.

    The private display tag is removed before delegating to the environment;
    inference observations and robot action dictionaries keep their old schema.
    """

    def __init__(self, environment, stream):
        self._environment, self._stream = environment, stream

    def reset(self):
        return self._environment.reset()

    def is_episode_complete(self):
        return self._environment.is_episode_complete()

    def get_observation(self):
        return self._environment.get_observation()

    def apply_action(self, action):
        ref = action.get(ACTION_REF_KEY)
        clean = {key: value for key, value in action.items() if key != ACTION_REF_KEY} if ref is not None else action
        result = self._environment.apply_action(clean)
        if self._stream.enabled:
            self._stream.dispatched(ref)
        return result
