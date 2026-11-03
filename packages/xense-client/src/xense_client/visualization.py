"""Robot-independent lifecycle for optional XMV visualization.

Only a robot's adapter knows its action layout, SDK and URDF. XMV and Rerun
are imported when visualization starts, never by the inference brokers.
"""

from collections import deque
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
    images_enabled: bool = True
    image_hz: float = 10.0
    image_max_edge: int = 640
    jpeg_quality: int = 85
    image_stale_after: float = 1.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.spawn_viewer, bool):
            raise ValueError("visualization enabled/spawn_viewer must be booleans")
        if not np.isfinite(self.display_hz) or self.display_hz <= 0:
            raise ValueError("visualization display_hz must be finite and positive")
        if type(self.queue_capacity) is not int or self.queue_capacity < 1:
            raise ValueError("visualization queue_capacity must be positive")
        if type(self.viewer_port) is not int or not 1 <= self.viewer_port <= 65535:
            raise ValueError("visualization viewer_port must be an integer in 1..65535")
        if not isinstance(self.images_enabled, bool):
            raise ValueError("images_enabled must be a boolean")
        for key in ("image_hz", "image_stale_after"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if type(self.image_max_edge) is not int or self.image_max_edge < 1:
            raise ValueError("image_max_edge must be positive")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in 1..100")


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

    def __init__(
        self,
        config,
        adapter_factory,
        *,
        display_factory=None,
        image_streams=(),
        instruction_metadata=None,
        dry_run=False,
    ):
        self.config, self._adapter_factory = config, adapter_factory
        self._display_factory = display_factory
        self.stream = ActionStream()
        self.stream.close()
        self._state_subscriptions = []
        self._stop = threading.Event()
        self._thread = self._display = self._adapter = self._plans = None
        self.errors = 0
        self._image_streams = tuple(image_streams)
        self._clock_origin = None
        self._generation = self._episode = self._event_sequence = 0
        self._events = deque(maxlen=256)
        self._task_lock = threading.Lock()
        self._running = self._had_error = False
        self._dry_run = dry_run
        self._task_status = "Waiting for runtime"
        self._pending_task = None
        metadata = instruction_metadata if isinstance(instruction_metadata, dict) else {}
        valid = metadata.get("applies_when") == "request_has_no_prompt" and isinstance(metadata.get("text"), str)
        self._instruction = metadata["text"] if valid else None
        self._instruction_source = str(metadata.get("source", "")) if valid else ""

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
        from xense_motion_viewer.live_process import ImageStream
        from xense_motion_viewer.live_process import LiveDisplayProcess
        from xense_motion_viewer.live_process import ProcessDisplayConfig

        config = ProcessDisplayConfig(
            display_hz=self.config.display_hz,
            image_hz=self.config.image_hz,
            image_max_edge=self.config.image_max_edge,
            jpeg_quality=self.config.jpeg_quality,
            stale_after=self.config.image_stale_after,
            capacity=self.config.queue_capacity,
            spawn_viewer=self.config.spawn_viewer,
            viewer_port=self.config.viewer_port,
            save_path=self.config.save_path,
        )
        streams = tuple(ImageStream(*stream) for stream in self._image_streams) if self.config.images_enabled else ()
        return LiveDisplayProcess(description, config=config, streams=streams, clock_origin=self._clock_origin)

    def prepare(self):
        """Start the isolated renderer before connecting hardware.

        This does not sample SDK state. Bind SDK readers after connection.
        """
        if self._display is not None:
            return
        if self._thread is not None:
            self.close()
        self._clock_origin = time.monotonic()
        self._events.clear()
        self._generation = self._episode = self._event_sequence = 0
        self._running = self._had_error = False
        self._pending_task = None
        self._adapter = self._adapter_factory()
        self._display = self._make_display(self._adapter.describe())

    def set_enabled(self, enabled: bool):
        if not enabled:
            self.close()
            return
        if self.enabled:
            return
        if self._thread is not None:
            # A timed-out SDK read must finish before another sampler starts.
            self.close()
        self.prepare()
        self.stream.reopen()
        self._plans = self.stream.subscribe_plans()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, name="xmv-sdk-state", daemon=True)
        self._thread.start()
        self.report_event("Display session connected", "Waiting for runtime")

    def publish_observation(self, raw_observation):
        """Borrow image references only. No copies, camera reads, encoding or IPC."""
        if not self.enabled or not self.config.images_enabled or self._display is None:
            return
        try:
            self._display.publish_images(
                time.monotonic() - self._clock_origin, raw_observation, generation=self._generation
            )
        except Exception:
            self.errors += 1

    def report_event(self, text, status, *, error=False):
        if self._display is None:
            return
        try:
            from xense_motion_viewer.live_process import TaskEvent
            from xense_motion_viewer.live_process import TaskSnapshot

            with self._task_lock:
                at = time.monotonic() - self._clock_origin
                self._event_sequence += 1
                self._had_error = self._had_error or error
                self._task_status = status
                self._events.append(TaskEvent(self._event_sequence, at, text))
                snapshot = TaskSnapshot(
                    at,
                    self._generation,
                    self._episode,
                    self._instruction,
                    self._instruction_source,
                    status,
                    tuple(self._events),
                    max(0, self._event_sequence - 256),
                )
                self._pending_task = snapshot
            self._flush_task()
        except Exception:
            self.errors += 1

    def _flush_task(self):
        snapshot = self._pending_task
        if snapshot is not None and self._display is not None:
            try:
                if self._display.update_task(snapshot):
                    with self._task_lock:
                        if self._pending_task is snapshot:
                            self._pending_task = None
            except Exception:
                self.errors += 1

    def on_episode_start(self):
        self._generation += 1
        self._episode += 1
        self._running = False
        self.report_event(f"Episode {self._episode} started", "Warming up")

    def on_step(self, observation, action):
        if not self._running:
            self._running = True
            status = "Dry run Running (no robot actions sent)" if self._dry_run else "Running"
            self.report_event(f"Episode {self._episode} executing steps", status)

    def on_episode_end(self):
        self.report_event(f"Episode {self._episode} ended (not proof of task success)", "Episode ended")

    def finish(self, outcome):
        if self._had_error:
            outcome += " (errors occurred; see runtime events)"
        self.report_event(outcome, outcome)

    def _sample_loop(self):
        from xense_motion_viewer.live import FutureTrajectory
        from xense_motion_viewer.live import LiveFrame

        adapter, display, plans = self._adapter, self._display, self._plans
        period = 1 / self.config.display_hz
        generation = None
        futures = {}
        reset_pending = True
        while not self._stop.is_set():
            tick = time.monotonic()
            display_generation = self._generation
            self._flush_task()
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
            now = time.monotonic() - self._clock_origin
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
            if display_generation != self._generation:
                continue
            if display.publish(frame, generation=display_generation):
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
            self._flush_task()
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

    def __init__(self, environment, stream, *, events=None):
        self._environment, self._stream = environment, stream
        self._events = events

    def _report(self, text, status, *, error=False):
        if self._events is not None:
            self._events.report_event(text, status, error=error)

    def reset(self):
        self._report("Reset started", "Resetting")
        try:
            result = self._environment.reset()
        except Exception as exc:
            self._report(f"Reset failed: {exc}", "Reset failed", error=True)
            raise
        self._report("Reset completed", "Reset complete")
        return result

    def is_episode_complete(self):
        return self._environment.is_episode_complete()

    def get_observation(self):
        try:
            return self._environment.get_observation()
        except Exception as exc:
            self._report(f"Observation error: {exc}", "Observation error", error=True)
            raise

    def apply_action(self, action):
        ref = action.get(ACTION_REF_KEY)
        clean = {key: value for key, value in action.items() if key != ACTION_REF_KEY} if ref is not None else action
        try:
            result = self._environment.apply_action(clean)
        except Exception as exc:
            self._report(f"Action submission error: {exc}", "Action submission error", error=True)
            raise
        if self._stream.enabled:
            self._stream.dispatched(ref)
        return result
