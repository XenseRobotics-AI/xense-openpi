"""Keyboard-gated episode runtime for the recording example."""

from __future__ import annotations

import os
import queue
import select
import sys
import termios
import threading
import time
import tty

from lerobot.utils.robot_utils import get_logger
from xense_client.runtime import runtime as _runtime

logger = get_logger("KeyboardEpisodeControl")


def _extract_right_arrow_events(buffer: bytes) -> tuple[bytes, int]:
    """Extract right-arrow escape sequences and keep partial input."""
    events = 0
    while buffer:
        matches = [(buffer.find(seq), len(seq)) for seq in (b"\x1b[C", b"\x1bOC")]
        matches = [(index, length) for index, length in matches if index >= 0]
        if matches:
            index, length = min(matches)
            buffer = buffer[index + length :]
            events += 1
            continue
        escape_index = buffer.find(b"\x1b")
        if escape_index < 0:
            return b"", events
        suffix = buffer[escape_index:]
        if len(suffix) < 3:
            return suffix, events
        buffer = suffix[1:]
    return b"", events


class RightArrowListener:
    """Read Right Arrow presses from the controlling terminal."""

    def __init__(self, *, debounce_s: float = 0.2) -> None:
        self._debounce_s = debounce_s
        self._events: queue.Queue[None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_terminal_attrs: list | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if not sys.stdin.isatty():
            raise RuntimeError(
                "--keyboard_episode_control requires a controlling terminal; "
                "run it from the terminal that should receive Right Arrow presses."
            )
        self._fd = sys.stdin.fileno()
        self._old_terminal_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="right-arrow-listener", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._fd is not None
        buffer = b""
        last_event_time = 0.0
        while not self._stop_event.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                data = os.read(self._fd, 64)
            except (OSError, ValueError):
                break
            if not data:
                break
            buffer += data
            buffer, event_count = _extract_right_arrow_events(buffer)
            now = time.monotonic()
            for _ in range(event_count):
                if now - last_event_time >= self._debounce_s:
                    self._events.put(None)
                    last_event_time = now

    def consume_press(self) -> bool:
        try:
            self._events.get_nowait()
            return True
        except queue.Empty:
            return False

    def wait_for_press(self, stop_event: threading.Event | None = None) -> bool:
        while not self._stop_event.is_set() and not (stop_event and stop_event.is_set()):
            try:
                self._events.get(timeout=0.1)
                return True
            except queue.Empty:
                continue
        return False

    def discard_pending(self) -> None:
        while self.consume_press():
            pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None and self._old_terminal_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_terminal_attrs)
            except (OSError, termios.error):
                pass
        self._fd = None
        self._old_terminal_attrs = None

    def __enter__(self) -> RightArrowListener:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


class KeyboardEpisodeRuntime(_runtime.Runtime):
    """Synchronous Runtime with reset-and-next gated by Right Arrow."""

    def __init__(self, *, keyboard: RightArrowListener, **kwargs) -> None:
        super().__init__(**kwargs)
        self._keyboard = keyboard
        self._stop_event = threading.Event()
        self._skip_next_environment_reset = False

    def request_stop(self) -> None:
        self._stop_event.set()
        super().request_stop()

    def _step(self) -> None:
        # Check before obtaining an observation/action, so the terminating key
        # does not send one additional command to the robot.
        if self._keyboard.consume_press():
            self.mark_episode_complete()
            return
        if self._stop_event.is_set() or self._stop_requested:
            self.mark_episode_complete()
            return
        super()._step()

    def _run_episode(self) -> None:
        if self._skip_next_environment_reset:
            self._skip_next_environment_reset = False
            logger.info("Starting next episode after manual reset")
        else:
            self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        logger.info("Pre-warming agent (JIT compilation)...")
        warmup_obs = self._environment.get_observation()
        self._agent.warmup(warmup_obs)
        logger.info("Agent warmed up. Starting control loop.")

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()
        while self._in_episode:
            self._step()
            self._episode_steps += 1
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        logger.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def run(self) -> None:
        self._stop_requested = False
        self._stop_event.clear()
        try:
            for episode_index in range(self._num_episodes):
                if self._stop_requested:
                    break
                self._run_episode()
                if episode_index + 1 >= self._num_episodes or self._stop_requested:
                    break

                logger.info("Entering reset mode after episode completion...")
                self._environment.reset()
                self._keyboard.discard_pending()
                logger.info("Reset complete. Press Right Arrow to start the next inference.")
                if not self._keyboard.wait_for_press(stop_event=self._stop_event):
                    break
                self._skip_next_environment_reset = True
        finally:
            self._environment.reset()

