"""Lerobot-style keyboard episode control for BiFlexiv Rizon4 RT inference.

Provides a non-blocking pynput listener that lets the operator delimit
recording episodes with the keyboard, mirroring lerobot's record-loop events:

    Right arrow : start a new episode (IDLE) or end + save it (RUNNING)
    Left arrow  : discard the current episode and re-record it immediately
    Enter       : end + save the episode (is_success=True with --confirm_success)
    Backspace   : end + save the episode (is_success=False with --confirm_success)
    ESC         : end + save the current episode (if any) and exit cleanly

Enter/Backspace always end + save the episode. ``confirm_success`` only
controls whether the frame-level ``observation.is_success`` column is recorded
and backfilled with the confirmed value. ESC follows lerobot semantics: the
in-progress episode is saved (not discarded), then the program exits.

The controller is edge-triggered (press only), so holding a key does not
repeat. All shared state is guarded by a lock because the pynput callback
runs on its own thread while the runtime loop reads/writes from the main
thread.
"""

from __future__ import annotations

from enum import Enum
import threading
from typing import Optional

from lerobot.utils.robot_utils import get_logger

logger = get_logger("KeyboardEpisodeControl")


class EpisodeEndReason(str, Enum):
    """Why the current episode ended; consumed by the recorder."""

    SAVE = "save"  # right arrow: end + save, success unconfirmed
    DISCARD = "discard"  # left arrow: discard buffer, re-record immediately
    SUCCESS = "success"  # enter: end + save, is_success=True
    FAILURE = "failure"  # backspace: end + save, is_success=False
    EXIT = "exit"  # esc: discard buffer and exit


class KeyboardExit(Exception):
    """Raised when ESC is pressed in the IDLE (waiting) state."""


class KeyboardEpisodeController:
    """Owns the pynput listener and exposes edge-triggered episode events.

    State is a simple two-state machine:

    * IDLE: no episode in progress. Right arrow sets a pending start request;
      ESC sets a pending exit request. ``consume_start`` / ``consume_exit``
      are polled by the environment wrapper while it blocks waiting.
    * RUNNING: an episode is being recorded. Any of right/left/enter/
      backspace/esc records a pending end reason. ``_running`` stays True
      while the reason is pending: the runtime's next step reads an
      observation *before* checking ``is_episode_complete``, so flipping to
      IDLE immediately would make ``get_observation`` block in
      ``_wait_for_start`` and swallow the reason. The episode only returns
      to IDLE when the recorder consumes the reason in ``on_episode_end``.

    End reasons are peeked (without consuming) by the environment wrapper's
    ``is_episode_complete`` and consumed exactly once by the recorder in
    ``on_episode_end`` (or drained by the wrapper's reset when no recorder
    is attached).
    """

    def __init__(self, confirm_success: bool = False) -> None:
        self._confirm_success = confirm_success
        self._lock = threading.Lock()
        self._running = False
        self._start_requested = False
        self._exit_requested = False
        self._end_reason: Optional[EpisodeEndReason] = None
        self._listener: Optional[object] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the non-blocking keyboard listener and print the key help."""
        try:
            from pynput import keyboard
        except Exception as e:
            raise RuntimeError(
                "pynput is required for --keyboard_control but could not be "
                f"imported (headless environment?): {e}"
            ) from e

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()
        self._print_help()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _print_help(self) -> None:
        print("Keyboard episode control:")
        print("  Right arrow: start a new episode / end and save the current one")
        print("  Left arrow : discard the current episode and re-record it")
        print("  Enter      : end and save the current episode")
        if self._confirm_success:
            print("               (with --confirm_success: Enter = is_success=True)")
            print("  Backspace  : end episode with is_success=False")
        print("  ESC        : end and save the current episode, then exit")

    # ------------------------------------------------------------------
    # Key callbacks (pynput thread)
    # ------------------------------------------------------------------
    def _on_press(self, key: object) -> None:
        try:
            from pynput import keyboard

            if key == keyboard.Key.right:
                self._handle_right()
            elif key == keyboard.Key.left:
                self._handle_left()
            elif key == keyboard.Key.enter:
                self._handle_enter()
            elif key == keyboard.Key.backspace:
                self._handle_backspace()
            elif key == keyboard.Key.esc:
                self._handle_esc()
        except Exception as e:
            logger.warning(f"Error handling key press: {e}")

    def _handle_right(self) -> None:
        with self._lock:
            if self._running:
                self._end_reason = EpisodeEndReason.SAVE
                logger.info("Right arrow: ending episode (save)")
            else:
                self._start_requested = True
                logger.info("Right arrow: start requested (waiting for next obs tick)")

    def _handle_left(self) -> None:
        with self._lock:
            if self._running:
                self._end_reason = EpisodeEndReason.DISCARD
                logger.info("Left arrow: discarding current episode (re-record)")
            else:
                logger.info("Left arrow: no episode in progress to discard")

    def _handle_enter(self) -> None:
        with self._lock:
            if self._running:
                if self._confirm_success:
                    self._end_reason = EpisodeEndReason.SUCCESS
                    logger.info("Enter: ending episode (is_success=True)")
                else:
                    self._end_reason = EpisodeEndReason.SAVE
                    logger.info("Enter: ending episode (save)")
            else:
                logger.info("Enter: no episode in progress")

    def _handle_backspace(self) -> None:
        with self._lock:
            if self._running:
                if self._confirm_success:
                    self._end_reason = EpisodeEndReason.FAILURE
                    logger.info("Backspace: ending episode (is_success=False)")
                else:
                    self._end_reason = EpisodeEndReason.SAVE
                    logger.info("Backspace: ending episode (save)")
            else:
                logger.info("Backspace: no episode in progress")

    def _handle_esc(self) -> None:
        with self._lock:
            if self._running:
                # Lerobot semantics: ESC ends the in-progress episode and
                # saves it (data is never lost), then requests a clean exit.
                self._end_reason = EpisodeEndReason.SAVE
                logger.info("ESC: ending episode (save) and exiting")
            self._exit_requested = True
            logger.info("ESC: stop recording and exit")

    # ------------------------------------------------------------------
    # Polled by the environment wrapper / recorder (main thread)
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def consume_start(self) -> bool:
        """Return True exactly once when a start was requested; enters RUNNING."""
        with self._lock:
            if self._start_requested:
                self._start_requested = False
                self._running = True
                logger.info("Episode started (recording).")
                return True
            return False

    def consume_exit(self) -> bool:
        """Return True exactly once when ESC was pressed while IDLE."""
        with self._lock:
            if self._exit_requested:
                self._exit_requested = False
                return True
            return False

    def peek_end_reason(self) -> Optional[EpisodeEndReason]:
        """Return the pending end reason without consuming it."""
        with self._lock:
            return self._end_reason

    def consume_end_reason(self) -> Optional[EpisodeEndReason]:
        """Consume the pending end reason exactly once.

        Called by the recorder in ``on_episode_end`` (or drained by the
        environment wrapper's reset when no recorder is attached). Returning
        a reason also transitions back to IDLE (``_running=False``), so the
        next episode blocks in the environment wrapper until a new start key.
        """
        with self._lock:
            reason = self._end_reason
            if reason is not None:
                self._end_reason = None
                self._running = False
            return reason
