"""Sampler-thread button monitor tests, without robot or VR hardware.

The module is loaded by file path so the examples package (and its hardware
imports) stays out of the test, same as test_rlt_operator_signals.py. The
test env ships upstream lerobot (no xense fork), so the heavy top-level
imports of intervention.py are stubbed before loading and restored right
after — other tests in the same session need the real packages.
"""

# ruff: noqa: N802 - the fake SDK mirrors the XenseVR getter names.

import importlib.util
import logging
import pathlib
import sys
import threading
import time
import types

import pytest

_STUB_NAMES = (
    "lerobot",
    "lerobot.utils",
    "lerobot.utils.robot_utils",
    "xense_client",
    "xense_client.runtime",
    "xense_client.runtime.agents",
    "xense_client.action_chunk_broker",
    "xense_client.runtime.agent",
    "xense_client.runtime.environment",
    "xense_client.runtime.agents.policy_agent",
)


def _install_stubs() -> dict[str, types.ModuleType | None]:
    saved = {name: sys.modules.get(name) for name in _STUB_NAMES}

    def package(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package so submodule imports resolve
        sys.modules[name] = mod
        return mod

    def leaf(parent: types.ModuleType, name: str, **attrs: object) -> None:
        full = f"{parent.__name__}.{name}"
        mod = types.ModuleType(full)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[full] = mod
        parent.__dict__[name] = mod

    root = package("xense_client")
    runtime = package("xense_client.runtime")
    root.runtime = runtime
    agents = package("xense_client.runtime.agents")
    runtime.agents = agents

    lerobot_utils = package("lerobot.utils")
    leaf(lerobot_utils, "robot_utils", get_logger=logging.getLogger)
    lerobot = package("lerobot")
    lerobot.utils = lerobot_utils

    leaf(root, "action_chunk_broker", ActionChunkBroker=object)
    leaf(runtime, "agent", Agent=object)
    leaf(runtime, "environment", Environment=object)
    leaf(agents, "policy_agent", PolicyAgent=object)
    return saved


def _restore(saved: dict[str, types.ModuleType | None]) -> None:
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


_saved = _install_stubs()
try:
    _MODULE_PATH = pathlib.Path(__file__).resolve().parent / "intervention.py"
    _spec = importlib.util.spec_from_file_location("intervention", _MODULE_PATH)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
finally:
    _restore(_saved)
Pico4ButtonMonitor = _module.Pico4ButtonMonitor


class _FakeXrt:
    """Scriptable stand-in for the XenseVR SDK module."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed = {"A": False, "B": False, "X": False, "Y": False}
        self.failing = False

    def get_time_stamp_ns(self) -> int:
        return 1  # Constant is fine: tests finish well under the 0.5 s stale timeout.

    def _get(self, name: str) -> bool:
        with self._lock:
            if self.failing:
                raise OSError("simulated SDK read failure")
            return self._pressed[name]

    def get_A_button(self) -> bool:
        return self._get("A")

    def get_B_button(self) -> bool:
        return self._get("B")

    def get_X_button(self) -> bool:
        return self._get("X")

    def get_Y_button(self) -> bool:
        return self._get("Y")

    def tap(self, name: str, duration_s: float) -> None:
        with self._lock:
            self._pressed[name] = True
        time.sleep(duration_s)
        with self._lock:
            self._pressed[name] = False


class _FakeTeleop:
    def __init__(self, xrt: _FakeXrt) -> None:
        self._xrt = xrt


def _monitor() -> tuple[Pico4ButtonMonitor, _FakeXrt]:
    xrt = _FakeXrt()
    monitor = Pico4ButtonMonitor(_FakeTeleop(xrt))  # type: ignore[arg-type]
    monitor.start()
    # Warmup must exceed the debounce window: ButtonEdges only marks the
    # baseline "released" on an unsuppressed sample, which takes debounce_s
    # after the first observation.
    time.sleep(0.08)
    return monitor, xrt


def test_short_tap_is_caught_between_control_steps():
    monitor, xrt = _monitor()
    try:
        xrt.tap("A", 0.06)  # Far shorter than the old poll-interval + debounce loss window.
        time.sleep(0.05)  # Let the sampler evaluate the release past debounce.
        assert monitor.consume_events() == ["A"]
        assert monitor.consume_events() == []  # One edge per press.
    finally:
        monitor.stop()


def test_glitch_shorter_than_debounce_is_ignored():
    monitor, xrt = _monitor()
    try:
        xrt.tap("B", 0.005)  # A single-sample blip must not fire an edge.
        time.sleep(0.05)
        assert monitor.consume_events() == []
    finally:
        monitor.stop()


def test_sdk_fault_is_latched_and_surfaced_by_poll_then_recovers():
    monitor, xrt = _monitor()
    try:
        xrt.failing = True
        time.sleep(0.02)
        with pytest.raises(RuntimeError, match="unavailable"):
            monitor.poll()

        xrt.failing = False
        time.sleep(0.06)  # Past debounce again so the reset baseline marks "released".
        monitor.poll()  # Transient fault delivered once; healthy again after recovery.
        xrt.tap("Y", 0.06)
        time.sleep(0.05)
        assert monitor.consume_events() == ["Y"]
    finally:
        monitor.stop()


def test_disconnected_teleop_latches_abort():
    monitor, _xrt = _monitor()
    try:
        monitor._teleop._xrt = None
        time.sleep(0.02)
        with pytest.raises(RuntimeError, match="disconnected"):
            monitor.poll()
    finally:
        monitor.stop()
