"""Pure input-state tests, without importing robot or VR SDK packages."""

# ruff: noqa: FBT003
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("button_edges", Path(__file__).parent / "button_edges.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_held_at_connect_requires_release_then_new_press():
    edges = module.ButtonEdges()
    assert not edges.update("A", True, 0)
    assert not edges.update("A", True, 1)
    assert not edges.update("A", False, 2)
    assert not edges.update("A", False, 2.1)
    assert not edges.update("A", True, 3)
    assert edges.update("A", True, 3.1)
    assert not edges.update("A", True, 4)


def test_glitch_and_reconnect_do_not_synthesize_press():
    edges = module.ButtonEdges()
    edges.update("A", False, 0)
    edges.update("A", False, 0.1)
    assert not edges.update("A", True, 1)
    assert not edges.update("A", False, 1.01)
    assert not edges.update("A", False, 2)
    edges.reset()
    assert not edges.update("A", True, 3)
    assert not edges.update("A", True, 4)


def test_packet_watchdog_detects_stale_data_and_clock_restart():
    health = module.StreamHealth(timeout_s=0.5)
    health.update(100, now=0)
    health.update(100, now=0.4)
    health.update(101, now=0.45)
    with pytest.raises(RuntimeError, match="stale"):
        health.update(101, now=1)
    with pytest.raises(RuntimeError, match="restarted"):
        health.update(1, now=1.1)


def test_packet_watchdog_uses_progress_not_wall_clock_offset():
    health = module.StreamHealth()
    health.update(1, now=1000)
    health.update(2, now=2000)  # a long inference gap, but fresh device data
    with pytest.raises(RuntimeError, match="no valid"):
        health.update(0, now=2001)
