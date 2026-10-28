"""Optional lifecycle must not add work when disabled or leak workers on stop."""

import subprocess
import sys
import threading

import numpy as np
import pytest

from xense_client.visualization import StateSample
from xense_client.visualization import VisualizationConfig
from xense_client.visualization import VisualizationSession
from xense_client.visualization import load_visualization_config


def test_disabled_does_not_import_optional_packages_or_construct_adapter():
    code = """
import sys, threading
from xense_client.visualization import VisualizationConfig, VisualizationSession
before = set(threading.enumerate())
def forbidden():
    raise AssertionError('disabled adapter was constructed')
session = VisualizationSession(VisualizationConfig(enabled=False), forbidden)
session.set_enabled(False)
assert set(threading.enumerate()) == before
assert 'rerun' not in sys.modules
assert 'xense_motion_viewer' not in sys.modules
assert 'libpyelite' not in sys.modules
assert not session.enabled
assert session.subscribe_chunks().closed
assert session.subscribe_plans().closed
assert session.subscribe_robot_state().closed
assert not session.has_new_chunk(0)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_recipe_visualization_is_separate_and_cli_override_wins(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("robot:\n  type: anything\nvisualization:\n  enabled: true\n  display_hz: 12\n")
    assert load_visualization_config(path).enabled
    assert load_visualization_config(path).display_hz == 12
    assert not load_visualization_config(path, enabled=False).enabled
    path.write_text("robot:\n  type: anything\n")
    assert not load_visualization_config(path).enabled


def test_disabled_recipe_ignores_unused_display_settings(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("robot: {}\nvisualization: {enabled: false, display_hz: 0, misspelled_option: 1}\n")
    assert not load_visualization_config(path).enabled
    path.write_text("robot: {}\nvisualization: {enabled: true, display_hz: 0}\n")
    assert not load_visualization_config(path, enabled=False).enabled
    with pytest.raises(ValueError, match="display_hz"):
        load_visualization_config(path)


def test_blocked_sdk_reader_still_closes_display_and_cannot_overlap_on_restart():
    pytest.importorskip("xense_motion_viewer.live")
    entered, release = threading.Event(), threading.Event()
    closed, factories = [], []

    class Adapter:
        def describe(self):
            return object()

        def read_state(self, received_at):
            entered.set()
            release.wait(10)
            return StateSample(None)

    class Display:
        def publish(self, frame):
            return not closed

        def close(self):
            closed.append(True)
            return True

    def make_adapter():
        factories.append(True)
        return Adapter()

    session = VisualizationSession(VisualizationConfig(), make_adapter, display_factory=lambda _: Display())
    session.set_enabled(True)
    assert entered.wait(2)
    try:
        with pytest.raises(RuntimeError, match="SDK reader did not stop"):
            session.close()
        assert closed
        assert session._display is None
        assert not session.enabled
        with pytest.raises(RuntimeError, match="SDK reader did not stop"):
            session.set_enabled(True)
        assert len(factories) == 1
    finally:
        release.set()
        session.close()
    assert session._thread is None


def test_switch_closes_every_subscription_and_sampler():
    pytest.importorskip("xense_motion_viewer.live")
    from xense_motion_viewer import RobotState

    frames, reads, factories = [], [], []
    displayed = threading.Event()

    class Adapter:
        def describe(self):
            return object()

        def decode_actions(self, actions):
            return {"generic-tool": np.tile(np.eye(4), (len(actions), 1, 1))}

        def read_state(self, received_at):
            reads.append(received_at)
            return StateSample(RobotState(received_at, {}), str(len(reads)))

    class Display:
        def publish(self, frame):
            frames.append(frame)
            displayed.set()
            return True

        def close(self):
            return True

    def make_adapter():
        factories.append(True)
        return Adapter()

    session = VisualizationSession(
        VisualizationConfig(display_hz=100), make_adapter, display_factory=lambda description: Display()
    )
    assert not factories
    session.set_enabled(True)
    chunks, plans, states = session.subscribe_chunks(), session.subscribe_plans(), session.subscribe_robot_state()
    refs = session.stream.chunk_received(np.ones((3, 7)))
    session.stream.plan_updated(np.ones((3, 7)), refs)
    sample = states.get(timeout=2)
    assert sample.state is not None
    assert displayed.wait(2)
    session.set_enabled(False)
    assert chunks.closed
    assert plans.closed
    assert states.closed
    assert session._thread is None
    assert session._display is None
    count = len(reads)
    assert not session.enabled
    assert session.stream.chunk_received(np.ones((2, 7))) is None
    assert len(reads) == count
    # Re-enable starts fresh resources and clears old plans, without resetting
    # a broker or issuing a robot command.
    displayed.clear()
    session.set_enabled(True)
    assert displayed.wait(2)
    session.close()
    assert len(factories) == 2
    assert frames[-1].futures == {}
