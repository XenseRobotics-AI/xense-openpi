"""Check the shipped Elite recipe against the installed driver and native SDK.

No robot, camera or policy connection is opened by these tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt", reason="Elite driver not installed")

import examples.bi_elite_cs66_rt.recipe as _recipe


@pytest.mark.parametrize("name", _recipe.available_recipes())
def test_recipe_initializes_native_collision_geometry(name):
    assembly_module = pytest.importorskip("libpyelite.assembly")
    collision_module = pytest.importorskip("libpyelite.collision")
    config = _recipe.load_robot_config(name, enable_tactile=False)
    assert config.native_collision_profile == "auto"

    assembly = assembly_module.Assembly(config.native_assembly)
    tools = [np.asarray(getattr(config, f"{side}_tool_transform")).reshape(4, 4) for side in ("left", "right")]
    station = collision_module.StationLayer(assembly, config.native_collision_profile, tools)
    # The native SDK validates that calibration covers the movable finger joints,
    # sensor range and URDF limits; YAML decoding alone cannot check this.
    station.configure_grippers(config.native_collision_gripper_calibration, config.native_collision_gripper_max_age_s)
    assert station.path.is_file()


@pytest.mark.parametrize("name", _recipe.available_recipes())
def test_inference_entry_loads_current_recipe_before_connecting(name, monkeypatch):
    import examples.bi_elite_cs66_rt.main as entrypoint

    class ReachedPolicyConnectionError(Exception):
        pass

    def stop_before_connection(**kwargs):
        raise ReachedPolicyConnectionError

    monkeypatch.setattr(entrypoint._websocket_client_policy, "WebsocketClientPolicy", stop_before_connection)
    # Exercise main's real config loading and CLI overrides. A retired override
    # such as log_manipulability would fail before reaching this sentinel.
    with pytest.raises(ReachedPolicyConnectionError):
        entrypoint.main(entrypoint.Args(robot_recipe=name))


@pytest.mark.parametrize("connect_fails", [False, True])
def test_visualization_is_prepared_before_robot_connect_and_sampling_follows_it(monkeypatch, connect_fails):
    pytest.importorskip("xense_motion_viewer")
    import xense_client.visualization as lifecycle

    import examples.bi_elite_cs66_rt.main as entrypoint
    import examples.bi_elite_cs66_rt.visualization as elite_display

    events = []

    class ConnectionFailedError(Exception):
        pass

    class Environment:
        def __init__(self, config, **kwargs):
            assert not kwargs["setup_robot"]
            self._env = SimpleNamespace(robot=object(), setup_robot=self.connect)

        def connect(self):
            events.append("connect")
            if connect_fails:
                raise ConnectionFailedError

        def disconnect(self):
            events.append("disconnect")

    class Adapter:
        def __init__(self, robot):
            pass

        def bind_state_readers(self):
            events.append("bind")

    class Session:
        stream = object()

        def __init__(self, *args):
            pass

        def prepare(self):
            events.append("prepare")

        def set_enabled(self, enabled):
            events.append("sample")

        def close(self):
            events.append("close")

    monkeypatch.setattr(entrypoint._env, "BiEliteCS66RTEnvironment", Environment)
    monkeypatch.setattr(elite_display, "EliteVisualizationAdapter", Adapter)
    monkeypatch.setattr(lifecycle, "VisualizationSession", Session)
    monkeypatch.setattr(
        entrypoint._websocket_client_policy,
        "WebsocketClientPolicy",
        lambda **kwargs: SimpleNamespace(get_server_metadata=dict),
    )
    monkeypatch.setattr(
        entrypoint._runtime,
        "Runtime",
        lambda **kwargs: SimpleNamespace(run=lambda: events.append("run")),
    )
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *args: None)
    args = entrypoint.Args(robot_recipe="diagonal-08-taccap", visualize=True)
    if connect_fails:
        with pytest.raises(ConnectionFailedError):
            entrypoint.main(args)
        assert events == ["prepare", "connect", "close"]
    else:
        entrypoint.main(args)
        assert events == ["prepare", "connect", "bind", "sample", "run", "close", "disconnect"]
