"""Read-only Elite SDK adapter tests; no robot, camera or policy server."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("xense_motion_viewer")
pytest.importorskip("libpyelite.assembly")
from libpyelite.assembly import Assembly
from xense_motion_viewer.urdf import URDFModel
from xense_motion_viewer.urdf import xyz_rpy

from examples.bi_elite_cs66_rt.visualization import EliteVisualizationAdapter


class Rtsi:
    stamp = 10.0
    connected = True

    def __init__(self):
        self.joints = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        self.tcp = np.array([0.1, 0.2, 0.3, 0.2, -0.3, 0.4])

    def isConnected(self):  # noqa: N802 - Elite SDK API
        return self.connected

    def getTimestamp(self):  # noqa: N802 - Elite SDK API
        return self.stamp

    def getActualJointPositions(self):  # noqa: N802 - Elite SDK API
        return self.joints.copy()

    def getActualTCPPose(self):  # noqa: N802 - Elite SDK API
        return self.tcp.copy()


class Gripper:
    valid = True

    def __init__(self):
        # Mirror the real TacCap API: Elite owns the snapshot reader, not the
        # shared gripper backend. Calibration is read only when binding it.
        self.config = SimpleNamespace(type="taccap_follower")
        self.is_connected = True
        self.calibration_reads = 0
        self._gripper = SimpleNamespace(get_gripper_config=self.calibration)
        self._loop = SimpleNamespace(snapshot=self.snapshot)

    def calibration(self):
        self.calibration_reads += 1
        return SimpleNamespace(min_open_rad=0.0, max_open_rad=2.0, flags=1)

    def snapshot(self):
        obs = SimpleNamespace(valid=self.valid, seq=1, raw_pos=1.0, age_ms=0.0)
        return SimpleNamespace(running=True, observation=obs)


@pytest.fixture
def adapter():
    config = SimpleNamespace(
        native_assembly="diagonal-08", left_tool_transform=np.eye(4), right_tool_transform=np.eye(4)
    )
    robot = SimpleNamespace(
        config=config,
        _assembly=Assembly("diagonal-08"),
        _rtsi={side: Rtsi() for side in ("left", "right")},
        _gripper={side: Gripper() for side in ("left", "right")},
    )
    result = EliteVisualizationAdapter(robot)
    result.bind_state_readers()
    return result


def test_sdk_joints_tcp_and_grippers_are_independent_of_policy(adapter):
    sample = adapter.read_state(0.5)
    model = URDFModel(adapter.describe())
    model.link_transforms(sample.state.joint_positions)
    for side in ("left", "right"):
        rtsi = adapter.robot._rtsi[side]
        actual = [sample.state.joint_positions[key] for key in adapter._joint_names[side]]
        np.testing.assert_array_equal(actual, rtsi.joints)
        expected = adapter.robot._assembly.world_from_base(side) @ xyz_rpy(rtsi.tcp[:3], rtsi.tcp[3:])
        np.testing.assert_allclose(sample.state.tcp_poses[side], expected)
        joint, knots = next(iter(adapter._calibration[side].items()))
        assert sample.state.joint_positions[joint] == np.interp(0.5, *np.asarray(knots).T)
    assert "left:10.000000000" in sample.sample_id
    adapter.read_state(0.6)
    assert all(gripper.calibration_reads == 1 for gripper in adapter.robot._gripper.values())


def test_gripper_readers_work_without_collision_layer_and_reject_disconnected_backend(adapter):
    assert not hasattr(adapter.robot, "_station")
    assert adapter.read_state(0).state is not None
    adapter.robot._gripper["right"].is_connected = False
    with pytest.raises(RuntimeError, match="right gripper state is invalid"):
        adapter.read_state(0.1)


def test_collision_readers_are_reused_without_calibration_io_during_active_servo(adapter):
    adapter.robot._station_gripper_readers = adapter._gripper_readers.copy()
    adapter.bind_state_readers()
    assert all(gripper.calibration_reads == 1 for gripper in adapter.robot._gripper.values())
    assert adapter.read_state(0).state is not None


def test_elite_only_action_layout_and_rotation_columns(adapter):
    actions = np.zeros((4, 20))
    actions[:, :9] = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0]
    actions[:, 10:19] = [-0.1, -0.2, 0.4, 0, 1, 0, -1, 0, 0]
    actions[:, [9, 19]] = 999  # Grippers are not TCP pose components.
    poses = adapter.decode_actions(actions)
    np.testing.assert_allclose(poses["left"][0, :3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(poses["right"][0, :3, 3], [-0.1, -0.2, 0.4])
    np.testing.assert_allclose(poses["right"][0, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="20"):
        adapter.decode_actions(np.ones((3, 18)))


def test_stale_invalid_and_disconnected_sdk_feedback_is_not_invented(adapter):
    adapter.read_state(0)
    adapter._changed_at["left"] = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="stale"):
        adapter.read_state(0.1)
    adapter.robot._rtsi["left"].connected = False
    with pytest.raises(RuntimeError, match="disconnected"):
        adapter.read_state(0.2)
    adapter.robot._rtsi["left"].connected = True
    adapter.robot._rtsi["left"].stamp += 1
    adapter.robot._rtsi["right"].joints[:] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        adapter.read_state(0.3)


@pytest.mark.parametrize("paced_mode", [False, True])
def test_elite_chunk_sdk_state_and_real_rerun_display_together(adapter, tmp_path, paced_mode):
    import rerun.dataframe as rdf
    from xense_client.base_policy import BasePolicy
    from xense_client.paced_broker import PacedBroker
    from xense_client.visualization import ObservedEnvironment
    from xense_client.visualization import VisualizationConfig
    from xense_client.visualization import VisualizationSession
    from xense_client.visualization_brokers import ObservedActionChunkBroker

    actions = np.zeros((40, 20), dtype=np.float32)
    actions[:, :9] = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0]
    actions[:, 10:19] = [-0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0]

    class Policy(BasePolicy):
        def infer(self, obs):
            return {"actions": actions.copy()}

    class Environment:
        def apply_action(self, action):
            assert set(action) == {"actions"}

    path = tmp_path / "elite-live.rrd"
    session = VisualizationSession(
        VisualizationConfig(spawn_viewer=False, save_path=str(path), display_hz=60), lambda: adapter
    )
    paced = None
    horizon = 30 if paced_mode else 3
    try:
        session.set_enabled(True)
        states = session.subscribe_robot_state()
        broker = ObservedActionChunkBroker(Policy(), action_horizon=horizon, action_stream=session.stream)
        if paced_mode:
            paced = PacedBroker(broker, queue_size=3, target_hz=30)
            paced.start()
            action = paced.infer({})
        else:
            action = broker.infer({})
        assert states.get(timeout=2).state is not None
        time.sleep(0.04)  # Allow the asynchronous renderer to consume the first plan.
        ObservedEnvironment(Environment(), session.stream).apply_action(action)
        adapter.robot._rtsi["left"].stamp += 1
        adapter.robot._rtsi["left"].joints[0] += 0.1
        time.sleep(0.04)
        display = session._display
    finally:
        if paced is not None:
            paced.stop()
        session.close()
    assert display.stats.errors == 0
    assert not display.stats.alive
    assert session.errors == 0
    assert states.closed
    recording = rdf.load_recording(path)
    entity = "/world/live/elite_station/future/left/steps"
    table = recording.view(index="host_elapsed", contents=entity).select().read_all()
    positions = table.column(entity + ":Points3D:positions").to_pylist()
    assert any(value is not None and len(value) == horizon for value in positions)
    radii = table.column(entity + ":Points3D:radii").to_pylist()
    assert any(value is not None and len(value) == horizon and value[0] == 0 for value in radii)
