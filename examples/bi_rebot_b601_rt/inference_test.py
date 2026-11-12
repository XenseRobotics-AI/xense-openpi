"""Exercise the robot adapter and real RTC/runtime wiring without hardware or a server."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("lerobot.robots.bi_rebot_b601_rt", reason="Rebot client configs not installed")

from examples.bi_rebot_b601_rt import env
from examples.bi_rebot_b601_rt import main
from examples.bi_rebot_b601_rt import recipe
import examples.run_config as run_config


def pose_action():
    # Nonzero positions expose accidental client-side delta conversion.
    return np.array([0.3, 0.1, 0.2, 1, 0, 0, 0, 1, 0, 0.25, 0.4, -0.2, 0.1, 1, 0, 0, 0, 1, 0, 0.75])


class FakeRobot:
    def __init__(self):
        self.connect_calls = []
        self.disconnect_calls = []
        self.reset_calls = 0
        self.sent = []
        self.reads = 0
        self.fail_connect = False
        self.fail_reset = False
        self.raw = dict(zip(env.STATE_KEYS, pose_action(), strict=True))
        self.raw.update(
            {key: np.full((16, 20, 3), index, np.uint8) for index, key in enumerate(env.DEFAULT_CAMERA_MAP.values(), 1)}
        )

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        if self.fail_connect:
            raise RuntimeError("connect failed")

    def get_observation(self):
        self.reads += 1
        return dict(self.raw)

    def send_action(self, action):
        self.sent.append(action)

    def reset_to_initial_position(self):
        self.reset_calls += 1

    @property
    def rt_moving(self):
        if self.fail_reset:
            raise TimeoutError("reset timed out")
        return False

    def disconnect(self, **kwargs):
        self.disconnect_calls.append(kwargs)


@pytest.fixture
def robot(monkeypatch):
    robot = FakeRobot()
    monkeypatch.setattr(env, "make_robot_from_config", lambda config: robot)
    return robot


@pytest.fixture
def config():
    return recipe.load_robot_config("dm-taccap00-09")


def test_observation_and_action_match_recorded_layout(config, robot):
    environment = env.BiRebotB601RTEnvironment(config, render_height=16, render_width=20, prompt="sort blocks")
    assert robot.connect_calls == [{"calibrate": False, "go_to_start": True}]
    observation = environment.get_observation()
    np.testing.assert_allclose(observation["state"], pose_action())
    assert observation["prompt"] == "sort blocks"
    assert env.STATE_KEYS[9] == "left_gripper.pos"
    assert env.STATE_KEYS[19] == "right_gripper.pos"
    assert env.STATE_KEYS[10] == "right_tcp.x"
    for index, name in enumerate(("head", "left_wrist", "right_wrist"), 1):
        np.testing.assert_array_equal(observation["images"][name], np.full((3, 16, 20), index, np.uint8))
    actions = pose_action()
    actions[9], actions[19] = -2, 3
    environment.apply_action({"actions": actions})
    expected = pose_action()
    expected[9], expected[19] = 0, 1
    np.testing.assert_allclose([robot.sent[0][key] for key in env.STATE_KEYS], expected)
    assert robot.reads == 1  # Sending never reads cameras.
    np.testing.assert_array_equal(actions[[9, 19]], [-2, 3])  # Caller retains its action.


@pytest.mark.parametrize("bad", [np.zeros(19), np.zeros((1, 20)), np.full(20, np.nan), np.full(20, np.inf)])
def test_invalid_action_never_reaches_robot(config, robot, bad):
    environment = env.BiRebotB601RTEnvironment(config)
    with pytest.raises(ValueError, match="finite 20D"):
        environment.apply_action({"actions": bad})
    assert not robot.sent


def test_bad_right_rotation_rejected_before_either_arm_moves(config, robot):
    environment = env.BiRebotB601RTEnvironment(config)
    action = pose_action()
    action[16:19] = action[13:16]
    with pytest.raises(ValueError, match=r"right.*parallel"):
        environment.apply_action({"actions": action})
    assert not robot.sent


def test_native_gripper_radians_not_clipped(config, robot):
    arms = {
        f"{side}_arm_config": dataclasses.replace(getattr(config, f"{side}_arm_config"), gripper=None, has_gripper=True)
        for side in ("left", "right")
    }
    environment = env.BiRebotB601RTEnvironment(dataclasses.replace(config, **arms))
    action = pose_action()
    action[9], action[19] = -0.2, 1.5
    environment.apply_action({"actions": action})
    assert robot.sent[0]["left_gripper.pos"] == -0.2
    assert robot.sent[0]["right_gripper.pos"] == 1.5


def test_missing_camera_is_an_error(config, robot):
    del robot.raw["right_right_wrist"]
    environment = env.BiRebotB601RTEnvironment(config)
    with pytest.raises(RuntimeError, match="right_right_wrist"):
        environment.get_observation()


def test_dry_run_and_stop_suppress_policy_actions(config, robot):
    environment = env.BiRebotB601RTEnvironment(config, dry_run=True)
    environment.apply_action({"actions": pose_action()})
    assert not robot.sent
    environment = env.BiRebotB601RTEnvironment(config)
    environment.request_stop()
    environment.apply_action({"actions": pose_action()})
    assert not robot.sent
    assert environment.is_episode_complete()
    environment.reset()
    environment.reset()
    assert not robot.reset_calls


def test_reset_waits_on_driver_and_propagates_timeout(config, robot):
    environment = env.BiRebotB601RTEnvironment(config)
    environment.reset()  # connect already reset the robot.
    assert not robot.reset_calls
    robot.fail_reset = True
    with pytest.raises(TimeoutError, match="reset timed out"):
        environment.reset()
    assert robot.reset_calls == 1


def test_no_go_to_start_and_partial_connect_cleanup(config, robot):
    environment = env.BiRebotB601RTEnvironment(config, go_to_start=False)
    environment.reset()
    environment.reset()
    assert not robot.reset_calls
    environment.disconnect()
    assert robot.disconnect_calls == [{}]
    robot.fail_connect = True
    with pytest.raises(RuntimeError, match="connect failed"):
        env.BiRebotB601RTEnvironment(config)
    assert robot.disconnect_calls[-1] == {"reset": False}


@pytest.mark.parametrize(
    ("joint", "gripper", "start"), [(True, True, True), (False, False, True), (False, True, False)]
)
def test_incompatible_recipe_rejected_before_connect(config, robot, joint, gripper, start):
    arm = dataclasses.replace(
        config.left_arm_config,
        use_joint_observation=joint,
        gripper=config.left_arm_config.gripper if gripper else None,
        has_gripper=False,
        reset_position_rad=config.left_arm_config.reset_position_rad if start else None,
    )
    with pytest.raises(ValueError, match="left:"):
        env.BiRebotB601RTEnvironment(dataclasses.replace(config, left_arm_config=arm))
    assert not robot.connect_calls


def test_nested_recipe_overrides_preserve_hardware(config):
    changed = recipe.load_robot_config(
        "dm-taccap00-09", enable_tactile=True, max_lin_speed=0.15, max_ang_speed=0.8, reset_on_exit=False
    )
    assert not changed.reset_on_exit
    for side in ("left", "right"):
        original = getattr(config, f"{side}_arm_config")
        arm = getattr(changed, f"{side}_arm_config")
        assert arm.gripper.enable_tactile
        assert arm.max_cartesian_linear_speed == 0.15
        assert arm.max_cartesian_angular_speed == 0.8
        assert arm.channel == original.channel
        assert arm.motor_configs == original.motor_configs
        assert arm.reset_position_rad == original.reset_position_rad
        assert arm.default_urdf == "DM_TacCap00/urdf/rebot_b601_dm_taccap_v1_gripper.urdf"


def test_upstream_recipe_relative_urdf_resolution(tmp_path):
    path = tmp_path / "recipes/record/bi_rebot/test.yaml"
    path.parent.mkdir(parents=True)
    urdf = tmp_path / "third_party/model.urdf"
    urdf.parent.mkdir()
    urdf.write_text('<robot name="test"/>')
    raw = yaml.safe_load(recipe.resolve_recipe_path("dm-taccap00-09").read_text())
    for side in ("left", "right"):
        raw["robot"][f"{side}_arm_config"]["urdf_path"] = "third_party/model.urdf"
    path.write_text(yaml.safe_dump(raw))
    decoded = recipe.load_robot_config(path)
    assert decoded.left_arm_config.urdf_path == str(urdf)
    assert decoded.right_arm_config.urdf_path == str(urdf)


def test_run_presets_and_cli_override():
    args = run_config.cli(
        main.main,
        main.Args,
        main.RUNS_DIR,
        ["--args.run", "pick-and-place-rtc", "--args.host", "test-server", "--args.no-rtc-enabled"],
    )
    assert args.robot_recipe == "dm-taccap00-09"
    assert args.host == "test-server"
    assert not args.rtc_enabled
    dry = run_config.load_defaults(main.Args, main.RUNS_DIR, ["--args.run", "dry-run"])
    assert dry.dry_run
    assert dry.rtc_enabled


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.closed = False
        self._ws = SimpleNamespace(close=self.close)

    def get_server_metadata(self):
        return {}

    def reset(self):
        pass

    def infer(self, observation, **kwargs):
        if self.closed:
            raise RuntimeError("closed")
        self.calls.append((observation, kwargs))
        return {"actions": np.tile(pose_action(), (50, 1))}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("rtc", [False, True])
@pytest.mark.parametrize("action_hz", [0.0, 30.0])
def test_main_real_runtime_and_broker_without_hardware(monkeypatch, robot, rtc, action_hz):
    policy = FakePolicy()
    monkeypatch.setattr(main.websocket_client_policy, "WebsocketClientPolicy", lambda **kwargs: policy)
    main.main(
        main.Args(
            robot_recipe="dm-taccap00-09",
            rtc_enabled=rtc,
            action_hz=action_hz,
            max_episode_steps=3,
            render_height=16,
            render_width=20,
            action_queue_size_to_get_new_actions=49,
        )
    )
    assert robot.sent
    assert policy.closed
    assert robot.disconnect_calls == [{}]
    np.testing.assert_allclose([robot.sent[0][key] for key in env.STATE_KEYS], pose_action())
    if rtc:
        assert len(policy.calls) >= 3, "RTC must replenish the queue after warmup"
        prefixes = [
            kwargs["prev_chunk_left_over"]
            for _, kwargs in policy.calls
            if kwargs.get("prev_chunk_left_over") is not None
        ]
        assert prefixes, "RTC must forward its absolute chunk prefix to the server"
        for prefix in prefixes:
            np.testing.assert_allclose(prefix, np.tile(pose_action(), (len(prefix), 1)))
        assert all(kwargs["execution_horizon"] == 50 for _, kwargs in policy.calls)


def test_rtc_frequency_tracks_consumer_and_cleanup_on_error(monkeypatch, robot):
    policy = FakePolicy()
    broker = main.make_broker(main.Args(rtc_enabled=True, runtime_hz=30, action_hz=60), policy)
    assert broker._frequency_hz == 60
    assert broker._delta_state_dim == 0
    broker.stop()
    monkeypatch.setattr(main.websocket_client_policy, "WebsocketClientPolicy", lambda **kwargs: policy)
    del robot.raw["head"]
    with pytest.raises(RuntimeError, match="Required camera"):
        main.main(main.Args(robot_recipe="dm-taccap00-09", rtc_enabled=True))
    assert policy.closed
    assert robot.disconnect_calls == [{}]
    assert not robot.sent


def test_decoupled_camera_failure_stops_actions_and_reaches_caller(monkeypatch, robot):
    policy = FakePolicy()
    monkeypatch.setattr(main.websocket_client_policy, "WebsocketClientPolicy", lambda **kwargs: policy)
    read = robot.get_observation

    def fail_after_warmup():
        if robot.reads:
            raise RuntimeError("camera disconnected")
        return read()

    monkeypatch.setattr(robot, "get_observation", fail_after_warmup)
    with pytest.raises(RuntimeError, match="camera disconnected"):
        main.main(main.Args(robot_recipe="dm-taccap00-09", rtc_enabled=True, action_hz=30, max_episode_steps=10))
    assert policy.closed
    assert robot.disconnect_calls == [{}]
    assert not robot.reset_calls  # No episode reset after a worker fault.


@pytest.mark.parametrize("action_hz", [0, 30])
def test_each_episode_has_a_new_rtc_queue(monkeypatch, robot, action_hz):
    policy = FakePolicy()
    monkeypatch.setattr(main.websocket_client_policy, "WebsocketClientPolicy", lambda **kwargs: policy)
    original_make_broker = main.make_broker
    brokers = []

    def capture_broker(args, client):
        broker = original_make_broker(args, client)
        brokers.append(broker)
        return broker

    monkeypatch.setattr(main, "make_broker", capture_broker)
    main.main(
        main.Args(
            robot_recipe="dm-taccap00-09", rtc_enabled=True, num_episodes=2, max_episode_steps=1, action_hz=action_hz
        )
    )
    assert len(brokers) == 2
    assert brokers[0] is not brokers[1]
    assert all(not broker._thread.is_alive() for broker in brokers)
    assert sum(kwargs["prev_chunk_left_over"] is None for _, kwargs in policy.calls) == 2
    assert policy.closed


@pytest.mark.parametrize(("name", "value"), [("runtime_hz", 0), ("action_hz", float("nan")), ("default_delay", -1)])
def test_invalid_runtime_settings_fail_before_connect(robot, name, value):
    with pytest.raises(ValueError, match=name):
        main.main(dataclasses.replace(main.Args(robot_recipe="dm-taccap00-09"), **{name: value}))
    assert not robot.connect_calls
