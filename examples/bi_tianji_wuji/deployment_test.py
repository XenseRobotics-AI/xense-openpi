"""Offline contract and lifecycle checks for the Tianji/Wuji client; these tests never connect hardware."""

import dataclasses
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from examples import run_config
from examples.bi_tianji_wuji import main as entry
from examples.bi_tianji_wuji import recipe
from examples.bi_tianji_wuji.env import TianjiWujiEnvironment
from examples.bi_tianji_wuji.main import RUNS_DIR
from examples.bi_tianji_wuji.main import Args
from examples.bi_tianji_wuji.main import run_episode
from examples.bi_tianji_wuji.schema import ACTION_KEYS
from examples.bi_tianji_wuji.schema import CAMERAS
from examples.bi_tianji_wuji.schema import action_to_dict
from examples.bi_tianji_wuji.schema import state_from_observation
from examples.bi_tianji_wuji.schema import validate_actions


def valid_chunk(n=50):
    chunk = np.zeros((n, 58), dtype=np.float32)
    chunk[:, [3, 7, 12, 16]] = 1  # identity rotations, first two columns
    chunk[:, 18:] = np.arange(40) / 10  # fingers are not gripper values in [0,1]
    return chunk


def fake_robot():
    state = valid_chunk(1)[0]
    raw = dict(reversed(list(zip(ACTION_KEYS, state, strict=True))))
    raw.update({name: np.full((24, 32, 3), (20, 40, 60), dtype=np.uint8) for name in CAMERAS})
    raw["left_joint_1.pos"] = 999  # diagnostic fields must not leak into state
    return Mock(
        action_features=dict.fromkeys(ACTION_KEYS, float),
        get_observation=Mock(return_value=raw),
        wait_for_reset_completion=Mock(return_value=True),
    )


def test_explicit_state_order_and_finger_targets():
    robot = fake_robot()
    state = state_from_observation(robot.get_observation())
    np.testing.assert_array_equal(state, valid_chunk(1)[0])
    command = action_to_dict(state)
    assert command["l_index_finger_mcp_flex.pos"] == 0
    assert command["l_pinky_mcp_flex.pos"] == pytest.approx(0.8)
    assert command["l_ring_finger_mcp_flex.pos"] == pytest.approx(1.2)
    assert command["l_thumb_cmc_flex.pos"] == pytest.approx(1.6)
    assert command["r_thumb_ip.pos"] == pytest.approx(3.9)
    assert "left_gripper.pos" not in command


@pytest.mark.parametrize("shape", [(58,), (1, 20), (1, 64), (0, 58), (1, 1, 58)])
def test_invalid_action_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        validate_actions(np.zeros(shape))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_values_anywhere_in_chunk(bad):
    chunk = valid_chunk()
    chunk[-1, -1] = bad
    with pytest.raises(ValueError, match="finite"):
        validate_actions(chunk)


def test_degenerate_rotation_rejected():
    chunk = valid_chunk()
    chunk[20, 6:9] = chunk[20, 3:6]
    with pytest.raises(ValueError, match="rotation"):
        validate_actions(chunk)


def test_dry_run_suppresses_all_application_motion():
    robot = fake_robot()
    env = TianjiWujiEnvironment(robot, dry_run=True)
    env.connect()
    env.reset()
    env.hold()
    env.apply_action(valid_chunk(1)[0])
    observation = env.get_observation("sort")
    env.disconnect()
    robot.start_reset_to_initial_position.assert_not_called()
    robot.cancel.assert_not_called()
    robot.send_action.assert_not_called()
    robot.disconnect.assert_called_once()
    assert observation["state"].shape == (58,)
    assert observation["prompt"] == "sort"
    for frame in observation["images"].values():
        assert frame.shape == (3, 224, 224)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame[:, 112, 112], [20, 40, 60])


def test_reset_waits_for_both_devices_and_timeout_prevents_inference(monkeypatch):
    robot = fake_robot()
    robot.wait_for_reset_completion.side_effect = [False, True]
    env = TianjiWujiEnvironment(robot, dry_run=False)
    env.reset()
    assert robot.wait_for_reset_completion.call_count == 2
    robot.wait_for_reset_completion.side_effect = None
    robot.wait_for_reset_completion.return_value = False
    ticks = iter([0.0, 16.0])
    monkeypatch.setattr("examples.bi_tianji_wuji.env.time.monotonic", lambda: next(ticks))
    policy = Mock()
    with pytest.raises(TimeoutError, match="reset"):
        run_episode(env, policy, Args(robot_recipe="block-sort"))
    policy.infer.assert_not_called()
    robot.send_action.assert_not_called()


def test_loop_exact_steps_fresh_observation_and_absolute_targets(monkeypatch):
    monkeypatch.setattr("examples.bi_tianji_wuji.main.time.sleep", lambda _: None)
    robot = fake_robot()
    env = TianjiWujiEnvironment(robot, dry_run=False)
    events = []
    robot.cancel.side_effect = lambda: events.append("hold")

    def infer(obs):
        assert events[-1] == "hold"
        events.append("infer")
        chunk = valid_chunk()
        chunk[:, 0] = 0.42
        return {"actions": chunk}

    policy = Mock(infer=Mock(side_effect=infer))
    run_episode(env, policy, Args(robot_recipe="block-sort", action_horizon=2, max_episode_steps=3))
    assert robot.send_action.call_count == 3
    assert robot.get_observation.call_count == 3
    assert policy.infer.call_count == 2
    assert events == ["hold", "infer", "hold", "infer", "hold"]
    assert robot.send_action.call_args.args[0]["left_tcp.x"] == pytest.approx(0.42)


@pytest.mark.parametrize("failure", ["short", "late_nan", "timeout"])
def test_bad_chunk_never_partially_executes(failure):
    robot = fake_robot()
    env = TianjiWujiEnvironment(robot, dry_run=False)
    chunk = valid_chunk(2 if failure == "short" else 50)
    if failure == "late_nan":
        chunk[-1, -1] = np.nan
    policy = Mock(infer=Mock(return_value={"actions": chunk}))
    if failure == "timeout":
        policy.infer.side_effect = TimeoutError("network")
    with pytest.raises((ValueError, TimeoutError)):
        run_episode(env, policy, Args(robot_recipe="block-sort"))
    robot.send_action.assert_not_called()


@pytest.mark.parametrize("name", ["block-sort", "dry-run"])
def test_run_yaml_and_cli_override(name):
    args = run_config.cli(
        lambda args: args,
        Args,
        RUNS_DIR,
        ["--args.run", name, "--args.host", "server", "--args.max-episode-steps", "7"],
    )
    args.validate()
    assert args.host == "server"
    assert args.max_episode_steps == 7
    assert args.dry_run == (name == "dry-run")


@pytest.mark.parametrize(
    ("field", "value"),
    [("runtime_hz", 0), ("runtime_hz", float("nan")), ("action_horizon", 51), ("max_episode_steps", 0)],
)
def test_invalid_settings(field, value):
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(Args(robot_recipe="block-sort"), **{field: value}).validate()


def test_recipe_against_installed_hardware_schema():
    pytest.importorskip("lerobot.robots.tianji_arm_wuji")
    config = recipe.load_robot_config("block-sort")
    assert tuple(config.wuji.action_keys) == ACTION_KEYS[18:]
    assert not config.go_home_on_connect
    assert not config.wuji.return_to_zero_on_disconnect
    assert all(not camera.use_depth for camera in config.cameras.values())


def test_main_cleans_partial_connection_and_rejects_wrong_server(monkeypatch, tmp_path):
    robot = fake_robot()
    robot.connect.side_effect = RuntimeError("second hand failed")
    monkeypatch.setitem(sys.modules, "lerobot.robots.tianji_arm_wuji", SimpleNamespace(TianjiArmWuji=lambda _: robot))
    monkeypatch.setattr(recipe, "load_robot_config", lambda _: SimpleNamespace(command_timeout_s=0.5))
    policy = Mock(get_server_metadata=Mock(return_value={"config": entry.TRAIN_CONFIG}))
    monkeypatch.setitem(
        sys.modules, "xense_client.websocket_client_policy", SimpleNamespace(WebsocketClientPolicy=lambda **_: policy)
    )
    args = Args(robot_recipe="block-sort", log_file=str(tmp_path / "test.log"))
    with pytest.raises(RuntimeError, match="second hand"):
        entry.main(args)
    robot.disconnect.assert_called_once()
    policy.disconnect.assert_called_once()
    robot.reset_mock()
    policy.reset_mock()
    policy.get_server_metadata.return_value = {"config": "flexiv"}
    with pytest.raises(ValueError, match="Wrong policy"):
        entry.main(args)
    robot.connect.assert_not_called()
    policy.disconnect.assert_called_once()
