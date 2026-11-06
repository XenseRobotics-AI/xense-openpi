"""Reset watchdogs and partial-connection cleanup, without contacting hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("lerobot.robots.bi_elite_cs66_rt", reason="Elite client configs not installed")

from examples.bi_elite_cs66_rt import real_env


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRobot:
    def __init__(self, clock):
        self.clock = clock
        self.start_delay = 0.1
        self.finish_delay = 0.3
        self.is_connected = False  # Includes the partially connected case.
        self.connect = Mock()
        self.disconnect = Mock()
        self.reset_to_initial_position = Mock()

    @property
    def rt_moving(self):
        return self.start_delay <= self.clock.now < self.finish_delay


@pytest.fixture
def setup(monkeypatch):
    clock = FakeClock()
    robot = FakeRobot(clock)
    config = SimpleNamespace(
        control_mode=real_env.BiEliteCS66RTControlMode.CARTESIAN_SERVO,
        observe_tcp=True,
        left_gripper=object(),
        right_gripper=object(),
        gripper=None,
        left_robot_ip="fake-left",
        right_robot_ip="fake-right",
    )
    monkeypatch.setattr(real_env, "make_robot_from_config", lambda config: robot)
    monkeypatch.setattr(real_env, "time", clock)
    return config, robot, clock


def test_reset_waits_until_motion_finishes(setup):
    config, robot, clock = setup
    environment = real_env.BiEliteCS66RTRealEnv(config)
    environment.reset()
    robot.reset_to_initial_position.assert_called_once_with()
    assert robot.finish_delay <= clock.now < 1.0


@pytest.mark.parametrize(
    ("start_delay", "finish_delay", "deadline", "message"),
    [(float("inf"), float("inf"), 1.0, "did not start"), (0.1, float("inf"), 15.0, "did not finish")],
)
def test_reset_timeout_raises(setup, start_delay, finish_delay, deadline, message):
    config, robot, clock = setup
    robot.start_delay, robot.finish_delay = start_delay, finish_delay
    environment = real_env.BiEliteCS66RTRealEnv(config)
    with pytest.raises(TimeoutError, match=message):
        environment.reset()
    assert deadline <= clock.now < deadline + 0.1


def test_no_go_to_start_skips_reset(setup):
    config, robot, clock = setup
    environment = real_env.BiEliteCS66RTRealEnv(config, go_to_start=False)
    robot.connect.assert_called_once_with(calibrate=False, go_to_start=False)
    environment.reset()
    robot.reset_to_initial_position.assert_not_called()
    assert clock.now == 0.0


def test_disconnect_releases_partial_connection(setup):
    config, robot, _ = setup
    environment = real_env.BiEliteCS66RTRealEnv(config, setup_robot=False)
    environment.disconnect()
    robot.disconnect.assert_called_once_with()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_connect_cleans_up_and_preserves_original_error(setup, cleanup_fails):
    config, robot, _ = setup
    error = RuntimeError("right arm connect failed")
    robot.connect.side_effect = error
    if cleanup_fails:
        robot.disconnect.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="right arm connect failed") as raised:
        real_env.BiEliteCS66RTRealEnv(config)
    assert raised.value is error
    robot.disconnect.assert_called_once_with()
