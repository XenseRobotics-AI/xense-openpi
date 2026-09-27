"""Motion-gated Pico takeover (docs/architecture.md § 4.49).

Holding the grip (the motion key) arms the takeover; it starts as soon as a
controller moves past the threshold. A gripper that disagrees with the robot's
is reported, not blocked, and only the hand whose grip is down follows its own
trigger. Reuses the stubbed load of intervention.py from
test_pico_button_monitor.py, so no robot, VR SDK or xense lerobot fork is needed.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from examples.bi_flexiv_rizon4_rt.pico_button_monitor_test import _module

TakeoverMotion = _module.TakeoverMotion
Pico4InterventionController = _module.Pico4InterventionController
ACTION_KEYS = _module._ACTION_KEYS_IN_ORDER

MOTION = TakeoverMotion(position_m=0.005, rotation_rad=math.radians(3.0))
_IDENTITY = [0.0, 0.0, 0.0, 1.0]  # qx, qy, qz, qw
OPEN, CLOSED = 1.0, 0.0
LEFT_GRIPPER, RIGHT_GRIPPER = 18, 19  # the 20D action's last two dims


class _FakeXrt:
    def __init__(self) -> None:
        self.grips = [0.0, 0.0]
        self.poses = [np.array([0.1, 0.2, 0.3, *_IDENTITY]), np.array([-0.1, 0.2, 0.3, *_IDENTITY])]
        self.triggers = [0.0, 0.0]  # released trigger -> gripper command 1.0 (open)

    def get_left_grip(self) -> float:
        return self.grips[0]

    def get_right_grip(self) -> float:
        return self.grips[1]

    def get_left_controller_pose(self):
        return self.poses[0].copy()

    def get_right_controller_pose(self):
        return self.poses[1].copy()

    def get_left_trigger(self) -> float:
        return self.triggers[0]

    def get_right_trigger(self) -> float:
        return self.triggers[1]


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        self.warnings.append(message)


@pytest.fixture
def log(monkeypatch):
    recorder = _Log()
    monkeypatch.setattr(_module, "logger", recorder)
    return recorder


def _controller(motion=MOTION, measured_grippers=(OPEN, OPEN)):
    xrt = _FakeXrt()
    resets = []
    tcp = {"x": 0.0}

    def reset_to_pose(left, right, left_gripper, right_gripper):
        resets.append(float(left[0]))

    def current_tcp():
        return np.array([tcp["x"], 0, 0, 1, 0, 0, 0, measured_grippers[0]]), np.array(
            [tcp["x"], 0, 0, 1, 0, 0, 0, measured_grippers[1]]
        )

    config = SimpleNamespace(
        grip_enable_threshold=0.5, grip_disable_threshold=0.3, left_gripper_width=1.0, right_gripper_width=1.0
    )

    def get_action():
        """BiPico4's own mapping: both grippers follow their trigger, held grip or not."""
        action = dict.fromkeys(ACTION_KEYS, 0.0)
        action["left_gripper.pos"] = 1.0 - xrt.triggers[0] * config.left_gripper_width
        action["right_gripper.pos"] = 1.0 - xrt.triggers[1] * config.right_gripper_width
        return action

    teleop = SimpleNamespace(_xrt=xrt, config=config, reset_to_pose=reset_to_pose, get_action=get_action)
    base_env = SimpleNamespace(_env=SimpleNamespace(robot=SimpleNamespace(get_current_tcp_pose_quat=current_tcp)))
    return Pico4InterventionController(teleop, base_env, takeover_motion=motion), xrt, resets, tcp


def _arm_and_move(controller, xrt, *, hand=0, delta=0.006, command=(OPEN, OPEN)):
    xrt.grips[hand] = 0.9
    assert controller.poll_and_decide(gripper_command=command) is False  # armed
    xrt.poses[hand][:3] += [delta, 0.0, 0.0]
    return controller.poll_and_decide(gripper_command=command)


def test_held_grip_arms_and_motion_takes_over_from_the_live_tcp(log):
    controller, xrt, resets, tcp = _controller()
    xrt.grips[0] = 0.9
    for policy_step in range(4):
        tcp["x"] = 0.01 * policy_step  # the policy keeps moving the arm while armed
        assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is False
    assert resets == []
    xrt.poses[0][:3] += [0.006, 0.0, 0.0]
    tcp["x"] = 0.05
    assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is True
    assert resets == [0.05]  # re-anchored at the takeover tick, not at the grip press
    assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is True
    assert resets == [0.05]
    assert log.warnings == []


@pytest.mark.parametrize(
    ("change", "takes_over"),
    [
        ("position_4mm", False),
        ("rotation_2deg", False),
        ("trigger_only", False),
        ("position_6mm", True),
        ("rotation_4deg", True),
    ],
)
def test_takeover_needs_controller_motion_not_trigger_travel(log, change, takes_over):
    controller, xrt, _, _ = _controller()
    xrt.grips[1] = 0.9
    assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is False
    if change.startswith("position"):
        xrt.poses[1][2] += 0.004 if change.endswith("4mm") else 0.006
    elif change.startswith("rotation"):
        half = math.radians(2.0 if change.endswith("2deg") else 4.0) / 2
        xrt.poses[1][3:] = [math.sin(half), 0.0, 0.0, math.cos(half)]
    else:
        xrt.triggers[1] = 0.3  # gripper command 0.7: still open, and no controller motion
    assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is takes_over


def test_gripper_mismatch_takes_over_anyway_and_warns_about_the_jump(log):
    controller, xrt, resets, _ = _controller()
    robot = (CLOSED, OPEN)  # the left gripper is shut; the released left trigger opens it on takeover
    assert _arm_and_move(controller, xrt, command=robot) is True
    assert len(resets) == 1
    assert len(log.warnings) == 1
    assert "left" in log.warnings[0]
    assert "0.00 -> 1.00" in log.warnings[0]


def test_only_the_driving_hand_follows_its_trigger(log):
    """A one-handed takeover must not open the other hand's gripper (it may be holding something)."""
    controller, xrt, _, _ = _controller()
    robot = (CLOSED, OPEN)  # the left gripper is shut on an object; its trigger is released
    assert _arm_and_move(controller, xrt, hand=1, command=robot) is True
    action = controller.get_override_action()
    assert action[LEFT_GRIPPER] == pytest.approx(CLOSED)  # held at the robot's command, not the released trigger
    assert action[RIGHT_GRIPPER] == pytest.approx(OPEN)  # the driving hand follows its own trigger
    assert log.warnings == []  # the idle hand neither jumps nor reports

    xrt.triggers[1] = 1.0  # squeeze the driving hand: only that gripper moves
    assert controller.poll_and_decide(gripper_command=robot) is True
    action = controller.get_override_action()
    assert action[LEFT_GRIPPER] == pytest.approx(CLOSED)
    assert action[RIGHT_GRIPPER] == pytest.approx(CLOSED)


def test_a_second_grip_mid_takeover_starts_driving_its_own_gripper(log):
    controller, xrt, _, _ = _controller()
    robot = (CLOSED, OPEN)
    assert _arm_and_move(controller, xrt, hand=1, command=robot) is True
    assert controller.get_override_action()[LEFT_GRIPPER] == pytest.approx(CLOSED)

    xrt.grips[0] = 0.9  # the left hand joins the takeover: its released trigger now drives
    assert controller.poll_and_decide(gripper_command=robot) is True
    assert controller.get_override_action()[LEFT_GRIPPER] == pytest.approx(OPEN)
    assert len(log.warnings) == 1
    assert "left" in log.warnings[0]


def test_the_hold_is_dropped_once_the_takeover_ends(log):
    controller, xrt, _, _ = _controller()
    assert _arm_and_move(controller, xrt, hand=1, command=(CLOSED, OPEN)) is True
    xrt.grips[1] = 0.0
    assert controller.poll_and_decide(gripper_command=(CLOSED, OPEN)) is False
    assert controller._held_gripper == [None, None]


def test_a_matching_trigger_reports_no_jump(log):
    """Holding a cable the jaws measure 0.4 while commanded closed; the command is the reference."""
    controller, xrt, _, _ = _controller(measured_grippers=(0.4, OPEN))
    xrt.triggers[0] = 1.0
    assert _arm_and_move(controller, xrt, command=(CLOSED, OPEN)) is True
    assert log.warnings == []


def test_without_a_command_yet_the_measured_gripper_is_the_reference(log):
    controller, xrt, _, _ = _controller(measured_grippers=(CLOSED, OPEN))
    assert _arm_and_move(controller, xrt, command=None) is True
    assert len(log.warnings) == 1


@pytest.mark.parametrize(
    ("moving", "flip_gripper", "warnings"), [(False, False, 0), (True, False, 1), (False, True, 1)]
)
def test_release_warns_unless_hand_and_grippers_are_still(log, moving, flip_gripper, warnings):
    controller, xrt, _, _ = _controller()
    assert _arm_and_move(controller, xrt) is True
    for _ in range(3):
        if moving:
            xrt.poses[0][:3] += [0.004, 0.0, 0.0]
        controller.poll_and_decide(gripper_command=(OPEN, OPEN))
    if flip_gripper:
        xrt.triggers[1] = 0.9
    xrt.grips[0] = 0.0
    assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is False
    assert controller.consume_release_event() is True  # a warning never blocks the release
    assert len(log.warnings) == warnings


def test_release_event_only_after_an_actual_takeover(log):
    controller, xrt, _, _ = _controller()
    xrt.grips[0] = 0.9
    controller.poll_and_decide(gripper_command=(OPEN, OPEN))
    xrt.grips[0] = 0.0
    assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is False
    assert controller.consume_release_event() is False  # below the motion threshold: no replan

    assert _arm_and_move(controller, xrt) is True
    xrt.grips[0] = 0.0
    assert controller.poll_and_decide(gripper_command=(OPEN, OPEN)) is False
    assert controller.consume_release_event() is True


def test_without_takeover_motion_a_held_grip_takes_over_at_once(log):
    controller, xrt, resets, _ = _controller(motion=None)
    xrt.grips[0] = 0.9
    assert controller.poll_and_decide() is True
    assert len(resets) == 1
