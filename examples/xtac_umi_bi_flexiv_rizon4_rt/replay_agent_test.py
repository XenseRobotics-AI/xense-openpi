"""Tests for the open-loop UMI replay plan.

The plan is executed verbatim by the arms, so pin what it must guarantee: it starts
at the arms' own pose, preserves the recorded motion up to the estimated yaw, and
refuses episodes that were not started the way the arms start.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.xtac_umi_bi_flexiv_rizon4_rt import gripper_frame
from examples.xtac_umi_bi_flexiv_rizon4_rt import rebase_umi_dataset as rebase
from examples.xtac_umi_bi_flexiv_rizon4_rt import replay_agent

LEFT, RIGHT = rebase.LEFT_TCP, rebase.RIGHT_TCP

# UMI gripper axes (x fingertip) expressed in the Flexiv gripper frame (z fingertip).
_UMI_TO_FLEXIV = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _pose9(rotation: np.ndarray, position) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return gripper_frame.matrix_to_pose9(matrix)


def _vector(left: np.ndarray, right: np.ndarray, grippers=(0.5, 0.5)) -> np.ndarray:
    return np.concatenate((left, [grippers[0]], right, [grippers[1]]))


def _umi_start(world_yaw_deg: float = 0.0, right_extra_yaw_deg: float = 0.0) -> np.ndarray:
    """Hands toed in and pitched 20 deg down, in a Pico world turned by ``world_yaw_deg``."""
    left = _rot("z", world_yaw_deg - 40) @ _rot("y", 20)
    right = _rot("z", world_yaw_deg + 40 + right_extra_yaw_deg) @ _rot("y", 20)
    return _vector(_pose9(left, [0.3, 0.2, -0.4]), _pose9(right, [0.3, -0.2, -0.4]))


def _home() -> np.ndarray:
    """Arms holding the same physical pose as `_umi_start(0)`, in the Flexiv TCP convention."""
    left = _rot("z", -40) @ _rot("y", 20) @ _UMI_TO_FLEXIV.T
    right = _rot("z", 40) @ _rot("y", 20) @ _UMI_TO_FLEXIV.T
    return _vector(_pose9(left, [1.0, 0.0, -0.2]), _pose9(right, [1.1, 0.0, -0.2]))


def _episode(states: np.ndarray) -> replay_agent.Episode:
    """action[t] = state[t+1], as the rig records."""
    return replay_agent.Episode(states=states[:-1], actions=states[1:], fps=30.0)


def _moved(start: np.ndarray, offset, world_rotation: np.ndarray, steps: int) -> np.ndarray:
    """Both hands translate by ``offset`` and turn by ``world_rotation`` over ``steps`` frames."""
    frames = [start]
    rotvec = Rotation.from_matrix(world_rotation).as_rotvec()
    for k in range(1, steps + 1):
        fraction = k / steps
        delta = Rotation.from_rotvec(rotvec * fraction).as_matrix()
        sides = []
        for tcp in (LEFT, RIGHT):
            matrix = gripper_frame.pose9_to_matrix(start[tcp])
            sides.append(_pose9(delta @ matrix[:3, :3], matrix[:3, 3] + np.asarray(offset) * fraction))
        frames.append(_vector(sides[0], sides[1], grippers=(0.5 + 0.4 * fraction, 0.5 - 0.4 * fraction)))
    return np.stack(frames)


@pytest.mark.parametrize("world_yaw_deg", [0.0, 35.0, -170.0])
def test_yaw_recovers_pico_world_heading(world_yaw_deg):
    estimate = replay_agent.estimate_yaw(_umi_start(world_yaw_deg), _home())
    # Pico-world -> arm yaw undoes the world's own turn.
    for value in (estimate.left, estimate.right, estimate.mean):
        assert np.degrees(replay_agent._wrap(value + np.radians(world_yaw_deg))) == pytest.approx(0.0, abs=1e-6)
    assert estimate.spread == pytest.approx(0.0, abs=1e-9)
    assert estimate.max_elevation_deg == pytest.approx(20.0)


def test_yaw_mean_splits_disagreement_across_the_wrap():
    estimate = replay_agent.estimate_yaw(_umi_start(175.0, right_extra_yaw_deg=10.0), _home())
    assert np.degrees(estimate.spread) == pytest.approx(10.0)
    # left = -175, right = +175 (i.e. -185): the mean is -180, not 0.
    assert abs(np.degrees(estimate.mean)) == pytest.approx(180.0)


@pytest.mark.parametrize("world_yaw_deg", [0.0, 60.0])
def test_plan_preserves_motion_in_the_arm_frame(world_yaw_deg):
    world = _rot("z", world_yaw_deg)
    # Physical motion (arm axes): 10 cm forward, 5 cm up, 30 deg about the vertical.
    offset, turn = np.array([0.10, 0.0, 0.05]), _rot("z", 30)
    start = _umi_start(world_yaw_deg)
    states = _moved(start, world @ offset, turn, steps=10)
    home = _home()

    yaw = replay_agent.estimate_yaw(start, home).mean
    plan = replay_agent.build_plan(_episode(states), home, yaw)

    assert plan.shape == (10, 20)
    for tcp in (LEFT, RIGHT):
        home_matrix = gripper_frame.pose9_to_matrix(home[tcp])
        last = gripper_frame.pose9_to_matrix(plan[-1, tcp])
        np.testing.assert_allclose(last[:3, 3], home_matrix[:3, 3] + offset, atol=1e-9)
        np.testing.assert_allclose(last[:3, :3], turn @ home_matrix[:3, :3], atol=1e-9)
    # Grippers pass through.
    np.testing.assert_allclose(plan[:, [9, 19]], states[1:, [9, 19]])


def test_first_target_is_one_source_step_from_home():
    states = _moved(_umi_start(), [0.10, 0.0, 0.0], np.eye(3), steps=10)
    home = _home()
    plan = replay_agent.build_plan(_episode(states), home, 0.0)
    np.testing.assert_allclose(plan[0, LEFT][:3] - home[LEFT][:3], [0.01, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(plan[0, LEFT][3:], home[LEFT][3:], atol=1e-9)


def test_speed_scale_stretches_time_and_keeps_the_path():
    states = _moved(_umi_start(), [0.10, 0.0, 0.05], _rot("z", 30), steps=10)
    home = _home()
    episode = _episode(states)
    full = replay_agent.build_plan(episode, home, 0.0)
    slow = replay_agent.build_plan(episode, home, 0.0, speed_scale=0.25)

    assert len(slow) == 40
    np.testing.assert_allclose(slow[3::4], full, atol=1e-9)
    # Halfway between two source frames: half the step, and still a valid rotation.
    np.testing.assert_allclose(slow[1, LEFT][:3] - home[LEFT][:3], [0.005, 0.0, 0.0025], atol=1e-9)
    rotation = gripper_frame.pose9_to_matrix(slow[1, LEFT])[:3, :3]
    np.testing.assert_allclose(rotation, _rot("z", 1.5) @ gripper_frame.pose9_to_matrix(home[LEFT])[:3, :3], atol=1e-9)


def test_speed_scale_always_ends_on_the_last_frame():
    states = _moved(_umi_start(), [0.10, 0.0, 0.0], np.eye(3), steps=10)
    episode = _episode(states)
    full = replay_agent.build_plan(episode, _home(), 0.0)
    slow = replay_agent.build_plan(episode, _home(), 0.0, speed_scale=0.3)
    np.testing.assert_allclose(slow[-1], full[-1], atol=1e-9)


@pytest.mark.parametrize("speed_scale", [0.0, -1.0, 1.5])
def test_speed_scale_out_of_range_is_rejected(speed_scale):
    states = _moved(_umi_start(), [0.10, 0.0, 0.0], np.eye(3), steps=10)
    with pytest.raises(ValueError, match="speed_scale"):
        replay_agent.build_plan(_episode(states), _home(), 0.0, speed_scale=speed_scale)


def test_agent_hands_out_the_plan_then_holds():
    states = _moved(_umi_start(), [0.10, 0.0, 0.0], np.eye(3), steps=3)
    home = _home()
    agent = replay_agent.UmiReplayAgent(_episode(states))
    agent.reset()
    agent.warmup({"state": home})
    assert not agent.done

    targets = [agent.get_action({"state": home})["actions"] for _ in range(3)]
    assert agent.done
    assert targets[0].dtype == np.float32
    np.testing.assert_allclose(targets[-1][LEFT][:3] - home[LEFT][:3], [0.10, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(agent.get_action({"state": home})["actions"], targets[-1])

    agent.reset()
    assert not agent.done


def test_agent_rejects_tracker_relocalization():
    states = _moved(_umi_start(), [0.10, 0.0, 0.0], np.eye(3), steps=10)
    states[5:, LEFT.start] += 0.5
    agent = replay_agent.UmiReplayAgent(_episode(states))
    with pytest.raises(ValueError, match="relocalization"):
        agent.warmup({"state": _home()})


def test_agent_rejects_inconsistent_start_unless_yaw_is_given():
    states = _moved(_umi_start(right_extra_yaw_deg=40.0), [0.10, 0.0, 0.0], np.eye(3), steps=10)
    with pytest.raises(ValueError, match="disagree"):
        replay_agent.UmiReplayAgent(_episode(states)).warmup({"state": _home()})
    replay_agent.UmiReplayAgent(_episode(states), yaw_deg=0.0).warmup({"state": _home()})


def test_agent_rejects_vertical_fingertips_unless_yaw_is_given():
    start = _umi_start()
    start[LEFT] = _pose9(_rot("y", 80), [0.3, 0.2, -0.4])
    states = _moved(start, [0.10, 0.0, 0.0], np.eye(3), steps=10)
    with pytest.raises(ValueError, match="too steep"):
        replay_agent.UmiReplayAgent(_episode(states)).warmup({"state": _home()})
    replay_agent.UmiReplayAgent(_episode(states), yaw_deg=0.0).warmup({"state": _home()})


def test_agent_rejects_targets_too_far_from_home():
    states = _moved(_umi_start(), [0.80, 0.0, 0.0], np.eye(3), steps=40)
    with pytest.raises(ValueError, match="from home"):
        replay_agent.UmiReplayAgent(_episode(states)).warmup({"state": _home()})


def test_agent_saves_plan_and_measured_states(tmp_path):
    states = _moved(_umi_start(), [0.10, 0.0, 0.0], np.eye(3), steps=3)
    home = _home()
    agent = replay_agent.UmiReplayAgent(_episode(states))
    agent.warmup({"state": home})
    for _ in range(2):
        agent.get_action({"state": home})
    agent.save_trajectory(tmp_path / "logs" / "run.npz")

    saved = np.load(tmp_path / "logs" / "run.npz")
    assert saved["plan"].shape == (3, 20)
    assert saved["states"].shape == (2, 20)
    assert saved["times"].shape == (2,)
    np.testing.assert_allclose(saved["home"], home)
