"""No-hardware coverage for the XTac-UMI environment's previous-state pairing."""

import numpy as np
import pytest

from examples.xtac_umi_bi_flexiv_rizon4_rt import env as _env


class _FakeRealEnv:
    def __init__(self):
        self.step = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_observation(self):
        self.step += 1
        image = np.zeros((48, 64, 3), np.uint8)
        return {"qpos": np.full(20, self.step, np.float32), "images": {"left_wrist": image, "right_wrist": image}}


def _environment(*, send_previous_state):
    environment = object.__new__(_env.XtacUmiBiFlexivRizon4RTEnvironment)
    environment._env = _FakeRealEnv()
    environment._render_height = environment._render_width = 32
    environment._send_previous_state = send_previous_state
    environment._previous_state = None
    return environment


def test_default_state_is_one_frame():
    assert _environment(send_previous_state=False).get_observation()["state"].shape == (20,)


def test_previous_state_pairs_consecutive_control_steps():
    environment = _environment(send_previous_state=True)
    environment.reset()
    first = environment.get_observation()["state"]
    second = environment.get_observation()["state"]
    third = environment.get_observation()["state"]

    assert first.shape == (2, 20)
    # The first observation of an episode pairs the state with itself, as training does.
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(second[0], first[1])
    np.testing.assert_array_equal(third[0], second[1])
    assert third[1, 0] == 3


def test_reset_starts_a_new_pair():
    environment = _environment(send_previous_state=True)
    environment.reset()
    environment.get_observation()
    environment.get_observation()
    environment.reset()
    state = environment.get_observation()["state"]
    np.testing.assert_array_equal(state[0], state[1])


def _main():
    # main.py pulls in the xense-client brokers, which need dm-tree.
    pytest.importorskip("tree")
    from examples.xtac_umi_bi_flexiv_rizon4_rt import main

    return main


def test_frame_relative_poses_refuse_the_gripper_change_of_basis():
    _main = globals()["_main"]()
    with pytest.raises(SystemExit, match="no-align-gripper-frames"):
        _main.check_args(_main.Args(frame_relative_poses=True, align_gripper_frames=True))
    _main.check_args(_main.Args(frame_relative_poses=True, align_gripper_frames=False))
    _main.check_args(_main.Args())


def test_frame_relative_run_file_is_consistent():
    import yaml

    _main = globals()["_main"]()

    run = yaml.safe_load((_main.RUNS_DIR / "frame-relative.yaml").read_text())
    args = _main.Args(**run)
    _main.check_args(args)
    assert args.frame_relative_poses
    assert not args.align_gripper_frames
