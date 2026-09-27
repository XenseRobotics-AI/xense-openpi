"""The robot-side RLT session against the server-side collector, with fake hardware."""

import numpy as np
import pytest

from examples.bi_flexiv_rizon4_rt import rlt_mode
from openpi.rlt import collector_test
from openpi.rlt import replay as _replay

C = collector_test.C


class FakeEnv:
    def __init__(self):
        self.t = 0
        self.applied = []

    def reset(self):
        pass

    def get_observation(self):
        state = np.zeros(20, np.float32)
        state[3], state[7], state[12], state[16] = 1, 1, 1, 1
        return {"state": state, "images": {"t": np.asarray(self.t)}}

    def apply_action(self, action):
        self.applied.append(np.asarray(action["actions"]))
        self.t += 1


class FakeController:
    """Buttons by env step (pressed before that step executes); takeover over a step range."""

    def __init__(self, env, buttons, takeover=()):
        self.env, self.buttons, self.takeover = env, dict(buttons), takeover
        self.was_active = False
        self.release = False
        self.gate_presses = 0

    def set_takeover_motion(self, motion):
        self.motion = motion

    def reset_for_new_episode(self):
        self.was_active = False

    def poll_buttons(self):
        pass

    def consume_button_events(self):
        return [self.buttons.pop(self.env.t)] if self.env.t in self.buttons else []

    def poll_and_decide(self, gripper_command=None):
        active = self.env.t in self.takeover
        self.release = self.was_active and not active
        self.was_active = active
        return active

    def consume_release_event(self):
        release, self.release = self.release, False
        return release

    def get_override_action(self):
        return np.full(20, 0.5, np.float32)


class Direct:
    """The server's env seam, calling the robot session in-process instead of over a socket."""

    def __init__(self, session):
        self.session = session

    def request(self, message):
        return getattr(self.session, message["op"])(message)

    def status(self, text):
        pass


def _run(buttons, takeover=()):
    env = FakeEnv()
    controller = FakeController(env, {0: "A", **buttons}, takeover)
    session = rlt_mode.Session(env, controller, step_dt=None, takeover_motion=lambda *motion: motion)
    collector, _, _ = collector_test._setup([])
    collector.env = Direct(session)
    return collector.run_round(), env


def test_labeled_phase_with_a_takeover_round_trips():
    # t=0: A at the gate starts the round. B at t=1 (chunk 1) opens the window at the next
    # boundary, t=4. A takeover over t=6..9 finishes chunk 2 and runs a 2-step continuation;
    # releasing at t=10 ends the reply. B at t=13 labels success, reported when chunk 3 completes
    # at t=14: a 10-step phase. A at t=18 ends the round with an empty segment.
    result, env = _run({1: "B", 13: "B", 18: "A"}, takeover=range(6, 10))
    rows = result["rows"]
    assert result["metrics"]["success"] == 1
    assert env.t == 18
    # C=4, stride 2 over 10 phase steps: windows at phase steps 0, 2, 4, 6 (t = 4, 6, 8, 10).
    assert [int(row["curr_obs"]["z_rl"][0]) for row in rows] == [4, 6, 8, 10]
    assert [int(row["next_obs"]["z_rl"][0]) for row in rows] == [8, 10, 12, 14]
    assert [row["terminated"] for row in rows] == [False, False, False, True]
    np.testing.assert_array_equal(rows[-1]["chunk_rewards"], [0, 0, 0, 1])
    human = [np.flatnonzero(row["intervention_mask"]).tolist() for row in rows]
    assert human == [[2, 3], [0, 1, 2, 3], [0, 1], []]
    for row in rows:
        np.testing.assert_array_equal(row["action_source"] == _replay.SOURCE_HUMAN, row["intervention_mask"])


def test_window_needs_a_label_and_discard_drops_labeled_data():
    # Window opens at t=4 and is labeled failure (Y at t=6, reported at t=8); X at t=9 discards; A ends.
    result, _ = _run({1: "B", 6: "Y", 9: "X", 13: "A"})
    assert result["rows"] == []
    assert result["metrics"]["failure"] == 1
    assert result["metrics"]["discards"] == 1


def test_robot_refuses_the_actor_outside_a_window():
    env = FakeEnv()
    session = rlt_mode.Session(env, FakeController(env, {}), step_dt=None, takeover_motion=lambda *m: m)
    with pytest.raises(RuntimeError, match="outside an open recording window"):
        session.chunk({"actions": np.zeros((C, 20)), "source": "actor"})
    assert env.applied == []


def test_usage_tally_separates_routing_from_driving():
    env = FakeEnv()
    session = rlt_mode.Session(env, FakeController(env, {}, takeover=range(1, 3)), step_dt=None, takeover_motion=tuple)
    session.recording = True
    session.chunk({"actions": np.zeros((C, 20)), "source": "actor"})
    assert (session.tally.actor_chunks, session.tally.actor_steps, session.tally.overridden_steps) == (1, 1, 2)
