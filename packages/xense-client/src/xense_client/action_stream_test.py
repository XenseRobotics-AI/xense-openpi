"""Observer-on/off equivalence, horizons, RTC blending and buffered progress."""

import threading

import numpy as np
import pytest

from xense_client.action_chunk_broker import ActionChunkBroker
from xense_client.action_queue import ActionQueue
from xense_client.action_stream import ACTION_REF_KEY
from xense_client.action_stream import ActionStream
from xense_client.action_stream import SubscriptionClosedError
from xense_client.base_policy import BasePolicy
from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker
from xense_client.visualization import ObservedEnvironment
from xense_client.visualization_brokers import ObservedActionChunkBroker
from xense_client.visualization_brokers import ObservedRTCActionChunkBroker
from xense_client.visualization_brokers import _ObservedActionQueue


class Policy(BasePolicy):
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": self.values.copy(), "metadata": "unchanged"}


def test_horizon_and_enabled_disabled_control_equivalence():
    values = np.arange(40, dtype=np.float32).reshape(10, 4)
    baseline = ActionChunkBroker(Policy(values), action_horizon=3)
    stream = ActionStream()
    chunks, plans = stream.subscribe_chunks(), stream.subscribe_plans()
    observed = ObservedActionChunkBroker(Policy(values), action_horizon=3, action_stream=stream)
    for _ in range(8):
        expected, actual = baseline.infer({}), observed.infer({})
        assert actual["actions"].tobytes() == expected["actions"].tobytes()
        assert actual["metadata"] == expected["metadata"]
        assert ACTION_REF_KEY not in expected
        ref = actual[ACTION_REF_KEY]
        stream.dispatched(ref)
        plan = plans.latest()
        assert len(plan.refs) <= 2
        assert len(plan.actions) == len(plan.refs)
    assert baseline._policy.calls == observed._policy.policy.calls == 3
    arrival = chunks.latest()
    assert arrival.actions.shape == (10, 4)
    assert arrival.usable_steps == 3
    assert stream.has_new_chunk(2)
    assert not stream.has_new_chunk(3)
    with pytest.raises(ValueError, match="WRITEABLE"):
        arrival.actions.setflags(write=True)
    stream.close()
    frozen_sequence = stream.status().latest_chunk_sequence
    for _ in range(6):
        actual = observed.infer({})
        assert ACTION_REF_KEY not in actual
    assert chunks.closed
    assert plans.closed
    assert stream.status().latest_chunk_sequence == frozen_sequence


@pytest.mark.parametrize("rtc", [True, False])
@pytest.mark.parametrize("blend", [0, 2])
def test_rtc_queue_math_is_identical_and_paced_buffer_survives_replace(rtc, blend):
    stream = ActionStream()
    plans = stream.subscribe_plans(capacity=1)
    baseline = ActionQueue(rtc_enabled=rtc, blend_steps=blend)
    observed = _ObservedActionQueue(rtc_enabled=rtc, blend_steps=blend, action_stream=stream)
    first = np.arange(24, dtype=np.float64).reshape(6, 4)
    chunks = stream.subscribe_chunks()
    baseline.merge(first, first, 0)
    observed.merge(first, first, 0)
    refs = plans.latest().refs
    # Popped into PacedBroker but not dispatched to the environment yet.
    for _ in range(2):
        np.testing.assert_array_equal(baseline.get(), observed.get())
    second = first + 100
    baseline.merge(second, second, 1, action_index_before_inference=0)
    observed.merge(second, second, 1, action_index_before_inference=0)
    arrival = chunks.latest()
    new_refs = tuple((arrival.generation, arrival.sequence, i) for i in range(len(second)))
    assert observed.queue.tobytes() == baseline.queue.tobytes()
    assert observed.original_queue.tobytes() == baseline.original_queue.tobytes()
    plan = plans.latest()
    assert plan.refs[:2] == refs[:2]
    np.testing.assert_array_equal(plan.actions[:2], first[:2])
    np.testing.assert_array_equal(plan.actions[2:], observed.queue)
    if rtc:
        assert plan.refs[2:] == new_refs[2:]  # Actual consumption delay, not estimate=1.
    stream.dispatched(refs[0])
    assert plans.latest().refs[0] == refs[1]
    for _ in range(baseline.qsize()):
        expected = baseline.get()
        actual = observed.get()
        ref = observed.last_ref
        assert actual.tobytes() == expected.tobytes()
        stream.dispatched(ref)
    stream.dispatched(refs[1])
    assert plans.latest().refs == ()
    assert stream.status().errors == 0


def test_environment_reports_only_successful_dispatch_and_strips_tag():
    stream = ActionStream()
    plans = stream.subscribe_plans()
    actions = np.ones((1, 7))  # Public plumbing has no Elite/20D assumption.
    refs = stream.chunk_received(actions)
    stream.plan_updated(actions, refs)
    stream.popped(refs[0], actions[0])

    class Env:
        fail = True

        def apply_action(self, action):
            assert set(action) == {"actions"}
            if self.fail:
                raise RuntimeError("submission failed")

    env = Env()
    wrapped = ObservedEnvironment(env, stream)
    action = {"actions": actions[0], ACTION_REF_KEY: refs[0]}
    with pytest.raises(RuntimeError, match="submission failed"):
        wrapped.apply_action(action)
    assert len(plans.latest().refs) == 1
    env.fail = False
    wrapped.apply_action(action)
    assert plans.latest().refs == ()


def test_invalid_display_data_is_isolated_and_reset_rejects_old_generation():
    stream = ActionStream()
    plans = stream.subscribe_plans(capacity=1)
    assert stream.chunk_received(None) is None
    assert stream.status().errors == 1
    refs = stream.chunk_received(np.ones((2, 3)))
    stream.plan_updated(np.ones((2, 3)), refs)
    stream.reset()
    stream.popped(refs[0], np.ones(3))
    assert plans.latest().refs == ()
    stream.close()
    assert stream.subscribe_chunks().closed


def test_rtc_background_warmup_and_returned_actions_are_unchanged():
    class RtcPolicy(Policy):
        def infer(self, obs, **kwargs):
            return super().infer(obs)

    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    stream = ActionStream()
    chunks = stream.subscribe_chunks()
    baseline = RTCActionChunkBroker(RtcPolicy(values), action_queue_size_to_get_new_actions=0)
    observed = ObservedRTCActionChunkBroker(RtcPolicy(values), action_queue_size_to_get_new_actions=0, action_stream=stream)
    try:
        for index in range(3):
            expected, actual = baseline.infer({"state": np.zeros(4)}), observed.infer({"state": np.zeros(4)})
            assert expected["actions"].tobytes() == actual["actions"].tobytes()
            assert actual[ACTION_REF_KEY][-1] == index
            assert ACTION_REF_KEY not in expected
            stream.dispatched(actual[ACTION_REF_KEY])
        assert baseline._policy.calls == observed._policy.calls == 2
        assert chunks.latest().sequence == 1  # First JIT result never entered the control queue.
        assert stream.status().errors == 0
    finally:
        baseline.stop()
        observed.stop()


def test_closing_a_subscription_wakes_a_blocked_consumer():
    stream = ActionStream()
    subscription = stream.subscribe_chunks()
    stopped = threading.Event()

    def consume():
        try:
            subscription.get()
        except SubscriptionClosedError:
            stopped.set()

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    stream.close()
    assert stopped.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


@pytest.mark.parametrize("action_dim", [7, 14, 20, 24])
@pytest.mark.parametrize("horizon", [1, 3])
def test_observed_broker_preserves_robot_independent_actions(action_dim, horizon):
    values = np.arange(8 * action_dim, dtype=np.float64).reshape(8, action_dim)
    base_policy, observed_policy = Policy(values), Policy(values)
    baseline = ActionChunkBroker(base_policy, action_horizon=horizon)
    stream = ActionStream()
    observed = ObservedActionChunkBroker(observed_policy, action_horizon=horizon, action_stream=stream)
    for _ in range(9):
        expected, actual = baseline.infer({}), observed.infer({})
        ref = actual.pop(ACTION_REF_KEY)
        np.testing.assert_array_equal(actual["actions"], expected["actions"])
        assert actual["metadata"] == expected["metadata"]
        stream.dispatched(ref)
    assert base_policy.calls == observed_policy.calls
    observed.reset()
    baseline.reset()
    np.testing.assert_array_equal(observed.infer({})["actions"], baseline.infer({})["actions"])


def test_shared_brokers_do_not_import_visualization_or_viewer():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import sys
from xense_client.action_chunk_broker import ActionChunkBroker
from xense_client.action_queue import ActionQueue
from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker
assert not any(name in sys.modules for name in (
    'xense_client.action_stream', 'xense_client.visualization',
    'xense_client.visualization_brokers', 'xense_motion_viewer', 'rerun',
))
"""], check=True)
