"""Use the real observation adapter with an already sampled robot observation."""

from types import SimpleNamespace

import numpy as np

from examples.bi_elite_cs66_rt.env import BiEliteCS66RTEnvironment
from examples.bi_elite_cs66_rt.real_env import STATE_KEYS
from examples.bi_elite_cs66_rt.real_env import BiEliteCS66RTRealEnv


def make_environment(listener=None):
    calls = []
    raw = {key: float(i) for i, key in enumerate(STATE_KEYS)}
    for key in (
        "head",
        "left_wrist",
        "right_wrist",
        "left_tactile_left",
        "left_tactile_right",
        "right_tactile_left",
        "right_tactile_right",
    ):
        raw[key] = np.full((12, 20, 3), 73, dtype=np.uint8)

    def observe():
        calls.append("observe")
        return raw

    env = BiEliteCS66RTRealEnv.__new__(BiEliteCS66RTRealEnv)
    env.robot = SimpleNamespace(get_observation=observe)
    env.observation_listener = listener
    wrapper = BiEliteCS66RTEnvironment.__new__(BiEliteCS66RTEnvironment)
    wrapper._env, wrapper._render_height, wrapper._render_width = env, 8, 8
    return wrapper, raw, calls


def test_image_tap_preserves_policy_input_and_reads_robot_only_once():
    captured = []
    enabled, raw, calls = make_environment(captured.append)
    disabled, _, baseline_calls = make_environment()
    baseline = disabled.get_observation()
    observed = enabled.get_observation()
    assert calls == baseline_calls == ["observe"]
    assert captured[0] is raw
    assert captured[0]["left_tactile_left"] is raw["left_tactile_left"]
    assert set(observed) == {"state", "images"}
    assert set(observed["images"]) == {"head", "left_wrist", "right_wrist"}
    np.testing.assert_array_equal(observed["state"], baseline["state"])
    for key in observed["images"]:
        np.testing.assert_array_equal(observed["images"][key], baseline["images"][key])
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            assert np.all(value == 73), key


def test_display_callback_failure_is_detached_without_changing_observation():
    def broken(_):
        raise RuntimeError("display unavailable")

    env, raw, calls = make_environment(broken)
    result = env.get_observation()
    assert calls == ["observe"]
    assert env._env.observation_listener is None
    assert result["state"].shape == (20,)
    assert "left_tactile_left" not in result["images"]
    assert raw["left_tactile_left"].shape == (12, 20, 3)
