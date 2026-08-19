"""Offline tests for DecoupledRuntime.

Uses fake env / policy / subscriber so the tests run without robot or server.
The fake env simulates a camera-bound get_observation (slow) so we can verify
that action emission decouples from the obs loop.
"""

import threading
import time

import pytest

from xense_client import base_policy
from xense_client.paced_broker import PacedBroker
from xense_client.runtime import environment as _environment
from xense_client.runtime import subscriber as _subscriber
from xense_client.runtime.decoupled_runtime import DecoupledRuntime

# Tolerance margin for rate assertions: threading is noisy, especially on
# loaded CI hosts. 25 % is a sensible loose bound for short test windows.
_RATE_TOL = 0.25


class FakeEnv(_environment.Environment):
    """Camera-bound env: get_observation sleeps to mimic a fixed FPS."""

    def __init__(self, obs_period_s: float = 0.033, episode_complete_after: int | None = None):
        self.obs_period_s = obs_period_s
        self._step = 0
        self._reset_count = 0
        self._episode_complete_after = episode_complete_after
        self._lock = threading.Lock()
        self.applied_actions: list[dict] = []
        self._action_timestamps: list[float] = []
        self._obs_timestamps: list[float] = []

    def reset(self) -> None:
        with self._lock:
            self._reset_count += 1
            self._step = 0

    def is_episode_complete(self) -> bool:
        if self._episode_complete_after is None:
            return False
        with self._lock:
            return len(self.applied_actions) >= self._episode_complete_after

    def get_observation(self) -> dict:
        time.sleep(self.obs_period_s)
        with self._lock:
            self._step += 1
            obs = {"step": self._step}
            self._obs_timestamps.append(time.time())
        return obs

    def apply_action(self, action: dict) -> None:
        with self._lock:
            self.applied_actions.append(action)
            self._action_timestamps.append(time.time())


class FakePolicy(base_policy.BasePolicy):
    """Echoes the obs step value back as the action; fast inference."""

    def __init__(self):
        self.infer_count = 0

    def infer(self, obs: dict) -> dict:
        self.infer_count += 1
        return {"actions": obs["step"]}


class RecordingSubscriber(_subscriber.Subscriber):
    def __init__(self):
        self.episode_starts = 0
        self.episode_ends = 0
        self.steps: list[tuple] = []  # (obs, action)

    def on_episode_start(self) -> None:
        self.episode_starts += 1

    def on_step(self, observation: dict, action: dict) -> None:
        self.steps.append((observation, action))

    def on_episode_end(self) -> None:
        self.episode_ends += 1


# ---------------------------------------------------------------------------
# T1: action frequency matches action_hz, decoupled from obs FPS
# ---------------------------------------------------------------------------
def test_action_rate_matches_action_hz():
    # Camera-bound obs at 30 Hz; action at 60 Hz; episode ends after 60 actions.
    # Expected wall time at 60 Hz: ~1.0 s.
    env = FakeEnv(obs_period_s=1.0 / 30, episode_complete_after=60)
    inner = FakePolicy()
    broker = PacedBroker(inner=inner, queue_size=50)
    sub = RecordingSubscriber()
    runtime = DecoupledRuntime(
        environment=env,
        broker=broker,
        subscribers=[sub],
        obs_hz=30.0,
        action_hz=60.0,
        num_episodes=1,
        max_episode_steps=0,
    )
    runtime.run()

    # Episode completed → 60 actions applied.
    assert len(env.applied_actions) >= 60, f"only {len(env.applied_actions)} actions"

    # Action rate over the bulk of the run (skip first/last few to avoid warmup/teardown noise).
    ts = env._action_timestamps[5:-2]
    duration = ts[-1] - ts[0]
    rate = (len(ts) - 1) / duration
    expected = 60.0
    assert abs(rate - expected) / expected < _RATE_TOL, f"action rate {rate:.1f} Hz, expected ~{expected} Hz"

    # Obs rate should be ~30 Hz (camera-bound).
    obs_ts = env._obs_timestamps[2:-2]
    obs_duration = obs_ts[-1] - obs_ts[0]
    obs_rate = (len(obs_ts) - 1) / obs_duration
    assert abs(obs_rate - 30.0) / 30.0 < _RATE_TOL, f"obs rate {obs_rate:.1f} Hz, expected ~30 Hz"

    # Sanity: action rate is meaningfully faster than obs rate.
    assert rate > obs_rate * 1.5, f"action_rate={rate:.1f}, obs_rate={obs_rate:.1f}"


# ---------------------------------------------------------------------------
# T2: subscribers see the action that was actually sent, paired with latest obs
# ---------------------------------------------------------------------------
def test_subscriber_pairs_action_with_latest_obs():
    env = FakeEnv(obs_period_s=1.0 / 30, episode_complete_after=30)
    broker = PacedBroker(inner=FakePolicy(), queue_size=50)
    sub = RecordingSubscriber()
    runtime = DecoupledRuntime(
        environment=env,
        broker=broker,
        subscribers=[sub],
        obs_hz=30.0,
        action_hz=60.0,
    )
    runtime.run()

    assert sub.episode_starts == 1
    assert sub.episode_ends == 1
    assert len(sub.steps) >= 30, f"subscriber got {len(sub.steps)} steps"
    # Each recorded action should match an actual applied action.
    applied_values = {a["actions"] for a in env.applied_actions}
    recorded_values = {a["actions"] for _, a in sub.steps}
    # Subscriber records every action it sees; we just want overlap, not
    # exact equality (subscriber may have skipped the very last action that
    # triggered episode complete before the on_step call).
    assert recorded_values.issubset(applied_values) or len(recorded_values & applied_values) >= len(sub.steps) - 1


# ---------------------------------------------------------------------------
# T3: max_episode_steps caps the action thread
# ---------------------------------------------------------------------------
def test_max_episode_steps_cap():
    env = FakeEnv(obs_period_s=1.0 / 30)  # never auto-completes
    broker = PacedBroker(inner=FakePolicy(), queue_size=20)
    runtime = DecoupledRuntime(
        environment=env,
        broker=broker,
        subscribers=[],
        obs_hz=30.0,
        action_hz=60.0,
        max_episode_steps=20,
    )
    runtime.run()
    assert len(env.applied_actions) == 20, f"got {len(env.applied_actions)}"


# ---------------------------------------------------------------------------
# T4: reset is called once per episode + a final reset on exit
# ---------------------------------------------------------------------------
def test_reset_counts():
    env = FakeEnv(obs_period_s=1.0 / 30, episode_complete_after=15)
    broker = PacedBroker(inner=FakePolicy(), queue_size=20)
    runtime = DecoupledRuntime(
        environment=env,
        broker=broker,
        subscribers=[],
        obs_hz=30.0,
        action_hz=60.0,
        num_episodes=2,
    )
    runtime.run()
    # 2 episode resets + 1 final reset = 3
    assert env._reset_count == 3, f"reset_count={env._reset_count}"


# ---------------------------------------------------------------------------
# T5: action_hz must be > 0
# ---------------------------------------------------------------------------
def test_action_hz_validation():
    with pytest.raises(ValueError, match="action_hz must be > 0"):
        DecoupledRuntime(
            environment=FakeEnv(),
            broker=PacedBroker(inner=FakePolicy()),
            subscribers=[],
            obs_hz=30.0,
            action_hz=0.0,
        )


# ---------------------------------------------------------------------------
# T6: shutdown happens promptly when episode completes mid-action
# ---------------------------------------------------------------------------
def test_prompt_shutdown_on_episode_complete():
    env = FakeEnv(obs_period_s=1.0 / 30, episode_complete_after=10)
    broker = PacedBroker(inner=FakePolicy(), queue_size=50)
    runtime = DecoupledRuntime(
        environment=env,
        broker=broker,
        subscribers=[],
        obs_hz=30.0,
        action_hz=60.0,
    )
    t0 = time.time()
    runtime.run()
    elapsed = time.time() - t0
    # 10 actions @ 60 Hz = 0.17 s. Plus startup overhead and the runtime's
    # 0.5 s poll on episode-complete from the main thread. Total < ~1.5 s.
    assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# T7: request_stop() ends the current episode promptly and prevents the
# next one from starting (the SIGINT path).
# ---------------------------------------------------------------------------
def test_request_stop_ends_run_promptly():
    # 3-episode run that would otherwise loop forever (no auto-complete).
    env = FakeEnv(obs_period_s=1.0 / 30)
    broker = PacedBroker(inner=FakePolicy(), queue_size=20)
    runtime = DecoupledRuntime(
        environment=env,
        broker=broker,
        subscribers=[],
        obs_hz=30.0,
        action_hz=60.0,
        num_episodes=3,
        max_episode_steps=0,
    )

    # Run in a thread so we can request_stop from outside.
    done = threading.Event()

    def _run():
        runtime.run()
        done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Let the runtime get going.
    time.sleep(0.3)

    # Trigger graceful shutdown.
    t0 = time.time()
    runtime.request_stop()
    finished = done.wait(timeout=2.0)
    elapsed = time.time() - t0

    assert finished, "runtime.run() did not return within 2 s after request_stop"
    assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s"

    # Only the first episode should have run — the next two were skipped.
    # FakeEnv's reset is called once per started episode + once for the
    # final env reset in run(), so total resets == 1 + 1 = 2.
    assert env._reset_count == 2, f"expected 1 episode + 1 final reset = 2 env resets, got {env._reset_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
