"""Offline tests for the synchronous Runtime — focused on the shutdown path.

The existing Runtime predates this test module; this file covers the
request_stop() addition introduced for graceful SIGINT handling.
"""

import threading
import time

from xense_client.runtime import agent as _agent
from xense_client.runtime import environment as _environment
from xense_client.runtime.runtime import Runtime


class FakeEnv(_environment.Environment):
    """Never auto-completes; resets are counted to check episode skipping."""

    def __init__(self, obs_period_s: float = 0.033):
        self.obs_period_s = obs_period_s
        self._reset_count = 0
        self.applied: list[dict] = []

    def reset(self) -> None:
        self._reset_count += 1

    def is_episode_complete(self) -> bool:
        return False

    def get_observation(self) -> dict:
        time.sleep(self.obs_period_s)
        return {"step": len(self.applied)}

    def apply_action(self, action: dict) -> None:
        self.applied.append(action)


class FakeAgent(_agent.Agent):
    def get_action(self, observation: dict) -> dict:
        return {"actions": observation["step"]}

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# request_stop() exits run() promptly and skips subsequent episodes
# ---------------------------------------------------------------------------
def test_request_stop_ends_run_promptly():
    env = FakeEnv(obs_period_s=1.0 / 30)
    runtime = Runtime(
        environment=env,
        agent=FakeAgent(),
        subscribers=[],
        max_hz=30.0,
        num_episodes=3,
        max_episode_steps=0,
    )

    done = threading.Event()

    def _run():
        runtime.run()
        done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Let the runtime get going.
    time.sleep(0.3)

    t0 = time.time()
    runtime.request_stop()
    finished = done.wait(timeout=2.0)
    elapsed = time.time() - t0

    assert finished, "runtime.run() did not return within 2 s after request_stop"
    assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s"

    # 1 episode reset + 1 final reset in run() = 2 total. Episodes 2 and 3
    # were skipped because _stop_requested was set.
    assert env._reset_count == 2, f"expected 1 episode + 1 final reset = 2, got {env._reset_count}"
