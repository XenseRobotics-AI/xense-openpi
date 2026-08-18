"""Two-thread runtime: obs at camera FPS, actions at a higher pace.

The default :class:`Runtime` runs `get_obs -> get_action -> apply_action`
strictly serially in one thread. When `get_obs` is pinned to the camera frame
rate (e.g. 30 Hz on this stack), chunk actions are consumed at that same
rate — capping robot motion to teleop pace.

This runtime splits that loop in two:

* **obs thread**  — runs at ~camera FPS. Calls ``env.get_observation()`` and
  feeds the latest obs to a :class:`PacedBroker` so the broker's inference
  always sees fresh state.
* **action thread** — runs at ``action_hz``. Pops the next action from the
  broker's queue and calls ``env.apply_action(...)``. Subscribers are also
  invoked here, so recorded ``(obs, action)`` pairs reflect actions that were
  actually sent.

Trade-offs / non-goals:

* The obs paired with each step in subscribers is the latest snapshot; it
  may have been observed up to one camera period (~33 ms) earlier than the
  action was sent. Acceptable for recording / monitoring; not suitable as a
  feedback signal to a tightly-coupled controller.
* Pico4 intervention is not handled here — switching control between policy
  and teleop while a producer thread is asynchronously filling the action
  queue requires extra coordination (queue drain on handoff). Use the
  synchronous :class:`Runtime` for intervention.
"""

import threading
import time

from xense_client.logger import get_logger
from xense_client.paced_broker import PacedBroker
from xense_client.runtime import environment as _environment
from xense_client.runtime import subscriber as _subscriber

logger = get_logger("DecoupledRuntime")


class DecoupledRuntime:
    """Runs obs and action loops on separate threads, paced independently.

    Args:
        environment: Wrapped environment (must be thread-safe for concurrent
            ``get_observation()`` and ``apply_action()``; both Flexiv RT
            envs are — observations read from one SHM region, actions write
            another, and the cameras have their own background threads).
        broker: PacedBroker wrapping the underlying chunked-policy broker.
        subscribers: Notified on each action sent (in the action thread).
        obs_hz: Target obs loop frequency. Will be clamped by camera FPS;
            pass 0 to run as fast as ``get_observation`` returns.
        action_hz: Target action emission frequency. Must be > 0.
        num_episodes: How many episodes to run.
        max_episode_steps: Optional cap on action-thread steps per episode
            (0 = no cap).
    """

    def __init__(
        self,
        environment: _environment.Environment,
        broker: PacedBroker,
        subscribers: list[_subscriber.Subscriber],
        obs_hz: float,
        action_hz: float,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        if action_hz <= 0:
            raise ValueError(f"action_hz must be > 0, got {action_hz}")
        self._environment = environment
        self._broker = broker
        self._subscribers = subscribers
        self._obs_hz = obs_hz
        self._action_hz = action_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._action_steps = 0
        # Set by request_stop() to prevent subsequent episodes from starting
        # after a SIGINT mid-episode. Checked between episodes in run().
        self._stop_requested = False

        # Latest obs snapshot — written by obs thread, read by action thread
        # for subscriber pairing. Single-slot, last-writer-wins.
        self._latest_obs: dict | None = None
        self._obs_lock = threading.Lock()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._stop_requested = False
        for _ in range(self._num_episodes):
            if self._stop_requested:
                break
            self._run_episode()
        # Final reset for the real env to return to home position.
        self._environment.reset()

    def mark_episode_complete(self) -> None:
        self._in_episode = False
        self._stop_event.set()

    def request_stop(self) -> None:
        """Request a graceful runtime shutdown.

        Ends the current episode (waking the main thread + signalling the
        action/obs threads via _stop_event) and prevents subsequent episodes
        from starting. Intended for SIGINT handlers — letting run() return
        naturally so the caller's finally clause can disconnect cleanly
        (including the home-position move via env.disconnect()).
        """
        self._stop_requested = True
        self.mark_episode_complete()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def _run_episode(self) -> None:
        logger.info("Starting decoupled episode...")
        self._environment.reset()
        # Reset broker BEFORE starting the producer thread. PacedBroker.reset
        # also drains its internal action queue and forwards to the inner
        # broker, so any stale chunk from the previous episode is dropped.
        self._broker.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        # Pre-warmup the policy with a real obs so JIT compilation /
        # first-inference latency happens before the action thread starts.
        logger.info("Pre-warming broker (JIT compilation)...")
        warmup_obs = self._environment.get_observation()
        self._broker.warmup(warmup_obs)
        with self._obs_lock:
            self._latest_obs = warmup_obs

        # Prime the broker's input slot and start its producer thread so the
        # first action is ready before the action thread tries to pop.
        self._broker.submit_obs(warmup_obs)
        self._broker.start()

        self._in_episode = True
        self._action_steps = 0
        self._stop_event.clear()

        obs_thread = threading.Thread(target=self._obs_loop, name="obs-loop", daemon=True)
        action_thread = threading.Thread(target=self._action_loop, name="action-loop", daemon=True)
        obs_thread.start()
        action_thread.start()

        # Wait for either thread to signal episode end.
        try:
            while self._in_episode and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)
                if self._environment.is_episode_complete():
                    self.mark_episode_complete()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — stopping episode")
            self.mark_episode_complete()

        # Shutdown order: stop broker producer first so action thread doesn't
        # pop new actions, then join both threads.
        self._broker.stop()
        action_thread.join(timeout=2.0)
        obs_thread.join(timeout=2.0)

        logger.info(f"Episode completed (action_steps={self._action_steps}).")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------
    def _obs_loop(self) -> None:
        """Read observations and forward them to the broker.

        Naturally rate-limited by camera FPS; ``obs_hz`` is a soft cap.
        """
        period = 1.0 / self._obs_hz if self._obs_hz > 0 else 0.0
        last_t = time.time()
        while self._in_episode and not self._stop_event.is_set():
            try:
                obs = self._environment.get_observation()
            except Exception as e:
                logger.error(f"get_observation failed: {e}")
                time.sleep(0.05)
                continue

            with self._obs_lock:
                self._latest_obs = obs
            self._broker.submit_obs(obs)

            if period > 0:
                now = time.time()
                dt = now - last_t
                if dt < period:
                    time.sleep(period - dt)
                    last_t = time.time()
                else:
                    last_t = now

    def _action_loop(self) -> None:
        """Pop actions from the broker queue and execute them at action_hz."""
        period = 1.0 / self._action_hz
        next_tick = time.time()

        while self._in_episode and not self._stop_event.is_set():
            try:
                # Bounded wait so we notice shutdown promptly even if the
                # producer hasn't put anything yet.
                action = self._broker.pop_action(timeout=0.5)
            except Exception:
                # queue.Empty after timeout — just loop and re-check stop flag.
                continue

            t_send_start = time.time()
            try:
                self._environment.apply_action(action)
            except Exception as e:
                logger.error(f"apply_action failed: {e}")
                self.mark_episode_complete()
                break

            with self._obs_lock:
                obs_snapshot = self._latest_obs
            if obs_snapshot is not None:
                for subscriber in self._subscribers:
                    try:
                        subscriber.on_step(obs_snapshot, action)
                    except Exception as e:
                        logger.warning(f"subscriber.on_step raised: {e}")

            self._action_steps += 1
            if self._max_episode_steps > 0 and self._action_steps >= self._max_episode_steps:
                self.mark_episode_complete()
                break
            if self._environment.is_episode_complete():
                self.mark_episode_complete()
                break

            # Pace.
            next_tick += period
            now = time.time()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind — log occasionally and reset the schedule so we
                # don't try to burst-catch-up forever after a stall.
                if self._action_steps % 60 == 0:
                    logger.debug(
                        f"action loop behind by {-sleep_for*1000:.1f} ms "
                        f"(send_action={((time.time()-t_send_start)*1000):.1f} ms)"
                    )
                next_tick = now
