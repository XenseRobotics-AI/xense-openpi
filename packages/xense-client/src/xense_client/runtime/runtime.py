import threading
import time

from xense_client.logger import get_logger
from xense_client.runtime import agent as _agent
from xense_client.runtime import environment as _environment
from xense_client.runtime import subscriber as _subscriber

logger = get_logger("Runtime")


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
        # Set by request_stop() so a SIGINT during episode N doesn't kick off
        # episode N+1. Checked between episodes in run().
        self._stop_requested = False

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        self._stop_requested = False
        for _ in range(self._num_episodes):
            if self._stop_requested:
                break
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def request_stop(self) -> None:
        """Request a graceful runtime shutdown.

        Ends the current episode and prevents subsequent episodes from
        starting. Intended for SIGINT handlers — letting run() return
        naturally so the caller's finally clause can disconnect cleanly
        (including the home-position move).
        """
        self._stop_requested = True
        self.mark_episode_complete()

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logger.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        # Pre-warmup: get a real observation and trigger JIT compilation
        # BEFORE the control loop starts, so the first _step() has no delay.
        # For non-RTC policies this is a no-op (Agent.warmup() default).
        logger.info("Pre-warming agent (JIT compilation)...")
        warmup_obs = self._environment.get_observation()
        self._agent.warmup(warmup_obs)
        logger.info("Agent warmed up. Starting control loop.")

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time

            # logger.info(f"dt={dt*1000:.2f}ms")

            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        logger.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """A single step of the runtime loop."""
        import time

        t0 = time.time()
        observation = self._environment.get_observation()
        t1 = time.time()

        action = self._agent.get_action(observation)
        t2 = time.time()

        self._environment.apply_action(action)
        t3 = time.time()

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)
        t4 = time.time()

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()

        # every 20 steps print the detailed timing
        # if self._episode_steps % 20 == 0:
        logger.info(
            f"⏱️  Runtime _step_time: "
            f"get_observation={((t1-t0)*1000):.2f}ms | "
            f"get_action={((t2-t1)*1000):.2f}ms | "
            f"apply_action={((t3-t2)*1000):.2f}ms | "
            f"subscriber={((t4-t3)*1000):.2f}ms | "
            f"total_time={((t4-t0)*1000):.2f}ms"
        )
