import abc


class Agent(abc.ABC):
    """An Agent is the thing with agency, i.e. the entity that makes decisions.

    Agents receive observations about the state of the world, and return actions
    to take in response.
    """

    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        """Query the agent for the next action."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the agent to its initial state."""

    def warmup(self, observation: dict) -> None:
        """Pre-warm the agent before the episode control loop starts.

        Called once after reset() and before the first get_action() call.
        Default is a no-op; override in subclasses that need JIT warm-up.

        Args:
            observation: A real observation from the environment.
        """
