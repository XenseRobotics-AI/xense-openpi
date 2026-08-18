import abc


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""

    def warmup(self, obs: dict) -> None:
        """Pre-warm the policy before the episode control loop starts.

        Override in subclasses that benefit from pre-compilation (e.g. JAX JIT).
        Default is a no-op.

        Args:
            obs: A real observation from the environment, used to trigger
                 inference so JIT compilation finishes before the loop.
        """
