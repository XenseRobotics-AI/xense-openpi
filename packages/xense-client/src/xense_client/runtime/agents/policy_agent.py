from typing import override

from xense_client import base_policy as _base_policy
from xense_client.runtime import agent as _agent


class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy

    @override
    def get_action(self, observation: dict) -> dict:
        return self._policy.infer(observation)

    def reset(self) -> None:
        self._policy.reset()

    @override
    def warmup(self, observation: dict) -> None:
        self._policy.warmup(observation)
