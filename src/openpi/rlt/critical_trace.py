"""Executed-step timeline of one critical phase, cut into sliding C-step windows.

While a critical phase (a recording window) runs, every executed step is
appended whoever drove it - VLA reference, actor, or human takeover. Features
(``z_rl``, ``state``, ``proprio``, ``ref_chunk``) are known at execution-unit
starts; raw observations are kept every ``stride`` steps. Once the operator
labels the phase success or failure, windows start every ``stride`` steps from
the phase start, plus one aligned to the phase end, which is the terminal
transition. The collector computes features for any window end still missing
them from the kept observations (``missing_feature_indices`` / ``observation``)
and then calls ``windows``. Unlabeled phases are dropped.
"""

import dataclasses

import numpy as np

from openpi.rlt import replay as _replay


@dataclasses.dataclass(frozen=True)
class Window:
    start: int
    features: dict[str, np.ndarray]
    # Executed actions in absolute (execution) space, (C, A).
    executed: np.ndarray
    rewards: np.ndarray
    human: np.ndarray
    source: np.ndarray
    next_features: dict[str, np.ndarray]
    terminal: bool
    actor_enabled: bool


class CriticalTrace:
    def __init__(self, horizon: int, stride: int):
        if horizon < 1 or stride < 1:
            raise ValueError(f"horizon and stride must be >= 1, got {horizon}, {stride}.")
        self.horizon = horizon
        self.stride = stride
        self._executed: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._source: list[int] = []
        self._actor_enabled: list[bool] = []
        self._features: dict[int, dict[str, np.ndarray]] = {}
        self._observations: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self._rewards)

    def extend(self, executed: np.ndarray, rewards: np.ndarray, source: np.ndarray, *, actor_enabled: bool) -> None:
        """Append one executed segment; ``source`` is VLA / actor / human per step (``replay.SOURCE_*``)."""
        if not len(executed) == len(rewards) == len(source):
            raise ValueError("Segment executed/rewards/source lengths differ.")
        self._executed.extend(np.asarray(executed, np.float32))
        self._rewards.extend(float(r) for r in rewards)
        self._source.extend(int(s) for s in source)
        self._actor_enabled.extend([actor_enabled] * len(rewards))

    def add_features(self, index: int, features: dict[str, np.ndarray]) -> None:
        """Features observed after ``index`` executed steps (the state step ``index`` starts from)."""
        self._features[index] = features

    def add_observation(self, index: int, observation: dict) -> None:
        """Keep a raw observation so features at ``index`` can be computed after labeling."""
        self._observations[index] = observation

    def set_terminal_reward(self, reward: float) -> None:
        """The label's reward lands on the last executed step."""
        if not self._rewards:
            raise ValueError("A terminal label needs at least one executed step.")
        self._rewards[-1] = float(reward)

    def anchors(self) -> list[int]:
        """Window starts: every ``stride`` steps, plus the one ending exactly at the phase end."""
        last = len(self) - self.horizon
        if last < 0:
            return []
        anchors = list(range(0, last + 1, self.stride))
        return anchors if anchors[-1] == last else [*anchors, last]

    def missing_feature_indices(self) -> list[int]:
        needed = {i for a in self.anchors() for i in (a, a + self.horizon)}
        return sorted(needed - self._features.keys())

    def has_observation(self, index: int) -> bool:
        return index in self._observations

    def observation(self, index: int) -> dict:
        if index not in self._observations:
            raise ValueError(f"No kept observation for step {index}; stride capture missed it.")
        return self._observations[index]

    def windows(self) -> list[Window]:
        """All windows of the labeled phase; the one ending at the phase end is terminal."""
        if missing := self.missing_feature_indices():
            raise ValueError(f"Features missing at steps {missing}.")
        end = len(self)
        return [
            Window(
                start=a,
                features=self._features[a],
                executed=np.stack(self._executed[a : a + self.horizon]),
                rewards=np.asarray(self._rewards[a : a + self.horizon], np.float32),
                human=np.asarray(self._source[a : a + self.horizon]) == _replay.SOURCE_HUMAN,
                source=np.asarray(self._source[a : a + self.horizon], np.int8),
                next_features=self._features[a + self.horizon],
                terminal=a + self.horizon == end,
                actor_enabled=self._actor_enabled[a],
            )
            for a in self.anchors()
        ]
