"""The Q gate: when the critic may steer the actor.

The actor's loss is ``-q_weight * Q1 + bc_weight * BC``; early in training the
critic knows nothing and following it hurts. The gate holds the warmup weights
until the critic provably ranks corrective actions above the VLA proposal, on
recent replay windows:

- rank accuracy: fraction of intervened windows with Q1(executed) > Q1(VLA proposal);
- margin: median of (Q1(executed) - Q1(proposal)) / batch-Q-std - accuracy can pass on
  noise-thin differences, the margin cannot;
- clean accuracy: fraction of clean windows with Q1(executed) > Q1(random) - a critic
  that hates every action also fails here.

Opening requires every threshold to hold for ``sustain_evals`` consecutive
evaluations; once open, any metric breaching its (looser) close threshold for
``close_evals`` consecutive evaluations closes the gate again. The asymmetry is
deliberate: slow to open, quick to close.
"""

from __future__ import annotations

import dataclasses

from openpi.rlt import config as _config


@dataclasses.dataclass
class QGate:
    """The gate's state machine; pure Python over per-round metric dicts."""

    config: _config.QGateConfig
    open: bool = False
    evals: int = 0
    consecutive_passes: int = 0
    consecutive_failures: int = 0
    # Actor-update index at opening; the weight ramp clocks from here.
    opened_at_update: int | None = None

    def observe(self, metrics: dict[str, float], actor_update: int) -> str | None:
        """One evaluation; returns "opened"/"closed" on a transition, else None.

        ``metrics`` needs rank_accuracy, margin_std, clean_accuracy and success_windows.
        """
        cfg = self.config
        self.evals += 1
        passed = (
            metrics["success_windows"] >= cfg.min_success_windows
            and metrics["rank_accuracy"] >= cfg.min_rank_accuracy
            and metrics["margin_std"] >= cfg.min_margin_std
            and metrics["clean_accuracy"] >= cfg.min_clean_accuracy
        )
        failed = (
            metrics["rank_accuracy"] < cfg.close_rank_accuracy
            or metrics["margin_std"] < cfg.close_margin_std
            or metrics["clean_accuracy"] < cfg.close_clean_accuracy
        )
        if not self.open:
            self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
            if self.consecutive_passes >= cfg.sustain_evals:
                self.open = True
                self.opened_at_update = actor_update
                self.consecutive_failures = 0
                return "opened"
            return None
        self.consecutive_failures = self.consecutive_failures + 1 if failed else 0
        if self.consecutive_failures >= cfg.close_evals:
            self.open = False
            self.opened_at_update = None
            self.consecutive_passes = 0
            return "closed"
        return None

    def state_dict(self) -> dict:
        return dataclasses.asdict(self)

    def load_state_dict(self, state: dict) -> None:
        self.open = bool(state["open"])
        self.evals = int(state["evals"])
        self.consecutive_passes = int(state["consecutive_passes"])
        self.consecutive_failures = int(state["consecutive_failures"])
        self.opened_at_update = state["opened_at_update"]
