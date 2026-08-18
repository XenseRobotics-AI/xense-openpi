from collections import deque

import numpy as np


class LatencyTracker:
    """Tracks recent latencies and provides max/percentile queries.

    Args:
        maxlen (int | None): Optional sliding window size. If provided, only the
            most recent ``maxlen`` latencies are kept. If ``None``, keeps all.
    """

    def __init__(self, maxlen: int = 100):
        self._values = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        """Clear all recorded latencies."""
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        """Add a latency sample (seconds)."""
        # Ensure numeric and non-negative
        val = float(latency)

        if val < 0:
            return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float | None:
        """Return the maximum latency or None if empty."""
        return self.max_latency

    def percentile(self, q: float) -> float | None:
        """Return the q-quantile (q in [0,1]) of recorded latencies or None if empty."""
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            return self.max_latency
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float | None:
        """Return the 95th percentile latency or None if empty."""
        return self.percentile(0.95)
