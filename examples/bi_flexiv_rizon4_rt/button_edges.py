"""Debounced input edges. A new/recovered input must first be released."""

import time


class ButtonEdges:
    def __init__(self, debounce_s=0.05):
        self.debounce_s = debounce_s
        self.reset()

    def reset(self):
        self._states = {}

    def update(self, name, pressed, now=None):
        now = time.monotonic() if now is None else now
        state = self._states.get(name)
        if state is None:
            self._states[name] = [pressed, now, pressed, False]
            return False
        if pressed != state[0]:
            state[0], state[1] = pressed, now
        if now - state[1] < self.debounce_s:
            return False
        if not pressed:
            state[3] = True
        changed = pressed != state[2]
        state[2] = pressed
        if changed and pressed and state[3]:
            state[3] = False
            return True
        return False


class StreamHealth:
    """Watch packet progress using local elapsed time, not device clock offset."""

    def __init__(self, timeout_s=0.5):
        self.timeout_s = timeout_s
        self._stamp = None
        self._changed_at = None

    def update(self, stamp, now=None):
        now = time.monotonic() if now is None else now
        if stamp <= 0:
            raise RuntimeError("Pico4 has no valid packet timestamp")
        if self._stamp is not None and stamp < self._stamp:
            raise RuntimeError("Pico4 packet clock restarted; reconnect and resynchronize TCP")
        if stamp != self._stamp:
            self._stamp, self._changed_at = stamp, now
        elif now - self._changed_at >= self.timeout_s:
            raise RuntimeError(f"Pico4 input stale for {now - self._changed_at:.3f}s; aborting round")
