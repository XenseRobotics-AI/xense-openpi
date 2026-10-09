"""Replay of labeled C-step transitions (numpy ring buffer, uniform sampling).

A row is one window cut from a labeled critical phase (see ``critical_trace``):
observations at both ends, the C executed actions in normalized space, per-step
rewards, and provenance. ``curr_ref_chunk`` is the raw VLA reference even at
intervened steps; ``td.training_reference`` swaps in the human actions to build
the BC target (the actor's reference input stays the raw VLA reference).
"""

import copy

import numpy as np

OBS_KEYS = ("z_rl", "state", "proprio", "ref_chunk")
# Per-step action provenance; a chunk mixing two or more of them is MIXED.
SOURCE_VLA, SOURCE_ACTOR, SOURCE_HUMAN, SOURCE_MIXED = 0, 1, 2, 3


def chunk_source(action_source: np.ndarray) -> int:
    """One provenance code for a whole window: its only per-step source, or MIXED."""
    kinds = np.unique(np.asarray(action_source))
    if kinds.size == 0:
        raise ValueError("action_source marks no executed step.")
    return SOURCE_MIXED if kinds.size > 1 else int(kinds[0])


class ReplayBuffer:
    """Fixed-capacity FIFO of transitions; sampling is uniform with replacement."""

    def __init__(
        self,
        capacity: int,
        *,
        z_dim: int,
        state_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int,
        seed: int = 0,
    ):
        obs = {
            "z_rl": ((z_dim,), np.float32),
            "state": ((state_dim,), np.float32),
            "proprio": ((state_dim,), np.float32),
            "ref_chunk": ((ref_num_action_chunks, action_dim), np.float32),
        }
        chunk = (num_action_chunks,)
        self._fields = {
            **{f"{side}_{key}": spec for side in ("curr", "next") for key, spec in obs.items()},
            "actions": ((*chunk, action_dim), np.float32),
            "chunk_rewards": (chunk, np.float32),
            "intervention_mask": (chunk, np.bool_),
            "action_source": (chunk, np.int8),
            # Labeled success/failure end: no bootstrap.
            "terminated": ((), np.bool_),
            # Provenance, for logging only.
            "success": ((), np.bool_),
            "actor_enabled": ((), np.bool_),
            "episode_id": ((), np.int64),
            "round_id": ((), np.int64),
            "source": ((), np.int8),
            # Server wall clock (Unix s) when the labeled phase became rows; shared by the phase.
            "timestamp": ((), np.float64),
        }
        self.capacity = capacity
        self._storage = {key: np.zeros((capacity, *shape), dtype) for key, (shape, dtype) in self._fields.items()}
        self._size = 0
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def prepare(self, row: dict) -> dict[str, np.ndarray]:
        """Validate and convert a row without touching the buffer (so a round commits all-or-nothing)."""
        flat = {key: row[key] for key in self._fields if key in row}
        for side in ("curr", "next"):
            flat.update({f"{side}_{key}": row[f"{side}_obs"][key] for key in OBS_KEYS})
        if missing := set(self._fields) - set(flat):
            raise ValueError(f"Transition is missing {sorted(missing)}.")
        out = {}
        for key, (shape, dtype) in self._fields.items():
            value = np.asarray(flat[key], dtype=dtype)
            if value.shape != shape:
                raise ValueError(f"Transition field {key!r} has shape {value.shape}, expected {shape}.")
            out[key] = value
        if not all(np.isfinite(out[key]).all() for key in ("actions", "chunk_rewards", "curr_z_rl", "next_z_rl")):
            raise ValueError("Transition holds non-finite values.")
        if not np.array_equal(out["action_source"] == SOURCE_HUMAN, out["intervention_mask"]):
            raise ValueError("action_source and intervention_mask disagree.")
        if out["source"] != chunk_source(out["action_source"]):
            raise ValueError("source and action_source disagree.")
        return out

    def add(self, row: dict) -> None:
        for key, value in self.prepare(row).items():
            self._storage[key][self._cursor] = value
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """``min(batch_size, len(self))`` rows as ``{"curr_obs": {...}, "next_obs": {...}, <fields>}``."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = self._rng.choice(self._size, size=min(batch_size, self._size), replace=True)
        batch: dict = {"curr_obs": {}, "next_obs": {}}
        for key, array in self._storage.items():
            side, _, obs_key = key.partition("_")
            if side in ("curr", "next") and obs_key in OBS_KEYS:
                batch[f"{side}_obs"][obs_key] = array[indices]
            else:
                batch[key] = array[indices]
        return batch

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "cursor": self._cursor,
            "storage": {key: value[: self._size].copy() for key, value in self._storage.items()},
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity or set(state["storage"]) != set(self._storage):
            raise ValueError("Replay capacity or fields differ from the checkpoint.")
        size = state["size"]
        for key, target in self._storage.items():
            value = state["storage"][key]
            if value.shape != (size, *target.shape[1:]) or value.dtype != target.dtype:
                raise ValueError(f"Replay field {key!r} differs from the checkpoint layout.")
            target[:size] = value
        self._size, self._cursor = size, state["cursor"]
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = copy.deepcopy(state["rng"])
