"""The normalized action/proprio space the phase-two actor and critic work in.

The MLP heads see the same representation the frozen VLA was trained on: TCP
dims as deltas from the current state, grippers absolute, everything
quantile-normalized with the VLA checkpoint's own statistics - i.e. the
``DeltaActions`` + ``Normalize`` steps of the VLA data pipeline, redone here in
JAX so they run batched and differentiably inside the losses. On top of that:

- normalized values are clipped to ``[-1, 1]``, the range the actor can emit, so
  critic inputs and BC targets never hold values the actor cannot produce;
- grippers are clipped to their physical ``[0, 1]``;
- rot6d blocks (first two rotation-matrix columns) are Gram-Schmidt
  orthonormalized; a degenerate block (zero, near-collinear or non-finite
  columns) falls back to the current state's rotation.

``canonicalize(a, s) = encode(decode(a, s), s)`` maps an actor output onto what
execution would re-encode to, so the critic scores the action the robot runs.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import openpi.transforms as _transforms

# Each contiguous run of delta dims holds one or more arms as xyz + rot6d.
_ARM_WIDTH = 9
_EPS = 1e-6
_COLLINEAR_EPS = 1e-3


@dataclasses.dataclass(frozen=True)
class ActionSpace:
    state_q01: np.ndarray
    state_q99: np.ndarray
    action_q01: np.ndarray
    action_q99: np.ndarray
    # True where the executed value is state + delta.
    delta_mask: np.ndarray
    gripper_dims: tuple[int, ...]
    rot6d_blocks: tuple[tuple[int, int], ...]

    @property
    def state_dim(self) -> int:
        return len(self.state_q01)

    @property
    def action_dim(self) -> int:
        return len(self.action_q01)

    @classmethod
    def from_data_config(cls, data_config) -> ActionSpace:
        """Bind to a VLA data config whose ``norm_stats`` come from the VLA checkpoint."""
        if not data_config.use_quantile_norm:
            raise ValueError("RLT phase two needs a VLA trained with quantile normalization (pi05).")
        stats = data_config.norm_stats
        if stats is None or not {"state", "actions"} <= stats.keys():
            raise ValueError("VLA norm stats must cover 'state' and 'actions'.")
        state, actions = stats["state"], stats["actions"]
        masks = [t.mask for t in data_config.data_transforms.inputs if isinstance(t, _transforms.DeltaActions)]
        if len(masks) != 1 or masks[0] is None:
            raise ValueError("RLT phase two needs a VLA data config with exactly one DeltaActions mask.")
        delta_mask = np.zeros(len(actions.q01), dtype=bool)
        delta_mask[: len(masks[0])] = masks[0]
        return cls(
            state_q01=np.asarray(state.q01, np.float32),
            state_q99=np.asarray(state.q99, np.float32),
            action_q01=np.asarray(actions.q01, np.float32),
            action_q99=np.asarray(actions.q99, np.float32),
            delta_mask=delta_mask,
            gripper_dims=tuple(int(i) for i in np.flatnonzero(~delta_mask)),
            rot6d_blocks=_rot6d_blocks(delta_mask),
        )

    def normalize_state(self, state: jax.Array) -> jax.Array:
        return jnp.clip(_normalize(state[..., : self.state_dim], self.state_q01, self.state_q99), -1.0, 1.0)

    def encode(self, actions: jax.Array, state: jax.Array) -> jax.Array:
        """Executed (absolute) actions ``(..., T, A)`` -> normalized ``[-1, 1]`` space."""
        base = self._base(state)
        absolute = self._project(actions, base)
        return jnp.clip(_normalize(absolute - base * self.delta_mask, self.action_q01, self.action_q99), -1.0, 1.0)

    def decode(self, normalized: jax.Array, state: jax.Array) -> jax.Array:
        """Normalized actions -> absolute actions the robot executes."""
        base = self._base(state)
        absolute = _unnormalize(normalized, self.action_q01, self.action_q99) + base * self.delta_mask
        return self._project(absolute, base)

    def canonicalize(self, normalized: jax.Array, state: jax.Array) -> jax.Array:
        return self.encode(self.decode(normalized, state), state)

    def diagnose(self, actions: np.ndarray, state: np.ndarray, *, normalized: bool = False) -> dict[str, int]:
        """What ``encode`` (or, for ``normalized`` actor output, ``decode``) silently corrects (for logging).

        ``out_of_range``: normalized values beyond the actor's [-1, 1], i.e. outside the VLA's
        q01/q99; ``gripper_clips``: openings outside [0, 1]; ``rot6d_fallbacks``: degenerate
        rotations replaced by the current pose.
        """
        actions = np.asarray(actions, np.float32)
        base = np.asarray(state, np.float32)[..., None, : self.action_dim]
        if normalized:
            actions = _unnormalize(actions, self.action_q01, self.action_q99) + base * self.delta_mask
        grippers = actions[..., list(self.gripper_dims)]
        fallbacks = sum(
            int((~np.asarray(_gram_schmidt(jnp.asarray(actions[..., a:b]))[1])).sum()) for a, b in self.rot6d_blocks
        )
        projected = np.asarray(self._project(jnp.asarray(actions), jnp.asarray(base)))
        normalized = _normalize(projected - base * self.delta_mask, self.action_q01, self.action_q99)
        return {
            "out_of_range": int((np.abs(normalized) > 1.0).sum()),
            "gripper_clips": int(((grippers < 0) | (grippers > 1)).sum()),
            "rot6d_fallbacks": fallbacks,
        }

    def _base(self, state: jax.Array) -> jax.Array:
        """Current state as a ``(..., 1, A)`` broadcast against action chunks."""
        return jax.lax.stop_gradient(state[..., None, : self.action_dim])

    def _project(self, absolute: jax.Array, base: jax.Array) -> jax.Array:
        grippers = np.zeros(self.action_dim, dtype=bool)
        grippers[list(self.gripper_dims)] = True
        out = jnp.where(grippers, jnp.clip(absolute, 0.0, 1.0), absolute)
        for start, end in self.rot6d_blocks:
            fallback = jnp.broadcast_to(base[..., start:end], out[..., start:end].shape)
            out = out.at[..., start:end].set(_orthonormalize(out[..., start:end], fallback))
        return out


def _normalize(x, q01, q99):
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _unnormalize(x, q01, q99):
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _rot6d_blocks(delta_mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    blocks = []
    padded = np.concatenate([[False], delta_mask, [False]]).astype(int)
    for start, end in zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1), strict=True):
        if (end - start) % _ARM_WIDTH:
            raise ValueError(f"Delta dims {start}..{end} are not whole xyz+rot6d arms.")
        blocks += [(int(arm) + 3, int(arm) + _ARM_WIDTH) for arm in range(start, end, _ARM_WIDTH)]
    return tuple(blocks)


def _gram_schmidt(rot6d: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Orthonormalize the two columns; also return whether the input was well-conditioned."""
    v1, v2 = rot6d[..., :3], rot6d[..., 3:]
    n1 = jnp.linalg.norm(v1, axis=-1, keepdims=True)
    u1 = v1 / jnp.maximum(n1, _EPS)
    r2 = v2 - jnp.sum(v2 * u1, axis=-1, keepdims=True) * u1
    n2 = jnp.linalg.norm(r2, axis=-1, keepdims=True)
    ok = jnp.isfinite(rot6d).all(axis=-1, keepdims=True)
    ok &= (n1 > _EPS) & (n2 > jnp.maximum(_EPS, _COLLINEAR_EPS * jnp.linalg.norm(v2, axis=-1, keepdims=True)))
    return jnp.concatenate([u1, r2 / jnp.maximum(n2, _EPS)], axis=-1), ok


def _orthonormalize(rot6d: jax.Array, fallback: jax.Array) -> jax.Array:
    """Orthonormalized ``rot6d``; degenerate rows take the (orthonormalized) fallback, else identity."""
    safe = jnp.where(jnp.isfinite(rot6d), rot6d, 0.0)
    ortho, ok = _gram_schmidt(safe)
    fallback_ortho, fallback_ok = _gram_schmidt(jnp.where(jnp.isfinite(fallback), fallback, 0.0))
    identity = jnp.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], rot6d.dtype)
    return jnp.where(ok, ortho, jnp.where(fallback_ok, fallback_ortho, identity))
