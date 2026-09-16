"""Latent Tactile Predictor (LTP): the training-only head that reads future tactile
latents out of one intermediate layer of the action expert.

Design (docs/action-conditioned-tactile-pretraining.md, section 2.3.3): a fixed
table of ``K x S`` learnable queries (``K`` horizons, ``S`` pads, query =
``e_k + p_s``) cross-attends once into the residual stream ``h^(m)`` of the
action-token positions, goes through one MLP, and is projected linearly to the
whitened target dimension. One layer on purpose: the head is meant to be a
read-out, not a forward model of its own -- if it could compute the future
tactile state from scratch, blocks 1..m would feel no pressure to carry it.

The head never runs at inference; ``Pi0TactileFastVit.sample_actions`` does not
touch it.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class TactileFuturePredictor(nnx.Module):
    """One-block cross-attention read-out: ``[b, t, width] -> [b, K, S, out_dim]``."""

    def __init__(
        self,
        *,
        width: int,
        num_horizons: int,
        num_pads: int,
        out_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_horizons = num_horizons
        self.num_pads = num_pads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = out_dim

        inner = num_heads * head_dim
        init = nnx.initializers.normal(stddev=0.02)
        # Additive query factorisation: 5 horizon vectors + 4 pad vectors instead of
        # 20 free queries, so horizons share "how far ahead" and pads share "which gel".
        self.horizon_embed = nnx.Param(init(rngs.params(), (num_horizons, width), jnp.float32))
        self.pad_embed = nnx.Param(init(rngs.params(), (num_pads, width), jnp.float32))

        self.kv_norm = nnx.RMSNorm(width, rngs=rngs)
        self.q_proj = nnx.Linear(width, inner, rngs=rngs)
        self.k_proj = nnx.Linear(width, inner, rngs=rngs)
        self.v_proj = nnx.Linear(width, inner, rngs=rngs)
        self.o_proj = nnx.Linear(inner, width, rngs=rngs)

        self.mlp_norm = nnx.RMSNorm(width, rngs=rngs)
        self.mlp_in = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(mlp_dim, width, rngs=rngs)

        self.out_norm = nnx.RMSNorm(width, rngs=rngs)
        self.out_proj = nnx.Linear(width, out_dim, rngs=rngs)

    @property
    def num_queries(self) -> int:
        return self.num_horizons * self.num_pads

    @at.typecheck
    def __call__(
        self,
        hidden: at.Float[at.Array, "b t d"],
        key_mask: at.Bool[at.Array, "b t"],
    ) -> at.Float[at.Array, "b k s z"]:
        # The residual stream arrives in the expert's compute dtype (bf16); the head
        # is small, run it in fp32.
        hidden = hidden.astype(jnp.float32)
        batch = hidden.shape[0]

        query = self.horizon_embed[:, None, :] + self.pad_embed[None, :, :]  # [K, S, d]
        query = query.reshape(1, self.num_queries, -1)
        query = jnp.broadcast_to(query, (batch, *query.shape[1:]))

        kv = self.kv_norm(hidden)
        q = self.q_proj(query).reshape(batch, self.num_queries, self.num_heads, self.head_dim)
        k = self.k_proj(kv).reshape(batch, hidden.shape[1], self.num_heads, self.head_dim)
        v = self.v_proj(kv).reshape(batch, hidden.shape[1], self.num_heads, self.head_dim)

        logits = jnp.einsum("bqhd,bthd->bhqt", q, k) * (self.head_dim**-0.5)
        logits = jnp.where(key_mask[:, None, None, :], logits, jnp.finfo(logits.dtype).min)
        probs = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhqt,bthd->bqhd", probs, v).reshape(batch, self.num_queries, -1)
        u = query + self.o_proj(attended)

        h = self.mlp_in(self.mlp_norm(u))
        u = u + self.mlp_out(jax.nn.gelu(h))

        z = self.out_proj(self.out_norm(u))
        return z.reshape(batch, self.num_horizons, self.num_pads, self.out_dim)


__all__ = ["TactileFuturePredictor"]
