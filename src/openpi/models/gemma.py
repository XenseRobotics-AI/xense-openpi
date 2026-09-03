# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax

# Private JAX API (pinned jax 0.5.3): the cuDNN fused-attention fwd/bwd rules that
# jax.nn.dot_product_attention(implementation="cudnn") wires into its custom_vjp.
# Used by _cudnn_attention_in_dtype to run the float16 kernel with loss scaling
# without recomputing the forward inside the backward.
from jax._src.cudnn import fused_attention_stablehlo as _cudnn_fa
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        elif modulation.ndim != x.ndim:
            raise ValueError(
                f"Adaptive RMSNorm expected cond rank {x.ndim - 1} or {x.ndim}, got modulation shape {modulation.shape}"
            )
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


def _stop_gradient_for_fully_masked_queries(q, attn_mask):
    """Cut fully masked query rows out of the backward pass, keeping the mask intact.

    Padding tokens are neither valid queries nor valid keys, so make_attn_mask
    produces query rows that are entirely false, and cuDNN 9.14 returned NaN
    q-gradients for them at the production shape. Zeroing the gradient of those rows
    fixes that without touching attn_mask.

    The alternative -- opening a dummy key for each empty row -- is equivalent
    numerically (measured bit-identical q/k/v-gradients on valid rows at the real
    training shape) but rebuilds the whole (B, 1, T, S) mask inside every layer,
    which costs about 0.3 s/step at batch 256. Prefer this one.

    Neither variant is the cure for the 2026-08-30 divergence, which is still open:
    the kernel itself is correct, but cuDNN's answer differs from the explicit path
    by about 2% per step and this recipe has no margin for that. Do not spend time on
    the mask again. See docs/2026-08-30-cudnn-attention-divergence.md.
    """
    query_has_key = jnp.any(attn_mask, axis=-1)[:, 0, :, None, None]
    return jnp.where(query_has_key, q, jax.lax.stop_gradient(q))


def _cudnn_attention_call(q, k, v, attn_mask):
    """The plain cuDNN fused attention call used by training (q is pre-scaled)."""
    q = _stop_gradient_for_fully_masked_queries(q, attn_mask)
    return jax.nn.dot_product_attention(q, k, v, mask=attn_mask, scale=1.0, implementation="cudnn")


def _cudnn_attention_in_dtype(q, k, v, attn_mask, compute_dtype):
    """Run cuDNN fused attention with q/k/v cast to ``compute_dtype`` (e.g. float16).

    Motivation: the flash-attention backward forms ``dS = P * (dP - rowsum(dO * O))``
    from the *rounded* stored output ``O``. For peaked attention rows (attention
    sinks) ``dP`` and ``rowsum(dO * O)`` nearly cancel, so the rounding error of
    ``O`` is a large, sign-consistent fraction of ``dS`` -- a bias the explicit
    path does not have because it forms ``sum(P * dP)`` from the same rounded
    ``dP``. float16 keeps 3 more mantissa bits than bfloat16, shrinking that bias
    about 8x, but has a much smaller exponent range, so the incoming cotangent is
    rescaled to a power of two near 1 (dynamic loss scaling) before the fused
    backward and unscaled in float32 afterwards. Forward values are unchanged up
    to the output rounding; q/k/v are exactly representable in float16 unless
    they exceed its range, which the probe script checks.
    """
    out_dtype = q.dtype

    # The forward and backward call JAX's cuDNN fwd/bwd rules directly (the same
    # functions jax.nn.dot_product_attention's own custom_vjp uses) instead of
    # re-running the forward through jax.vjp inside the backward: the softmax
    # stats and the output are kept as residuals, so with flax remat the backward
    # costs one recomputed forward plus one fused backward, like the bf16 path.
    # attn_mask is an explicit argument (not a closure) so that the backward,
    # which flax remat traces separately, does not capture a leaked tracer.
    @jax.custom_vjp
    def attention(q, k, v, attn_mask):
        return attention_fwd(q, k, v, attn_mask)[0]

    def attention_fwd(q, k, v, attn_mask):
        qc, kc, vc = (x.astype(compute_dtype) for x in (q, k, v))
        bias = jnp.where(attn_mask, jnp.asarray(0, compute_dtype), _cudnn_fa.get_large_negative_number(compute_dtype))
        zeros = jnp.zeros(0, dtype=compute_dtype)
        out, res = _cudnn_fa._dot_product_attention_fwd_rule(
            qc, kc, vc, bias, zeros, zeros, zeros, zeros, *_cudnn_static_args(bias.shape, qc.shape)
        )
        return out.astype(out_dtype), (res, attn_mask)

    def attention_bwd(residuals, d_out):
        res, attn_mask = residuals
        d_out32 = d_out.astype(jnp.float32)
        max_abs = jnp.max(jnp.abs(d_out32))
        tiny = jnp.finfo(jnp.float32).tiny
        # Power-of-two scale so max|d_out * scale| lands in [0.25, 0.5): exact
        # in the mantissa, leaves ~17 binades of headroom for dP = dO @ V^T
        # and the dQ/dK/dV outputs inside the float16 kernel.
        exponent = jnp.floor(jnp.log2(jnp.maximum(max_abs, tiny)))
        scale = jnp.where(max_abs > 0, jnp.exp2(-(exponent + 2.0)), 1.0)
        grads = _cudnn_fa._dot_product_attention_bwd_rule(
            *_cudnn_static_args(res[3].shape, res[0].shape), res, (d_out32 * scale).astype(compute_dtype)
        )
        dq, dk, dv = grads[:3]
        # Same semantics as _stop_gradient_for_fully_masked_queries: fully masked
        # query rows get a zero (never NaN) q-gradient.
        query_has_key = jnp.any(attn_mask, axis=-1)[:, 0, :, None, None]
        dq = jnp.where(query_has_key, dq, jnp.zeros_like(dq))
        inv_scale = 1.0 / scale

        def unscale(g):
            return (g.astype(jnp.float32) * inv_scale).astype(out_dtype)

        return unscale(dq), unscale(dk), unscale(dv), None

    attention.defvjp(attention_fwd, attention_bwd)
    return attention(q, k, v, attn_mask)


def _cudnn_static_args(bias_shape, query_shape):
    """Static parameters jax.nn.dot_product_attention(..., scale=1.0, implementation="cudnn") uses."""
    layout = _cudnn_fa._normalize_layout("BTNH")
    has_dbias = _cudnn_fa.should_export_dbias(bias_shape, query_shape, layout.value)
    # (scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length, cudnn_version)
    return (
        1.0,
        42,
        0.0,
        (True, has_dbias),
        _cudnn_fa.MaskType.NO_MASK,
        layout.value,
        None,
        _cudnn_fa.check_cudnn_version(),
    )


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]
    use_cudnn_attention: bool = False
    # Compute dtype handed to the cuDNN kernel: "bfloat16" (historical) or "float16"
    # (see _cudnn_attention_in_dtype). Ignored by the explicit path.
    cudnn_attention_dtype: str = "bfloat16"
    # Diagnostic only: run the explicit path with fp32 q/k/v/probs as a reference.
    explicit_attention_fp32: bool = False

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, use_cudnn_attention=None):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        def cudnn_attention(operands):
            q, k, v, mask = operands
            # q is already scaled above, so disable dot_product_attention's
            # default head-dimension scaling. Keep cached inference on the
            # existing implementation; this switch targets training throughput.
            # Fully masked query rows make cuDNN produce NaN q-gradients; drop
            # those rows from the backward pass instead of rebuilding the mask.
            compute_dtype = jnp.dtype(self.cudnn_attention_dtype)
            if compute_dtype == q.dtype:
                return _cudnn_attention_call(q, k, v, mask)
            return _cudnn_attention_in_dtype(q, k, v, mask, compute_dtype)

        def explicit_attention(operands):
            q, k, v, mask = operands
            if self.explicit_attention_fp32:
                q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
            grouped_q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
            logits = jnp.einsum("BTKGH,BSKH->BKGTS", grouped_q, k, preferred_element_type=jnp.float32)

            # big_neg = jnp.finfo(logits.dtype).min
            big_neg = -2.3819763e38  # See gemma/modules.py
            masked_logits = jnp.where(mask[:, :, None, :, :], logits, big_neg)

            probs = jax.nn.softmax(masked_logits, axis=-1).astype(v.dtype)

            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v).astype(dtype)
            return einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        if use_cudnn_attention is None:
            use_cudnn_attention = self.use_cudnn_attention
        operands = (q, k, v, attn_mask)
        if kv_cache is not None:
            encoded = explicit_attention(operands)
        elif isinstance(use_cudnn_attention, bool):
            encoded = cudnn_attention(operands) if use_cudnn_attention else explicit_attention(operands)
        else:
            encoded = jax.lax.cond(use_cudnn_attention, cudnn_attention, explicit_attention, operands)

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]
    use_cudnn_attention: bool = False
    cudnn_attention_layer_start: int = 0
    cudnn_attention_num_layers: int | None = None
    cudnn_attention_dtype: str = "bfloat16"
    explicit_attention_fp32: bool = False

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, layer_index=None, deterministic=True):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(
            configs=self.configs,
            use_cudnn_attention=self.use_cudnn_attention,
            cudnn_attention_dtype=self.cudnn_attention_dtype,
            explicit_attention_fp32=self.explicit_attention_fp32,
            name="attn",
        )
        layer_uses_cudnn = self.use_cudnn_attention
        if layer_uses_cudnn and self.cudnn_attention_num_layers is not None:
            layer_uses_cudnn = jnp.logical_and(
                layer_index >= self.cudnn_attention_layer_start,
                layer_index < self.cudnn_attention_layer_start + self.cudnn_attention_num_layers,
            )

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is None:
                gates.append(None)
            else:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
                gates.append(gate)
            pre_attn.append(x)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache, use_cudnn_attention=layer_uses_cudnn)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                gates.append(None)
            else:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
                x = lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
                gates.append(gate)
            out.append(x)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False
    use_cudnn_attention: bool = False
    cudnn_attention_layer_start: int = 0
    cudnn_attention_num_layers: int | None = None
    cudnn_attention_dtype: str = "bfloat16"
    explicit_attention_fp32: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(6,),  # 0=self, 7=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                0,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=layer_index, 5=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            use_cudnn_attention=self.use_cudnn_attention,
            cudnn_attention_layer_start=self.cudnn_attention_layer_start,
            cudnn_attention_num_layers=self.cudnn_attention_num_layers,
            cudnn_attention_dtype=self.cudnn_attention_dtype,
            explicit_attention_fp32=self.explicit_attention_fp32,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | at.Float[at.Array, "b _t _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        layer_indices = jnp.arange(self.configs[0].depth, dtype=jnp.int32)
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, layer_indices, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
