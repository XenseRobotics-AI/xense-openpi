"""Flax/NNX port of FastViT-T12 used as a tactile-image encoder.

This file mirrors the timm/PyTorch implementation in
``openpi/models/fastvit_t12_apple_dist_in1k.py`` block-for-block so that the
PyTorch checkpoint released for ``timm/fastvit_t12.apple_dist_in1k`` can be
converted into Flax variables by ``scripts/convert_fastvit_torch_to_flax.py``.

Key design points
-----------------
- Inputs follow OpenPI's convention: ``(B, H, W, 3)`` floats in ``[-1, 1]``.
  Internally we map back to ``[0, 1]`` and apply ImageNet mean/std (the values
  baked into the original FastViT pretraining recipe).
- BatchNorm runs with ``use_running_average=True`` everywhere: the pretrained
  running stats are frozen but ``scale``/``bias`` remain trainable. This matches
  how the SigLIP encoder is used in the rest of the codebase.
- ``deterministic=True`` always: DropPath / proj-dropout are skipped (FastViT-T12
  ships with 0.0 default rates).
- The module is implemented in ``flax.linen``; downstream code wraps it via
  ``flax.nnx.bridge.ToNNX`` so it composes with the NNX-based Pi0.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

import flax.linen as nn
import flax.nnx as nnx
from flax.nnx import bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("openpi")

FEATURE_DIM = 1024

# ImageNet stats baked into the original FastViT pretraining recipe.
# Stored as numpy arrays so the module can be imported safely while a JAX trace
# is active (e.g. during jax.eval_shape on init_train_state). A module-level
# jnp.asarray here would be evaluated under that trace and stash a
# DynamicJaxprTracer in the module globals, which then leaks into the next
# jax.jit trace and triggers UnexpectedTracerError.
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _normalize_inputs(x: jax.Array) -> jax.Array:
    """Map ``[-1, 1]`` to ImageNet-normalized inputs that the encoder expects."""
    # Cast to model dtype after normalization to keep numerics in fp32.
    in_dtype = x.dtype
    mean = jnp.asarray(_IMAGENET_MEAN)
    std = jnp.asarray(_IMAGENET_STD)
    x = x.astype(jnp.float32) * 0.5 + 0.5
    x = (x - mean) / std
    return x.astype(in_dtype)


def _pytorch_pad(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    """Replicate timm's ``get_padding``: ``((stride-1) + dilation*(kernel_size-1)) // 2``.

    Flax ``padding="SAME"`` does NOT match PyTorch's symmetric padding when
    ``stride > 1``: TF-style SAME (which JAX uses) puts the extra padding at
    the bottom/right (``(0,1)`` for kernel=3 stride=2), while PyTorch pads
    symmetrically (``(1,1)``). The half-pixel offset from this mismatch
    accumulates across the network and is the dominant source of error when
    porting FastViT — measured ``max|diff|`` drops from ~2.35 to ~1e-5 once
    padding is aligned.
    """
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def _conv(
    features: int,
    kernel_size: int,
    *,
    stride: int = 1,
    groups: int = 1,
    use_bias: bool,
    name: str,
    dtype: Any = jnp.float32,
) -> nn.Conv:
    """A timm-style 2D conv with explicit symmetric padding (matches PyTorch exactly).

    ``dtype`` is the compute dtype. Parameters stay in fp32 (``param_dtype``
    defaults to fp32), so the optimizer keeps a fp32 master copy while each
    forward/backward casts to ``dtype`` (typically bf16). We MUST pass dtype
    explicitly — Flax's default ``dtype=None`` runs ``promote_dtype`` which
    upcasts bf16 inputs to fp32 to match the fp32 params, defeating the entire
    purpose of feeding bf16 activations.
    """
    pad = _pytorch_pad(kernel_size, stride=stride)
    return nn.Conv(
        features=features,
        kernel_size=(kernel_size, kernel_size),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=groups,
        use_bias=use_bias,
        dtype=dtype,
        name=name,
    )


def _make_divisible(v: float, divisor: int = 8, *, min_value: int | None = None, round_limit: float = 0.9) -> int:
    """timm-equivalent rounding helper used by SqueezeExcite.

    Matches ``make_divisible`` in ``fastvit_t12_apple_dist_in1k.py`` so that the
    SE bottleneck dimension agrees with the PyTorch checkpoint exactly.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class _ConvBn(nn.Module):
    """Conv -> BN, no activation (matches timm.layers.ConvNormAct(apply_act=False))."""

    features: int
    kernel_size: int
    stride: int = 1
    groups: int = 1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = _conv(
            self.features,
            self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            use_bias=False,
            name="conv",
            dtype=self.dtype,
        )(x)
        x = nn.BatchNorm(use_running_average=True, dtype=self.dtype, name="bn")(x)
        return x


class _SqueezeExcite(nn.Module):
    """Squeeze-and-Excite matching timm's hidden-channel rounding.

    The bottleneck size is ``make_divisible(channels * rd_ratio, rd_divisor)``,
    which is what the upstream PyTorch checkpoint was trained with. Getting this
    wrong (e.g. assuming no reduction) leads to shape-mismatched parameters and
    silently drops the SE weights at conversion time.
    """

    rd_ratio: float = 1.0 / 16
    rd_divisor: int = 8
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        c = x.shape[-1]
        hidden = _make_divisible(c * self.rd_ratio, self.rd_divisor, round_limit=0.0)
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = nn.Conv(features=hidden, kernel_size=(1, 1), use_bias=True, dtype=self.dtype, name="fc1")(s)
        s = jax.nn.relu(s)
        s = nn.Conv(features=c, kernel_size=(1, 1), use_bias=True, dtype=self.dtype, name="fc2")(s)
        s = jax.nn.sigmoid(s)
        return x * s


class MobileOneBlock(nn.Module):
    """MobileOne-style multi-branch conv used throughout FastViT-T12.

    Non-inference-mode forward: ``act(se(scale_branch + identity_branch + sum_k(conv_kxk_k)))``.
    """

    out_chs: int
    kernel_size: int
    stride: int = 1
    group_size: int = 0  # 0 -> groups=1; otherwise groups = in_chs // group_size
    use_se: bool = False
    use_act: bool = True
    use_scale_branch: bool = True
    num_conv_branches: int = 1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_chs = x.shape[-1]
        groups = 1 if self.group_size == 0 else in_chs // self.group_size

        # identity branch (BN over input) — present when shapes match
        if self.out_chs == in_chs and self.stride == 1:
            out = nn.BatchNorm(use_running_average=True, dtype=self.dtype, name="identity")(x)
        else:
            h = (x.shape[1] + self.stride - 1) // self.stride
            w = (x.shape[2] + self.stride - 1) // self.stride
            out = jnp.zeros((x.shape[0], h, w, self.out_chs), dtype=x.dtype)

        # scale branch (1x1 conv) — only when kernel_size>1 and use_scale_branch
        if self.kernel_size > 1 and self.use_scale_branch:
            scale = _ConvBn(
                features=self.out_chs,
                kernel_size=1,
                stride=self.stride,
                groups=groups,
                dtype=self.dtype,
                name="conv_scale",
            )(x)
            out = out + scale

        # parallel k×k convs
        for i in range(self.num_conv_branches):
            branch = _ConvBn(
                features=self.out_chs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=groups,
                dtype=self.dtype,
                name=f"conv_kxk_{i}",
            )(x)
            out = out + branch

        if self.use_se:
            out = _SqueezeExcite(rd_divisor=1, dtype=self.dtype, name="se")(out)

        if self.use_act:
            out = jax.nn.gelu(out, approximate=False)
        return out


class _ReparamLargeKernelConv(nn.Module):
    """``large_conv`` + ``small_conv``; no activation in itself (act applied outside)."""

    out_chs: int
    kernel_size: int
    stride: int
    group_size: int
    small_kernel: int | None = 3
    use_se: bool = False
    use_act: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_chs = x.shape[-1]
        groups = 1 if self.group_size == 0 else in_chs // self.group_size
        out = _ConvBn(
            features=self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=groups,
            dtype=self.dtype,
            name="large_conv",
        )(x)
        if self.small_kernel is not None:
            out = out + _ConvBn(
                features=self.out_chs,
                kernel_size=self.small_kernel,
                stride=self.stride,
                groups=groups,
                dtype=self.dtype,
                name="small_conv",
            )(x)
        if self.use_se:
            out = _SqueezeExcite(rd_ratio=0.25, dtype=self.dtype, name="se")(out)
        if self.use_act:
            out = jax.nn.gelu(out, approximate=False)
        return out


class _PatchEmbed(nn.Module):
    """Strided large-kernel conv followed by a MobileOne 1x1 projection."""

    embed_dim: int
    patch_size: int
    stride: int
    use_se: bool = False
    lkc_use_act: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = _ReparamLargeKernelConv(
            out_chs=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            group_size=1,
            small_kernel=3,
            use_se=self.use_se,
            use_act=self.lkc_use_act,
            dtype=self.dtype,
            name="proj_0",
        )(x)
        x = MobileOneBlock(
            out_chs=self.embed_dim,
            kernel_size=1,
            stride=1,
            use_se=False,
            num_conv_branches=1,
            dtype=self.dtype,
            name="proj_1",
        )(x)
        return x


class _LayerScale2d(nn.Module):
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = x.shape[-1]
        gamma = self.param(
            "gamma",
            lambda key, shape: jnp.full(shape, self.init_values, dtype=jnp.float32),
            (dim,),
        )
        return x * gamma.astype(x.dtype)


class _RepMixer(nn.Module):
    """Two parallel MobileOne paths (mixer, norm) combined via ``x + ls*(mixer - norm)``."""

    kernel_size: int = 3
    layer_scale_init_value: float | None = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = x.shape[-1]
        # ``norm`` and ``mixer`` are both group-size=1 (depthwise) MobileOne blocks.
        # ``norm`` is constructed with ``use_act=False`` AND ``use_scale_branch=False``
        # AND ``num_conv_branches=0`` — i.e. only the identity BN branch is active.
        # That collapses to a single BN over the input.
        norm = nn.BatchNorm(use_running_average=True, dtype=self.dtype, name="norm_identity")(x)
        mixer = MobileOneBlock(
            out_chs=dim,
            kernel_size=self.kernel_size,
            stride=1,
            group_size=1,
            use_act=False,
            use_scale_branch=True,
            num_conv_branches=1,
            dtype=self.dtype,
            name="mixer",
        )(x)
        delta = mixer - norm
        if self.layer_scale_init_value is not None:
            delta = _LayerScale2d(init_values=self.layer_scale_init_value, name="layer_scale")(delta)
        return x + delta


class _ConvMlp(nn.Module):
    """Depthwise 7x7 conv -> 1x1 expand -> GELU -> 1x1 project."""

    hidden_channels: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = x.shape[-1]
        x = _ConvBn(features=dim, kernel_size=7, stride=1, groups=dim, dtype=self.dtype, name="conv")(x)
        x = nn.Conv(features=self.hidden_channels, kernel_size=(1, 1), use_bias=True, dtype=self.dtype, name="fc1")(x)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.Conv(features=dim, kernel_size=(1, 1), use_bias=True, dtype=self.dtype, name="fc2")(x)
        return x


class _RepMixerBlock(nn.Module):
    kernel_size: int = 3
    mlp_ratio: float = 3.0
    layer_scale_init_value: float | None = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = _RepMixer(
            kernel_size=self.kernel_size,
            layer_scale_init_value=self.layer_scale_init_value,
            dtype=self.dtype,
            name="token_mixer",
        )(x)
        delta = _ConvMlp(hidden_channels=int(x.shape[-1] * self.mlp_ratio), dtype=self.dtype, name="mlp")(x)
        if self.layer_scale_init_value is not None:
            delta = _LayerScale2d(init_values=self.layer_scale_init_value, name="layer_scale")(delta)
        x = x + delta
        return x


class _FastVitStage(nn.Module):
    embed_dim: int
    depth: int
    downsample: bool
    se_downsample: bool
    down_patch_size: int = 7
    down_stride: int = 2
    kernel_size: int = 3
    mlp_ratio: float = 3.0
    lkc_use_act: bool = False
    layer_scale_init_value: float | None = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.downsample:
            x = _PatchEmbed(
                embed_dim=self.embed_dim,
                patch_size=self.down_patch_size,
                stride=self.down_stride,
                use_se=self.se_downsample,
                lkc_use_act=self.lkc_use_act,
                dtype=self.dtype,
                name="downsample",
            )(x)
        for i in range(self.depth):
            x = _RepMixerBlock(
                kernel_size=self.kernel_size,
                mlp_ratio=self.mlp_ratio,
                layer_scale_init_value=self.layer_scale_init_value,
                dtype=self.dtype,
                name=f"blocks_{i}",
            )(x)
        return x


class FastVitT12Module(nn.Module):
    """Flax/linen FastViT-T12, returning a (B, 1024) global-pooled embedding.

    ``dtype`` controls the compute dtype for every conv/BN/dense in the module.
    Parameters remain stored in fp32 (``param_dtype`` defaults to fp32) so the
    optimizer keeps an fp32 master copy; the params are cast on the fly to
    ``dtype`` for each op. Setting ``dtype=jnp.bfloat16`` on H100/A100 roughly
    halves activation memory and unlocks bf16 tensor cores on the dense
    matmuls. Default stays fp32 to preserve the exact numerics used by the
    Torch→Flax conversion verifier.
    """

    layers: Sequence[int] = (2, 2, 6, 2)
    embed_dims: Sequence[int] = (64, 128, 256, 512)
    mlp_ratios: Sequence[float] = (3.0, 3.0, 3.0, 3.0)
    downsamples: Sequence[bool] = (False, True, True, True)
    se_downsamples: Sequence[bool] = (False, False, False, False)
    repmixer_kernel_size: int = 3
    cls_ratio: float = 2.0
    layer_scale_init_value: float | None = 1e-5
    lkc_use_act: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Normalization stays in fp32 for numerical stability, then cast to the
        # compute dtype. Every nn.Conv / nn.BatchNorm receives ``dtype=self.dtype``
        # explicitly because Flax's default ``dtype=None`` calls promote_dtype
        # which upcasts bf16 inputs to fp32 to match the fp32 params, defeating
        # the entire purpose. With dtype set explicitly, params stay stored as
        # fp32 (param_dtype default) but are cast to bf16 on the fly for each
        # op — i.e. the standard "fp32 master + bf16 compute" pattern.
        x = _normalize_inputs(x)
        x = x.astype(self.dtype)

        # Stem: 3 MobileOne blocks (kernels 3-3-1). Match timm naming "stem.0/1/2".
        x = MobileOneBlock(
            out_chs=self.embed_dims[0], kernel_size=3, stride=2, num_conv_branches=1, dtype=self.dtype, name="stem_0"
        )(x)
        x = MobileOneBlock(
            out_chs=self.embed_dims[0],
            kernel_size=3,
            stride=2,
            group_size=1,
            num_conv_branches=1,
            dtype=self.dtype,
            name="stem_1",
        )(x)
        x = MobileOneBlock(
            out_chs=self.embed_dims[0], kernel_size=1, stride=1, num_conv_branches=1, dtype=self.dtype, name="stem_2"
        )(x)

        # Stages
        prev_dim = self.embed_dims[0]
        for i, depth in enumerate(self.layers):
            downsample = bool(self.downsamples[i]) or (prev_dim != self.embed_dims[i])
            x = _FastVitStage(
                embed_dim=self.embed_dims[i],
                depth=depth,
                downsample=downsample,
                se_downsample=bool(self.se_downsamples[i]),
                down_patch_size=7,
                down_stride=2,
                kernel_size=self.repmixer_kernel_size,
                mlp_ratio=float(self.mlp_ratios[i]),
                lkc_use_act=self.lkc_use_act,
                layer_scale_init_value=self.layer_scale_init_value,
                dtype=self.dtype,
                name=f"stages_{i}",
            )(x)
            prev_dim = self.embed_dims[i]

        # final conv: MobileOne with SE
        final_features = int(self.embed_dims[-1] * self.cls_ratio)
        x = MobileOneBlock(
            out_chs=final_features,
            kernel_size=3,
            stride=1,
            group_size=1,
            use_se=True,
            num_conv_branches=1,
            dtype=self.dtype,
            name="final_conv",
        )(x)
        # Global average pool -> (B, C)
        x = jnp.mean(x, axis=(1, 2))
        return x


class TactileFastVitEncoder(nnx.Module):
    """NNX-friendly wrapper around :class:`FastVitT12Module`.

    The wrapper handles ``lazy_init`` so that this encoder slots into ``Pi0`` the
    same way the SigLIP image encoder does (see ``nnx_bridge.ToNNX`` usage in
    ``pi0.py``). The optional ``pretrained_path`` can point at a safetensors file
    produced by ``scripts/convert_fastvit_torch_to_flax.py``; if ``None`` the
    encoder is trained from scratch (which the user opted out of by default).
    """

    feature_dim: int = FEATURE_DIM

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        pretrained_path: str | Path | None = None,
        compute_dtype: Any = jnp.float32,
    ) -> None:
        self.feature_dim = FEATURE_DIM
        # The wrapped linen module uses ``compute_dtype`` for every conv/BN/dense
        # while ``param_dtype`` stays fp32. lazy_init still runs with the fake
        # input below — params therefore initialise in fp32 (the fp32 master
        # copy) and are cast to compute_dtype on the fly during each forward.
        self.module = nnx_bridge.ToNNX(FastVitT12Module(dtype=compute_dtype))
        # Initialise with a fake (B, 224, 224, 3) batch. We deliberately feed
        # fp32 here so lazy_init records fp32 params (overwritten later by
        # _load_pretrained); the compute_dtype cast happens inside __call__ so
        # bf16 mode does not affect parameter dtype.
        fake = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
        self.module.lazy_init(fake, rngs=rngs)

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"FastViT Flax checkpoint not found at {path}. Run "
                "scripts/convert_fastvit_torch_to_flax.py first."
            )
        logger.info(f"Loading FastViT Flax weights from {path}")
        from safetensors.flax import load_file

        flat = load_file(str(path))
        # ``flat`` keys are "module/<linen path>/<name>" with values as np arrays.
        # We walk our NNX state and overwrite any matching leaf.
        graphdef, state = nnx.split(self.module)
        state_dict = state.to_pure_dict()

        def _set(d: dict, key_path: tuple[str, ...], value: np.ndarray) -> bool:
            cur = d
            for k in key_path[:-1]:
                if k not in cur:
                    return False
                cur = cur[k]
            leaf = key_path[-1]
            if leaf not in cur:
                return False
            target = cur[leaf]
            arr = jnp.asarray(value).astype(target.dtype)
            if arr.shape != target.shape:
                logger.warning(
                    f"FastViT pretrained shape mismatch at {'/'.join(key_path)}: "
                    f"got {arr.shape}, expected {target.shape}; skipped."
                )
                return False
            cur[leaf] = arr
            return True

        loaded = 0
        for k, v in flat.items():
            key_path = tuple(k.split("/"))
            if _set(state_dict, key_path, v):
                loaded += 1
        logger.info(f"FastViT: loaded {loaded}/{len(flat)} tensors from {path}")
        state.replace_by_pure_dict(state_dict)
        self.module = nnx.merge(graphdef, state)

    def __call__(self, images: jax.Array) -> jax.Array:
        return self.module(images)


__all__ = [
    "FEATURE_DIM",
    "FastVitT12Module",
    "TactileFastVitEncoder",
]
