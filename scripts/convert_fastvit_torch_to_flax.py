#!/usr/bin/env python
"""Convert the PyTorch ``timm/fastvit_t12.apple_dist_in1k`` checkpoint into Flax weights.

Usage:
    uv run scripts/convert_fastvit_torch_to_flax.py \\
        --torch-checkpoint-dir checkpoint/fastvit_t12_apple_dist_in1k \\
        --out-path checkpoint/fastvit_t12_apple_dist_in1k_flax/params.safetensors \\
        --verify-numerics

The output is a flat dict of ``"<linen path>/<leaf>": np.ndarray`` saved via
``safetensors.flax.save_file`` and consumed by ``TactileFastVitEncoder._load_pretrained``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Make ``src/`` importable when the script is run via ``uv run``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import flax.nnx as nnx
import torch

import openpi.models.fastvit_t12_apple_dist_in1k as fastvit_torch
from openpi.models.tactile_encoders.fastvit import FastVitT12Module, TactileFastVitEncoder

logger = logging.getLogger("convert_fastvit_torch_to_flax")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------


_STEM_PATTERN = re.compile(r"^stem\.(\d+)\.(.*)$")
_STAGES_PATTERN = re.compile(r"^stages\.(\d+)\.(downsample|blocks)\.(.*)$")
_DOWN_BLOCK_PATTERN = re.compile(r"^proj\.(\d+)\.(.*)$")
_REPMIXER_PATTERN = re.compile(r"^(\d+)\.(token_mixer|mlp|layer_scale)\.(.*)$")


def _map_mobileone(prefix: str, suffix: str) -> str | None:
    """Map PyTorch MobileOneBlock suffix to the corresponding Flax linen path.

    ``prefix`` is the Flax path prefix up to (and excluding) the MobileOne block
    itself. ``suffix`` is the remainder of the PyTorch key after the MobileOne
    block name.

    PyTorch suffixes:
        identity.{running_mean, running_var, weight, bias, num_batches_tracked}
        conv_scale.conv.weight
        conv_scale.bn.{running_mean, running_var, weight, bias, num_batches_tracked}
        conv_kxk.{i}.conv.weight
        conv_kxk.{i}.bn.{running_mean, running_var, weight, bias, num_batches_tracked}
        se.fc1.{weight, bias}; se.fc2.{weight, bias}
    """
    if suffix.startswith("identity."):
        leaf = suffix[len("identity."):]
        return _map_bn_leaf(f"{prefix}/identity", leaf)
    if suffix.startswith("conv_scale."):
        rest = suffix[len("conv_scale."):]
        return _map_convbn(f"{prefix}/conv_scale", rest)
    m = re.match(r"^conv_kxk\.(\d+)\.(.*)$", suffix)
    if m:
        idx, rest = m.group(1), m.group(2)
        return _map_convbn(f"{prefix}/conv_kxk_{idx}", rest)
    if suffix.startswith("se."):
        rest = suffix[len("se."):]
        if rest.startswith("fc1.") or rest.startswith("fc2."):
            sub, kind = rest.split(".", 1)
            if kind == "weight":
                return f"{prefix}/se/{sub}/kernel"
            if kind == "bias":
                return f"{prefix}/se/{sub}/bias"
    return None


def _map_convbn(prefix: str, suffix: str) -> str | None:
    """Map ``conv.weight`` or ``bn.<x>`` under a ConvBn module."""
    if suffix == "conv.weight":
        return f"{prefix}/conv/kernel"
    if suffix.startswith("bn."):
        return _map_bn_leaf(f"{prefix}/bn", suffix[len("bn."):])
    return None


def _map_bn_leaf(prefix: str, leaf: str) -> str | None:
    """Map a torch BatchNorm leaf into the linen BN naming.

    PyTorch:               Linen (with use_running_average=True):
        weight     ->  scale
        bias       ->  bias
        running_mean -> mean       (stored in ``batch_stats`` collection)
        running_var  -> var        (stored in ``batch_stats`` collection)
        num_batches_tracked: skipped.
    """
    if leaf == "weight":
        return f"{prefix}/scale"
    if leaf == "bias":
        return f"{prefix}/bias"
    if leaf == "running_mean":
        return f"{prefix}/mean"
    if leaf == "running_var":
        return f"{prefix}/var"
    if leaf == "num_batches_tracked":
        return None
    return None


def _map_repmixer_block(stage_idx: int, block_idx: int, suffix: str) -> str | None:
    """Suffix starts with ``token_mixer.``/``mlp.``/``layer_scale.``."""
    prefix = f"stages_{stage_idx}/blocks_{block_idx}"
    if suffix.startswith("token_mixer."):
        rest = suffix[len("token_mixer."):]
        # token_mixer.norm.*  -> norm is a MobileOneBlock collapsing to identity-BN
        # In the PyTorch impl `norm` is a MobileOneBlock with num_conv_branches=0,
        # use_scale_branch=False, use_act=False. So only `identity` BN is present.
        if rest.startswith("norm."):
            inner = rest[len("norm."):]
            if inner.startswith("identity."):
                return _map_bn_leaf(f"{prefix}/token_mixer/norm_identity", inner[len("identity."):])
            return None
        if rest.startswith("mixer."):
            inner = rest[len("mixer."):]
            return _map_mobileone(f"{prefix}/token_mixer/mixer", inner)
        if rest.startswith("layer_scale."):
            inner = rest[len("layer_scale."):]
            if inner == "gamma":
                return f"{prefix}/token_mixer/layer_scale/gamma"
            return None
    if suffix.startswith("mlp."):
        rest = suffix[len("mlp."):]
        if rest.startswith("conv."):
            return _map_convbn(f"{prefix}/mlp/conv", rest[len("conv."):])
        if rest == "fc1.weight":
            return f"{prefix}/mlp/fc1/kernel"
        if rest == "fc1.bias":
            return f"{prefix}/mlp/fc1/bias"
        if rest == "fc2.weight":
            return f"{prefix}/mlp/fc2/kernel"
        if rest == "fc2.bias":
            return f"{prefix}/mlp/fc2/bias"
    if suffix == "layer_scale.gamma":
        return f"{prefix}/layer_scale/gamma"
    return None


def _map_downsample(stage_idx: int, suffix: str) -> str | None:
    """downsample is a PatchEmbed -> Sequential(ReparamLargeKernelConv, MobileOneBlock).

    PyTorch:  downsample.proj.0.* (LK conv), downsample.proj.1.* (MobileOne 1x1)
    Flax:     stages_{i}/downsample/proj_0/* and stages_{i}/downsample/proj_1/*
    """
    m = _DOWN_BLOCK_PATTERN.match(suffix)
    if not m:
        return None
    idx = int(m.group(1))
    rest = m.group(2)
    prefix = f"stages_{stage_idx}/downsample/proj_{idx}"
    if idx == 0:
        # ReparamLargeKernelConv: large_conv.{conv|bn}, small_conv.{conv|bn}
        if rest.startswith("large_conv."):
            return _map_convbn(f"{prefix}/large_conv", rest[len("large_conv."):])
        if rest.startswith("small_conv."):
            return _map_convbn(f"{prefix}/small_conv", rest[len("small_conv."):])
        if rest.startswith("se."):
            inner = rest[len("se."):]
            if inner.startswith("fc1.") or inner.startswith("fc2."):
                sub, kind = inner.split(".", 1)
                if kind == "weight":
                    return f"{prefix}/se/{sub}/kernel"
                if kind == "bias":
                    return f"{prefix}/se/{sub}/bias"
        return None
    if idx == 1:
        return _map_mobileone(prefix, rest)
    return None


def _map_stem_block(idx: int, suffix: str) -> str | None:
    return _map_mobileone(f"stem_{idx}", suffix)


def _torch_key_to_flax_path(key: str) -> str | None:
    """Map a PyTorch state-dict key to a ``"/"``-joined Flax variable path."""
    # stem
    m = _STEM_PATTERN.match(key)
    if m:
        return _map_stem_block(int(m.group(1)), m.group(2))

    # final_conv (a MobileOneBlock at the top level)
    if key.startswith("final_conv."):
        return _map_mobileone("final_conv", key[len("final_conv."):])

    # stages
    m = _STAGES_PATTERN.match(key)
    if m:
        stage_idx = int(m.group(1))
        kind = m.group(2)
        rest = m.group(3)
        if kind == "downsample":
            return _map_downsample(stage_idx, rest)
        if kind == "blocks":
            # rest = "<block_idx>.<inner>"
            mm = _REPMIXER_PATTERN.match(rest)
            if mm:
                block_idx = int(mm.group(1))
                inner_kind = mm.group(2)
                inner_rest = mm.group(3)
                return _map_repmixer_block(stage_idx, block_idx, f"{inner_kind}.{inner_rest}")
    return None


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------


def _torch_weight_to_flax(name: str, torch_array: np.ndarray) -> np.ndarray:
    """Reshape PyTorch tensors into Flax layout where needed."""
    if name.endswith("/kernel"):
        # Conv kernel: torch (out, in/groups, kH, kW) -> flax (kH, kW, in/groups, out)
        if torch_array.ndim == 4:
            return np.transpose(torch_array, (2, 3, 1, 0))
        # Dense kernel: torch (out, in) -> flax (in, out). Not used here.
        if torch_array.ndim == 2:
            return np.transpose(torch_array, (1, 0))
    if name.endswith("/gamma"):
        # LayerScale2d stores gamma as (C, 1, 1) in PyTorch (for NCHW broadcast).
        # In Flax NHWC we keep it as (C,); broadcast handles the spatial dims.
        if torch_array.ndim == 3 and torch_array.shape[1:] == (1, 1):
            return torch_array.reshape(torch_array.shape[0])
    return torch_array


# Keys that are expected to have no Flax counterpart and can be silently dropped.
# Everything else that fails to map is treated as an error in main().
_EXPECTED_UNMAPPED_PREFIXES = ("head.",)
_EXPECTED_UNMAPPED_SUFFIXES = (".num_batches_tracked",)


def _strip_wrapper_prefix(key: str) -> str:
    """Strip the ``model.`` prefix added by ``FastVitT12AppleDistIn1kEncoder``.

    ``FastVitT12AppleDistIn1kEncoder`` is a thin nn.Module wrapper that holds the
    real ``FastVit`` under ``self.model``. Calling ``state_dict()`` on the wrapper
    therefore yields keys like ``model.stem.0.conv_kxk.0.conv.weight``. Our path
    mapper expects the bare keys (``stem.0...``), so we drop the wrapper prefix
    here.
    """
    return key[len("model."):] if key.startswith("model.") else key


def convert(torch_state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], list[str]]:
    flat: dict[str, np.ndarray] = {}
    unmapped: list[str] = []
    for original_key, v in torch_state_dict.items():
        k = _strip_wrapper_prefix(original_key)
        v_np = v.detach().cpu().numpy()
        flax_path = _torch_key_to_flax_path(k)
        if flax_path is None:
            if k.startswith(_EXPECTED_UNMAPPED_PREFIXES):
                continue
            if k.endswith(_EXPECTED_UNMAPPED_SUFFIXES):
                # BatchNorm's num_batches_tracked is a non-trainable counter; Flax doesn't have it.
                continue
            unmapped.append(original_key)
            continue
        flat[flax_path] = _torch_weight_to_flax(flax_path, v_np)
    return flat, unmapped


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------


def _build_torch_model(torch_dir: Path) -> torch.nn.Module:
    encoder = fastvit_torch.create_encoder(
        pretrained=True,
        checkpoint_path=str(torch_dir),
        reparameterize=False,
        strict=False,
        in_chans=3,
    )
    encoder.eval()
    return encoder


def _build_flax_encoder(out_safetensors: Path) -> TactileFastVitEncoder:
    rngs = nnx.Rngs(0)
    return TactileFastVitEncoder(rngs=rngs, pretrained_path=out_safetensors)


def _verify(torch_model: torch.nn.Module, flax_encoder: TactileFastVitEncoder, tol: float = 5e-3) -> None:
    np.random.seed(0)
    x = np.random.uniform(-1.0, 1.0, size=(2, 224, 224, 3)).astype(np.float32)

    with torch.no_grad():
        # torch: (B, 3, H, W), normalized by ImageNet stats inside the encoder?
        # The timm model does NOT bake normalization in — it expects normalized input.
        # Our Flax module does the normalization internally. To compare apples to
        # apples we pre-normalize the torch input the same way.
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        x_unit = x * 0.5 + 0.5
        x_norm = (x_unit - mean) / std
        x_torch = torch.from_numpy(np.transpose(x_norm, (0, 3, 1, 2))).contiguous()
        y_torch = torch_model(x_torch).numpy()

    y_flax = np.asarray(jax.device_get(flax_encoder(jnp.asarray(x))))

    if y_torch.shape != y_flax.shape:
        raise RuntimeError(f"Shape mismatch: torch={y_torch.shape} flax={y_flax.shape}")

    diff = np.abs(y_torch - y_flax).reshape(-1)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    # Print a distribution of the diff so it's obvious whether a failure is a
    # structural bug (broad disagreement → high mean and tail percentiles) vs.
    # accumulated fp32 noise (low mean, a few outliers near the max).
    p50, p90, p99, p999 = (float(np.percentile(diff, q)) for q in (50, 90, 99, 99.9))
    frac_above_1e3 = float((diff >= 1e-3).mean())
    frac_above_5e4 = float((diff >= 5e-4).mean())
    logger.info(
        f"Verification: max|diff|={max_diff:.3e}  mean|diff|={mean_diff:.3e}  "
        f"p50={p50:.2e}  p90={p90:.2e}  p99={p99:.2e}  p99.9={p999:.2e}  "
        f"fraction≥1e-3={frac_above_1e3:.4%}  fraction≥5e-4={frac_above_5e4:.4%}"
    )
    if max_diff > tol:
        raise RuntimeError(
            f"FastViT Flax / PyTorch outputs disagree by max={max_diff:.3e} (tol={tol:.0e}).\n"
            "If max is just barely above tol while mean is two orders of magnitude smaller, "
            "this is fp32 accumulation noise across XLA/PyTorch reduction orders — pass "
            "`--tol 1e-2` to accept, or run on a different CPU build for a tighter result. "
            "If max AND mean are both large (mean/max ≳ 1/10), there is a structural bug."
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FastViT-T12 timm checkpoint to Flax safetensors")
    parser.add_argument(
        "--torch-checkpoint-dir",
        type=Path,
        default=Path("checkpoint/fastvit_t12_apple_dist_in1k"),
        help="Directory containing model.safetensors / pytorch_model.bin",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("checkpoint/fastvit_t12_apple_dist_in1k_flax/params.safetensors"),
    )
    parser.add_argument("--verify-numerics", action="store_true", help="Run PyTorch/Flax equivalence check")
    parser.add_argument(
        "--tol",
        type=float,
        default=5e-3,
        help=(
            "Maximum absolute element-wise diff allowed between PyTorch and Flax outputs. "
            "Default 5e-3 reflects accumulated fp32 noise across ~80 ops with reductions over "
            "~1024 channels — well above true ULP precision but well below any structural bug. "
            "Set --tol 1e-2 to be even more permissive (e.g. on hardware with weaker fp32 math)."
        ),
    )
    args = parser.parse_args()

    torch_dir = args.torch_checkpoint_dir.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading PyTorch FastViT from {torch_dir}")
    torch_model = _build_torch_model(torch_dir)
    # FastVitT12AppleDistIn1kEncoder is a wrapper around the real FastVit (under .model),
    # so state_dict() prefixes every key with "model.". convert() strips that prefix,
    # but we also expose the inner state_dict here so the count below is meaningful.
    state_dict = dict(torch_model.state_dict())
    logger.info(f"Read {len(state_dict)} tensors from PyTorch checkpoint")

    flat, unmapped = convert(state_dict)
    logger.info(f"Mapped {len(flat)} tensors; {len(unmapped)} unmapped (excluding head.* / num_batches_tracked)")
    for k in unmapped:
        logger.warning(f"Unmapped PyTorch key (UPDATE THE MAPPER): {k}")

    if not flat:
        raise SystemExit(
            "convert() produced zero mapped tensors — refusing to write an empty checkpoint. "
            "Check the warnings above; the most common cause is a wrapper prefix the mapper "
            "does not yet handle."
        )
    if unmapped:
        raise SystemExit(
            f"{len(unmapped)} PyTorch keys had no Flax target. They are listed above. "
            "Either extend _torch_key_to_flax_path or add them to _EXPECTED_UNMAPPED_*."
        )

    # Sanity-check shapes against a freshly-initialised Flax encoder so name
    # mismatches surface immediately rather than at training time.
    logger.info("Initialising fresh Flax encoder for shape verification...")
    fresh = TactileFastVitEncoder(rngs=nnx.Rngs(0), pretrained_path=None)
    _, fresh_state = nnx.split(fresh.module)
    fresh_dict = fresh_state.to_pure_dict()

    def _resolve(d: dict, path: tuple[str, ...]):
        cur = d
        for p in path[:-1]:
            cur = cur.get(p)
            if cur is None:
                return None
        return cur.get(path[-1]) if isinstance(cur, dict) else None

    mismatches = 0
    matched = 0
    for k, v in list(flat.items()):
        target = _resolve(fresh_dict, tuple(k.split("/")))
        if target is None:
            logger.warning(f"No Flax target for converted key {k} (shape {v.shape}); dropping.")
            del flat[k]
            mismatches += 1
            continue
        if tuple(target.shape) != tuple(v.shape):
            logger.warning(
                f"Shape mismatch at {k}: converted {v.shape} vs flax {target.shape}; dropping."
            )
            del flat[k]
            mismatches += 1
            continue
        matched += 1
    logger.info(f"Matched {matched} tensors against fresh Flax encoder; dropped {mismatches}")

    if mismatches:
        raise SystemExit(
            f"{mismatches} converted tensors did not match the Flax encoder. Aborting; "
            "fix the mapper or the Flax module before writing the checkpoint."
        )

    from safetensors.flax import save_file

    save_file({k: jnp.asarray(v) for k, v in flat.items()}, str(out_path))
    logger.info(f"Wrote {out_path}")

    if args.verify_numerics:
        logger.info("Running PyTorch / Flax equivalence check...")
        flax_encoder = _build_flax_encoder(out_path)
        _verify(torch_model, flax_encoder, tol=args.tol)
        logger.info("Equivalence check passed.")


if __name__ == "__main__":
    main()
