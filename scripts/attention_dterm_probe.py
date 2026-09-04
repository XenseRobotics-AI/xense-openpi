"""Test the flash-attention D-term hypothesis on real Pi0.5 activations.

Mechanism under test (confirmed in docs/training-optimization.md, section 4): the cuDNN
fused backward forms ``dS = P * (dP - D)`` with ``D_i = rowsum(dO_i * O_i)`` taken
from the *bf16-rounded* stored output ``O``. On attention-sink rows (P nearly
one-hot) ``dP`` and ``D`` cancel almost exactly, so the rounding error of ``O`` is
a large fraction of ``dS`` and breaks the softmax invariant ``sum_j dS_ij = 0``.
The explicit path forms ``D = sum_j P_ij dP_ij`` from the same rounded ``dP`` and
keeps the invariant.

If that is the mechanism, the cuDNN gradient error must be *predictable*::

    delta_i   = dO_i . (bf16(O_i) - O_i)               (per query row / head)
    err dQ_i  = -delta_i * sum_j P_ij K_j
    err dK_j  = -sum_i delta_i P_ij Q_i

This script captures q/k/v/mask and the true backward cotangent dO of every Gemma
layer from a real training step (real batch, pi05_base weights), then on each layer
compares, against an fp32 explicit reference:

    explicit_bf16      production path (known stable)
    cudnn_bf16         historical cuDNN path (known to diverge)
    cudnn_fp16         cuDNN kernel in float16 + dynamic scaling (candidate)
    cudnn_bf16_shift   cuDNN with V shifted by the sink key's V (candidate; makes
                       O of sink rows small so its rounding error shrinks)
    cudnn_fp16_shift   both

and reports rel error, the cosine between each variant's error and the predicted
D-term error, and the residual after subtracting the prediction. It also prints
where the attention sinks are (position histogram) and the fp16 range headroom.

Run on ONE GPU:
    CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$PWD/src \
        python scripts/attention_dterm_probe.py
"""

import dataclasses
import functools
import logging
import sys
import time

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.gemma as _gemma
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

sys.path.insert(0, "scripts")
import train as train_mod  # noqa: E402

CONFIG = "diag_cudnn_strict_order"
CAPTURE_FILE = "/root/localstorage/logs/dterm_capture.npz"
BATCH = 8
WORKERS = 8
BIG_NEG = -2.3819763e38
NUM_LAYERS = 18

# --------------------------------------------------------------------------------------
# Capture q/k/v/mask/d_out of every layer from a real train step.
# --------------------------------------------------------------------------------------
_CAPTURE: dict[int, dict[str, np.ndarray]] = {}
_LAYER = None  # tracer of the layer index of the Block currently being traced

_orig_block_call = _gemma.Block.__call__
_orig_cudnn_call = _gemma._cudnn_attention_call


@functools.wraps(_orig_block_call)
def _block_call(self, xs, kv_cache, positions, attn_mask, adarms_cond, layer_index=None, deterministic=True):
    global _LAYER
    _LAYER = layer_index
    return _orig_block_call(self, xs, kv_cache, positions, attn_mask, adarms_cond, layer_index, deterministic)


def _save_fwd(layer, q, k, v, mask):
    entry = _CAPTURE.setdefault(int(layer), {})
    entry.update(q=np.asarray(q), k=np.asarray(k), v=np.asarray(v), mask=np.asarray(mask))


def _save_bwd(layer, d_out):
    _CAPTURE.setdefault(int(layer), {})["d_out"] = np.asarray(d_out)


@jax.custom_vjp
def _tap(q, k, v, mask, layer):
    return _orig_cudnn_call(q, k, v, mask)


def _tap_fwd(q, k, v, mask, layer):
    jax.debug.callback(_save_fwd, layer, q, k, v, mask)
    return _tap(q, k, v, mask, layer), (q, k, v, mask, layer)


def _tap_bwd(res, d_out):
    q, k, v, mask, layer = res
    jax.debug.callback(_save_bwd, layer, d_out)
    _, vjp_fn = jax.vjp(lambda a, b, c: _orig_cudnn_call(a, b, c, mask), q, k, v)
    dq, dk, dv = vjp_fn(d_out)
    return dq, dk, dv, None, None


_tap.defvjp(_tap_fwd, _tap_bwd)


def _tapped_cudnn_call(q, k, v, attn_mask):
    return _tap(q, k, v, attn_mask, _LAYER)


def capture_real_activations(cfg):
    _gemma.Block.__call__ = _block_call
    _gemma._cudnn_attention_call = _tapped_cudnn_call
    try:
        mesh = sharding.make_mesh(cfg.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rng = jax.random.key(cfg.seed)
        train_rng, init_rng = jax.random.split(rng)
        state, _ = train_mod.init_train_state(cfg, init_rng, mesh, resume=False)
        params, model_def = state.params, state.model_def
        del state  # free the Adam moments
        jax.block_until_ready(params)
        print("weights loaded", flush=True)

        loader = _data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=False)
        obs, act = next(iter(loader))
        print("batch loaded", flush=True)

        def loss_fn(model, r, o, a):
            return jnp.mean(model.compute_loss(r, o, a, train=True))

        @jax.jit
        def run(params, r, o, a):
            model = nnx.merge(model_def, params)
            model.train()
            return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, cfg.trainable_filter))(model, r, o, a)

        with sharding.set_mesh(mesh):
            loss, grads = run(params, jax.random.fold_in(train_rng, 0), obs, act)
            jax.block_until_ready(grads)
        print(f"real train step: loss={float(loss):.6f}", flush=True)
        del grads, params
    finally:
        _gemma.Block.__call__ = _orig_block_call
        _gemma._cudnn_attention_call = _orig_cudnn_call
    missing = [i for i in range(NUM_LAYERS) if i not in _CAPTURE or "d_out" not in _CAPTURE[i]]
    if missing:
        raise RuntimeError(f"capture incomplete, missing layers {missing}; got {sorted(_CAPTURE)}")


# --------------------------------------------------------------------------------------
# Single-layer attention variants (mirror gemma.Attention exactly).
# --------------------------------------------------------------------------------------
def explicit_attention(q, k, v, mask, compute_dtype, probs_dtype, out_dtype):
    num_kv = k.shape[2]
    q, k, v = (x.astype(compute_dtype) for x in (q, k, v))
    grouped_q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", grouped_q, k, preferred_element_type=jnp.float32)
    masked = jnp.where(mask[:, :, None, :, :], logits, BIG_NEG)
    probs = jax.nn.softmax(masked, axis=-1).astype(probs_dtype)
    enc = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v).astype(out_dtype)
    return einops.rearrange(enc, "B T K G H -> B T (K G) H")


def _shift_v(v, sink_idx):
    """Return v - v[:, sink] (rounded once) and the per-(B,K) shift c (stop-gradient)."""
    c = jax.lax.stop_gradient(jnp.take_along_axis(v, sink_idx[:, None, :, None], axis=1))  # B 1 K H
    v_shift = (v.astype(jnp.float32) - c.astype(jnp.float32)).astype(v.dtype)
    return v_shift, c


def cudnn_shift_attention(q, k, v, mask, sink_idx, compute_dtype):
    v_shift, c = _shift_v(v, sink_idx)
    if jnp.dtype(compute_dtype) == q.dtype:
        out = _orig_cudnn_call(q, k, v_shift, mask)
    else:
        out = _gemma._cudnn_attention_in_dtype(q, k, v_shift, mask, compute_dtype)
    has_key = jnp.any(mask, axis=-1)[:, 0, :, None, None]  # B T 1 1
    c_full = einops.repeat(c, "B 1 K H -> B 1 (K G) H", G=q.shape[2] // k.shape[2])
    return (out.astype(jnp.float32) + jnp.where(has_key, c_full.astype(jnp.float32), 0.0)).astype(out.dtype)


VARIANT_NAMES = ("explicit_bf16", "cudnn_bf16", "cudnn_fp16", "cudnn_bf16_shift", "cudnn_fp16_shift")


def _variant_output(name, q, k, v, mask, sink_idx):
    if name == "explicit_bf16":
        return explicit_attention(q, k, v, mask, jnp.bfloat16, jnp.bfloat16, jnp.bfloat16)
    if name == "cudnn_bf16":
        return _orig_cudnn_call(q, k, v, mask)
    if name == "cudnn_fp16":
        return _gemma._cudnn_attention_in_dtype(q, k, v, mask, jnp.float16)
    if name == "cudnn_bf16_shift":
        return cudnn_shift_attention(q, k, v, mask, sink_idx, jnp.bfloat16)
    if name == "cudnn_fp16_shift":
        return cudnn_shift_attention(q, k, v, mask, sink_idx, jnp.float16)
    raise ValueError(name)


def grads_of(fn, q, k, v, mask, d_out):
    d32 = d_out.astype(jnp.float32)

    def objective(q_, k_, v_):
        return jnp.sum(fn(q_, k_, v_, mask).astype(jnp.float32) * d32)

    return jax.grad(objective, argnums=(0, 1, 2))(q, k, v)


@functools.partial(jax.jit, static_argnums=0)
def variant_grads(name, q, k, v, mask, d_out, sink_idx):
    return grads_of(lambda a, b, c, m: _variant_output(name, a, b, c, m, sink_idx), q, k, v, mask, d_out)


@jax.jit
def reference_and_prediction(q, k, v, mask, d_out):
    """fp32 explicit gradients + the predicted D-term error of a bf16 flash backward."""
    num_kv = k.shape[2]
    q32, k32, v32, d32 = (x.astype(jnp.float32) for x in (q, k, v, d_out))
    ref = grads_of(
        lambda a, b, c, m: explicit_attention(a, b, c, m, jnp.float32, jnp.float32, jnp.float32), q32, k32, v32, mask, d_out
    )

    grouped_q = einops.rearrange(q32, "B T (K G) H -> B T K G H", K=num_kv)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", grouped_q, k32, preferred_element_type=jnp.float32)
    masked = jnp.where(mask[:, :, None, :, :], logits, BIG_NEG)
    probs = jax.nn.softmax(masked, axis=-1)  # B K G T S
    o32 = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v32)
    # reduce_precision emulates the bf16 rounding; a plain astype round-trip is deleted by XLA
    # (xla_allow_excess_precision) and would make the prediction identically zero.
    o16 = jax.lax.reduce_precision(o32, exponent_bits=8, mantissa_bits=7)
    d_o = einops.rearrange(d32, "B T (K G) H -> B T K G H", K=num_kv)
    # H1: D formed from the bf16-rounded stored output O.
    delta = jnp.sum(d_o * (o16 - o32), axis=-1)  # B T K G
    pk = jnp.einsum("BKGTS,BSKH->BTKGH", probs, k32)
    pred_dq = einops.rearrange(-delta[..., None] * pk, "B T K G H -> B T (K G) H")
    pred_dk = -jnp.einsum("BTKG,BKGTS,BTKGH->BSKH", delta, probs, grouped_q)
    # H2: dP = dO V^T rounded to bf16 before dS = P (dP - D) with an exact D.
    dp = jnp.einsum("BTKGH,BSKH->BKGTS", d_o, v32)
    ddp = jax.lax.reduce_precision(dp, exponent_bits=8, mantissa_bits=7) - dp
    pred2_dq = einops.rearrange(jnp.einsum("BKGTS,BSKH->BTKGH", probs * ddp, k32), "B T K G H -> B T (K G) H")

    valid_q = jnp.any(mask, axis=-1)[:, 0, :]  # B T
    max_p = jnp.max(probs, axis=-1)  # B K G T
    arg_p = jnp.argmax(probs, axis=-1)  # B K G T
    mass = jnp.einsum("BKGTS,BT->BKS", probs, valid_q.astype(jnp.float32))  # B K S
    return ref, (pred_dq, pred_dk, pred2_dq), dict(valid_q=valid_q, max_p=max_p, arg_p=arg_p, mass=mass, delta=delta)


# --------------------------------------------------------------------------------------
# Metrics (float64 on host).
# --------------------------------------------------------------------------------------
def _f64(x):
    return np.asarray(jax.device_get(x)).astype(np.float64)


def _norm(x):
    return float(np.sqrt(np.sum(x * x)))


def _cos(a, b):
    return float(np.sum(a * b) / max(_norm(a) * _norm(b), 1e-300))


def _region(pos):
    if pos < 768:
        return f"img{pos // 256}[{pos % 256}]"
    if pos < 968:
        return f"txt[{pos - 768}]"
    return f"act[{pos - 968}]"


def analyse_layer(layer, entry):
    q, k, v, mask, d_out = (jnp.asarray(entry[n]) for n in ("q", "k", "v", "mask", "d_out"))
    ref, (pred_dq, pred_dk, pred2_dq), stats = reference_and_prediction(q, k, v, mask, d_out)
    ref = [_f64(g) for g in ref]
    pred = [_f64(pred_dq), _f64(pred_dk)]
    pred2 = _f64(pred2_dq)
    k_np = _f64(k)  # B S K H
    valid_q = np.asarray(stats["valid_q"])
    max_p = np.asarray(stats["max_p"])  # B K G T
    arg_p = np.asarray(stats["arg_p"])
    mass = np.asarray(stats["mass"])  # B K S

    # ---- sink statistics ----
    d_np = np.asarray(entry["d_out"]).astype(np.float32)
    masked_rows_dout = float(np.abs(d_np[~valid_q]).max()) if (~valid_q).any() else 0.0
    vq = np.broadcast_to(valid_q[:, None, None, :], max_p.shape)
    mp = max_p[vq]
    ap = arg_p[vq]
    frac = {t: float(np.mean(mp > t)) for t in (0.9, 0.99, 0.999)}
    # per (B, K) sink key = argmax of attention mass
    sink_idx = np.argmax(mass, axis=-1)  # B K
    sink_share = mass.max(axis=-1) / mass.sum(axis=-1)  # fraction of all valid attention mass on the sink key
    pos_counts = np.bincount(ap, minlength=q.shape[1])
    top = np.argsort(pos_counts)[::-1][:4]
    top_desc = ", ".join(f"{_region(int(p))}={pos_counts[p] / ap.size:.2%}" for p in top)
    print(
        f"\n=== layer {layer:2d} ===  valid rows maxP>0.9/0.99/0.999: {frac[0.9]:.3f}/{frac[0.99]:.3f}/{frac[0.999]:.3f}"
        f"   sink key per (b,kv): {sorted(set(sink_idx.ravel().tolist()))} share={sink_share.mean():.2%}"
        f"   argmax top: {top_desc}"
        f"   max|q|={float(jnp.abs(q).max()):.1f} |k|={float(jnp.abs(k).max()):.1f} |v|={float(jnp.abs(v).max()):.1f}"
        f" |dO|={float(jnp.abs(d_out).max()):.2e} |dO| on masked rows={masked_rows_dout:.1e}",
        flush=True,
    )

    # ---- gradient variants ----
    sink_idx_dev = jnp.asarray(sink_idx)

    sink_rows = np.transpose(max_p > 0.99, (0, 3, 1, 2)).reshape(max_p.shape[0], max_p.shape[3], -1)  # B T N
    arg_rows = np.transpose(arg_p, (0, 3, 1, 2)).reshape(sink_rows.shape)  # B T N: each row's own peak key
    valid_rows = np.broadcast_to(valid_q[:, :, None], sink_rows.shape)
    print(
        f"  {'variant':18s} {'dQ rel':>9s} {'cos(err,pred)':>13s} {'resid':>9s} {'sinkrow rel':>11s} {'other rel':>9s} |"
        f" {'dK rel':>9s} {'cos':>7s} {'resid':>9s} | {'dV rel':>9s} | nonfinite",
        flush=True,
    )
    results = {}
    for name in VARIANT_NAMES:
        t0 = time.time()
        g = variant_grads(name, q, k, v, mask, d_out, sink_idx_dev)
        g = [_f64(x) for x in g]
        nonfinite = sum(int(np.count_nonzero(~np.isfinite(x))) for x in g)
        err = [g[i] - ref[i] for i in range(3)]
        row = {}
        for i, nm in enumerate(("dq", "dk", "dv")):
            row[f"{nm}_rel"] = _norm(err[i]) / _norm(ref[i])
            if i < 2:
                row[f"{nm}_cos"] = _cos(err[i], pred[i])
                row[f"{nm}_resid"] = _norm(err[i] - pred[i]) / _norm(ref[i])
        # dQ error split by sink rows / other valid rows
        e_rows = np.sqrt(np.sum(err[0] ** 2, axis=-1))  # B T N
        r_rows = np.sqrt(np.sum(ref[0] ** 2, axis=-1))
        sel_s = sink_rows & valid_rows
        sel_o = (~sink_rows) & valid_rows
        row["dq_sink_rel"] = _norm(e_rows[sel_s]) / max(_norm(r_rows[sel_s]), 1e-300)
        row["dq_other_rel"] = _norm(e_rows[sel_o]) / max(_norm(r_rows[sel_o]), 1e-300)
            # --- peaked-row structure: is the error along the row's peak key (a softmax row-sum violation)? ---
        if sel_s.any():
            e_s = err[0][sel_s]  # R H
            b_idx = np.nonzero(sel_s)[0]
            ks = k_np[b_idx, arg_rows[sel_s], 0, :]  # R H: K of each peaked row's own argmax key (single kv head)
            ks_unit = ks / np.linalg.norm(ks, axis=-1, keepdims=True)
            along = np.sum(e_s * ks_unit, axis=-1)
            row["sink_frac_along_ksink"] = float(np.sum(along**2) / max(np.sum(e_s**2), 1e-300))
            row["sink_cos_h1"] = _cos(e_s, pred[0][sel_s])
            row["sink_cos_h2"] = _cos(e_s, pred2[sel_s])
            row["sink_cos_h12"] = _cos(e_s, pred[0][sel_s] + pred2[sel_s])
            row["sink_resid_h1"] = _norm(e_s - pred[0][sel_s]) / max(_norm(r_rows[sel_s]), 1e-300)
            row["sink_resid_h12"] = _norm(e_s - pred[0][sel_s] - pred2[sel_s]) / max(_norm(r_rows[sel_s]), 1e-300)
        else:
            for key in ("sink_frac_along_ksink", "sink_cos_h1", "sink_cos_h2", "sink_cos_h12", "sink_resid_h1", "sink_resid_h12"):
                row[key] = float("nan")
        results[name] = row
        print(
            f"  {name:18s} {row['dq_rel']:9.3e} {row['dq_cos']:13.4f} {row['dq_resid']:9.3e} {row['dq_sink_rel']:11.3e}"
            f" {row['dq_other_rel']:9.3e} | {row['dk_rel']:9.3e} {row['dk_cos']:7.4f} {row['dk_resid']:9.3e} |"
            f" {row['dv_rel']:9.3e} | {nonfinite} ({time.time() - t0:.1f}s)"
            f"\n      sink rows: frac(err along K_peak)={row['sink_frac_along_ksink']:.3f}  cos(err,H1)={row['sink_cos_h1']:+.3f}"
            f"  cos(err,H2)={row['sink_cos_h2']:+.3f}  cos(err,H1+H2)={row['sink_cos_h12']:+.3f}"
            f"  resid H1={row['sink_resid_h1']:.3e}  resid H1+H2={row['sink_resid_h12']:.3e}",
            flush=True,
        )
    pred_rel = [_norm(pred[i]) / _norm(ref[i]) for i in range(2)]
    n_sink = int(sel_s.sum())
    sink_pred = _norm(pred[0][sel_s]) / max(_norm(r_rows[sel_s]), 1e-300) if n_sink else float("nan")
    sink_pred2 = _norm(pred2[sel_s]) / max(_norm(r_rows[sel_s]), 1e-300) if n_sink else float("nan")
    print(
        f"  predicted error norm (rel to fp32 grads): H1 dQ {pred_rel[0]:.3e} dK {pred_rel[1]:.3e};"
        f" on {n_sink} sink rows: H1 {sink_pred:.3e}  H2 {sink_pred2:.3e}",
        flush=True,
    )
    results["_pred_rel"] = pred_rel
    results["_sink"] = dict(frac=frac, sink_idx=sink_idx.ravel().tolist(), share=float(sink_share.mean()))
    return results


def main():
    logging.basicConfig(level=logging.WARNING)
    cfg = _config.get_config(CONFIG)
    cfg = dataclasses.replace(
        cfg,
        exp_name="attention_dterm_probe",
        batch_size=BATCH,
        num_workers=WORKERS,
        fsdp_devices=1,
        wandb_enabled=False,
        model=dataclasses.replace(
            cfg.model,
            use_cudnn_attention=True,
            cudnn_attention_layer_start=0,
            cudnn_attention_num_layers=None,
            cudnn_attention_dtype="bfloat16",
            explicit_attention_fp32=False,
        ),
    )
    assert jax.device_count() == 1, "run with CUDA_VISIBLE_DEVICES=<one gpu>"
    t0 = time.time()
    import os

    if os.path.exists(CAPTURE_FILE):
        data = np.load(CAPTURE_FILE)
        import ml_dtypes

        for key in data.files:
            layer, name = key.split("/")
            arr = data[key]
            if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:  # bf16 saved by np.savez as raw bytes
                arr = arr.view(ml_dtypes.bfloat16)
            _CAPTURE.setdefault(int(layer), {})[name] = arr
        print(f"loaded capture from {CAPTURE_FILE}", flush=True)
    else:
        capture_real_activations(cfg)
        np.savez(CAPTURE_FILE, **{f"{l}/{n}": a for l, e in _CAPTURE.items() for n, a in e.items()})
        print(f"captured {len(_CAPTURE)} layers in {time.time() - t0:.0f}s -> {CAPTURE_FILE}", flush=True)
    e = _CAPTURE[0]
    print(f"shapes q{e['q'].shape} k{e['k'].shape} v{e['v'].shape} mask{e['mask'].shape} dO{e['d_out'].shape}", flush=True)

    per_layer = {}
    for layer in range(NUM_LAYERS):
        per_layer[layer] = analyse_layer(layer, _CAPTURE[layer])

    names = [n for n in per_layer[0] if not n.startswith("_")]
    print("\n=== SUMMARY over 18 layers (geometric mean of rel errors, mean of cosines) ===", flush=True)
    print(f"  {'variant':18s} {'dQ rel':>9s} {'cos(err,pred)':>13s} {'resid':>9s} {'sinkrow rel':>11s} | {'dK rel':>9s} {'cos':>7s} {'resid':>9s} | {'dV rel':>9s}")
    for name in names:
        gm = lambda key: float(np.exp(np.mean([np.log(max(per_layer[l][name][key], 1e-300)) for l in per_layer])))
        mean = lambda key: float(np.mean([per_layer[l][name][key] for l in per_layer]))
        print(
            f"  {name:18s} {gm('dq_rel'):9.3e} {mean('dq_cos'):13.4f} {gm('dq_resid'):9.3e} {gm('dq_sink_rel'):11.3e} |"
            f" {gm('dk_rel'):9.3e} {mean('dk_cos'):7.4f} {gm('dk_resid'):9.3e} | {gm('dv_rel'):9.3e}",
            flush=True,
        )
    for name in names:
        mean = lambda key: float(np.nanmean([per_layer[l][name][key] for l in per_layer]))
        print(
            f"  {name:18s} sink rows: frac along K_peak={mean('sink_frac_along_ksink'):.3f}"
            f" cos H1={mean('sink_cos_h1'):+.3f} cos H2={mean('sink_cos_h2'):+.3f} cos H1+H2={mean('sink_cos_h12'):+.3f}"
        )
    print("  predicted D-term rel norm per layer (dQ):", " ".join(f"{per_layer[l]['_pred_rel'][0]:.1e}" for l in per_layer))
    print("  sink share per layer:", " ".join(f"{per_layer[l]['_sink']['share']:.2f}" for l in per_layer))
    print("  sink keys per layer:", [sorted(set(per_layer[l]["_sink"]["sink_idx"])) for l in per_layer])


if __name__ == "__main__":
    main()
