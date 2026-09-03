"""Check that the direct-rule float16 cuDNN VJP in gemma.py matches the recompute-based one.

`gemma._cudnn_attention_in_dtype` now calls JAX's cuDNN fwd/bwd rules directly
(keeping softmax stats + output as residuals). This script re-implements the
previous version (jax.vjp over jax.nn.dot_product_attention inside the backward)
and compares forward outputs and dQ/dK/dV on real captured activations. The two
backwards run the same kernel on the same inputs, so their difference must be at
the level of the kernel's own run-to-run (atomic accumulation) noise.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src python scripts/cudnn_fp16_vjp_equivalence.py \
        /root/localstorage/logs/dterm_capture.npz
"""

import sys
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import openpi.models.gemma as _gemma

CAPTURE_FILE = sys.argv[1] if len(sys.argv) > 1 else "/root/localstorage/logs/dterm_capture.npz"
LAYERS = (0, 4, 8, 14, 17)


def _reference_in_dtype(q, k, v, attn_mask, compute_dtype):
    """The previous implementation: recompute the forward inside the backward via jax.vjp."""
    out_dtype = q.dtype

    @jax.custom_vjp
    def attention(q, k, v, attn_mask):
        qc, kc, vc = (x.astype(compute_dtype) for x in (q, k, v))
        return _gemma._cudnn_attention_call(qc, kc, vc, attn_mask).astype(out_dtype)

    def attention_fwd(q, k, v, attn_mask):
        return attention(q, k, v, attn_mask), (q, k, v, attn_mask)

    def attention_bwd(residuals, d_out):
        q, k, v, attn_mask = residuals
        qc, kc, vc = (x.astype(compute_dtype) for x in (q, k, v))
        d_out32 = d_out.astype(jnp.float32)
        max_abs = jnp.max(jnp.abs(d_out32))
        tiny = jnp.finfo(jnp.float32).tiny
        exponent = jnp.floor(jnp.log2(jnp.maximum(max_abs, tiny)))
        scale = jnp.where(max_abs > 0, jnp.exp2(-(exponent + 2.0)), 1.0)
        _, vjp_fn = jax.vjp(lambda a, b, c: _gemma._cudnn_attention_call(a, b, c, attn_mask), qc, kc, vc)
        dq, dk, dv = vjp_fn((d_out32 * scale).astype(compute_dtype))
        inv_scale = 1.0 / scale
        return (*((g.astype(jnp.float32) * inv_scale).astype(out_dtype) for g in (dq, dk, dv)), None)

    attention.defvjp(attention_fwd, attention_bwd)
    return attention(q, k, v, attn_mask)


def _fwd_and_grads(fn, q, k, v, mask, d_out):
    d32 = d_out.astype(jnp.float32)

    def objective(q_, k_, v_):
        out = fn(q_, k_, v_, mask, jnp.float16)
        return jnp.sum(out.astype(jnp.float32) * d32), out

    (_, out), grads = jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True)(q, k, v)
    return out, grads


new_fn = jax.jit(lambda *a: _fwd_and_grads(_gemma._cudnn_attention_in_dtype, *a))
old_fn = jax.jit(lambda *a: _fwd_and_grads(_reference_in_dtype, *a))


def _f64(x):
    return np.asarray(jax.device_get(x)).astype(np.float64)


def _rel(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2) / max(np.sum(b**2), 1e-300)))


def main():
    data = np.load(CAPTURE_FILE)
    for layer in LAYERS:
        e = {}
        for name in ("q", "k", "v", "mask", "d_out"):
            arr = data[f"{layer}/{name}"]
            if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
                arr = arr.view(ml_dtypes.bfloat16)
            e[name] = jnp.asarray(arr)
        args = (e["q"], e["k"], e["v"], e["mask"], e["d_out"])
        t0 = time.time()
        out_new, g_new = new_fn(*args)
        _, g_new2 = new_fn(*args)
        out_old, g_old = old_fn(*args)
        _, g_old2 = old_fn(*args)
        jax.block_until_ready((g_new, g_new2, g_old, g_old2))
        valid_q = np.asarray(jnp.any(e["mask"], axis=-1)[:, 0, :])
        dq_new = _f64(g_new[0])
        masked_dq = float(np.abs(dq_new[~valid_q]).max()) if (~valid_q).any() else 0.0
        nonfinite = sum(int(np.count_nonzero(~np.isfinite(_f64(x)))) for x in (*g_new, out_new))
        print(
            f"layer {layer:2d}: fwd new-vs-old rel={_rel(_f64(out_new), _f64(out_old)):.2e}"
            f" (bit-identical: {bool(np.array_equal(np.asarray(out_new), np.asarray(out_old)))})"
            f" | new-vs-old dQ {_rel(dq_new, _f64(g_old[0])):.2e} dK {_rel(_f64(g_new[1]), _f64(g_old[1])):.2e}"
            f" dV {_rel(_f64(g_new[2]), _f64(g_old[2])):.2e}"
            f" | new run-to-run dQ {_rel(dq_new, _f64(g_new2[0])):.2e} old run-to-run dQ {_rel(_f64(g_old[0]), _f64(g_old2[0])):.2e}"
            f" | masked-row |dQ| max={masked_dq:.1e} nonfinite={nonfinite} ({time.time() - t0:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
