"""Time the attention variants used by gemma.Attention at the training shape on one GPU.

Answers "where does the float16 cuDNN path lose time against the bfloat16 cuDNN
path?" by timing forward and forward+backward of each variant on real captured
activations (dterm_capture.npz, B=8, T=S=1018, 8 q heads / 1 kv head, head_dim 256),
optionally tiled along batch to the per-device training batch.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src python scripts/cudnn_attention_microbench.py \
        /root/localstorage/logs/dterm_capture.npz [batch_tile]
"""

import sys
import time

import einops
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import openpi.models.gemma as _gemma

CAPTURE_FILE = sys.argv[1] if len(sys.argv) > 1 else "/root/localstorage/logs/dterm_capture.npz"
TILE = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # 8 * 4 = 32 = per-device batch at global batch 256 on 8 GPUs
LAYER = 0
ITERS = 20


def explicit_bf16(q, k, v, mask):
    num_kv_heads = k.shape[2]
    grouped_q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", grouped_q, k, preferred_element_type=jnp.float32)
    masked_logits = jnp.where(mask[:, :, None, :, :], logits, -2.3819763e38)
    probs = jax.nn.softmax(masked_logits, axis=-1).astype(v.dtype)
    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v).astype(q.dtype)
    return einops.rearrange(encoded, "B T K G H -> B T (K G) H")


def cudnn_bf16(q, k, v, mask):
    return _gemma._cudnn_attention_call(q, k, v, mask)


def cudnn_fp16_plain(q, k, v, mask):
    # float16 kernel through jax's own custom_vjp, no loss scaling: isolates kernel cost.
    q16, k16, v16 = (x.astype(jnp.float16) for x in (q, k, v))
    return _gemma._cudnn_attention_call(q16, k16, v16, mask).astype(q.dtype)


def cudnn_fp16_v2(q, k, v, mask):
    return _gemma._cudnn_attention_in_dtype(q, k, v, mask, jnp.float16)


VARIANTS = {
    "cudnn_bf16": cudnn_bf16,
    "cudnn_fp16_plain": cudnn_fp16_plain,
    "cudnn_fp16_v2": cudnn_fp16_v2,
    "explicit_bf16": explicit_bf16,
}


def _timeit(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / ITERS * 1e3


def main():
    data = np.load(CAPTURE_FILE)
    e = {}
    for name in ("q", "k", "v", "mask", "d_out"):
        arr = data[f"{LAYER}/{name}"]
        if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
            arr = arr.view(ml_dtypes.bfloat16)
        if TILE > 1:
            arr = np.concatenate([arr] * TILE, axis=0)
        e[name] = jnp.asarray(arr)
    q, k, v, mask, d_out = e["q"], e["k"], e["v"], e["mask"], e["d_out"]
    print(f"shape q={q.shape} k={k.shape} mask={mask.shape} dtype={q.dtype} device={jax.devices()[0]}", flush=True)
    d32 = d_out.astype(jnp.float32)

    for name, fn in VARIANTS.items():
        fwd = jax.jit(fn)

        def objective(q_, k_, v_, fn=fn):
            return jnp.sum(fn(q_, k_, v_, mask).astype(jnp.float32) * d32)

        fwd_bwd = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))
        # Same structure as training: flax remat(nothing_saveable) recomputes the forward in the backward.
        fwd_bwd_remat = jax.jit(
            jax.grad(
                lambda q_, k_, v_, fn=fn: jnp.sum(
                    jax.checkpoint(lambda a, b, c: fn(a, b, c, mask), policy=jax.checkpoint_policies.nothing_saveable)(
                        q_, k_, v_
                    ).astype(jnp.float32)
                    * d32
                ),
                argnums=(0, 1, 2),
            )
        )
        t_fwd = _timeit(fwd, q, k, v, mask)
        t_fb = _timeit(fwd_bwd, q, k, v)
        t_fbr = _timeit(fwd_bwd_remat, q, k, v)
        print(
            f"{name:18s} fwd {t_fwd:7.2f} ms | fwd+bwd {t_fb:7.2f} ms | fwd+bwd(remat) {t_fbr:7.2f} ms"
            f" | x18 layers: fwd+bwd(remat) {t_fbr * 18 / 1e3:.3f} s/step",
            flush=True,
        )


if __name__ == "__main__":
    main()
