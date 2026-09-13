"""Verify the CUDA/cuDNN stack a JAX training process will actually load.

Checking cuda_versions.cudnn_get_version() alone is NOT sufficient: the number it
returns comes from the cuDNN sub-libraries, so a 9.10.2 dispatcher stub driving
9.14 engine libraries still reports 91400. This walks /proc/self/maps instead and
then exercises the real production attention shape.

Run it in the SAME shell that launches training:
    python scripts/check_cuda_stack.py
"""

import collections
import importlib.metadata
import os
import re
import sys

FAIL = []
# Major.minor only: pyproject pins these as ranges (torch >=2.11,<2.12 and friends), so an
# exact pin here would fail the gate on every upstream patch release even though the
# environment is still the supported one.
EXPECTED = {
    "torch": "2.11",
    "torchvision": "0.26",
    "torchcodec": "0.11",
    "nvidia-cudnn-cu12": "9.19",
    "cuda": "12.8",
}
# The three torch wheels must come from the cu128 index. A CPU-only wheel installs cleanly
# and silently removes everything this script is here to check.
CU_SUFFIX = "+cu128"
CU_WHEELS = ("torch", "torchvision", "torchcodec")
EXPECTED_CUDNN = (9, 19)


def major_minor(version):
    """'2.11.0+cu128' -> '2.11'. Also handles '9.19.0.56' and '12.8'."""
    return ".".join(re.split(r"[.+]", str(version))[:2])


def loaded_libs(pattern):
    out = collections.OrderedDict()
    with open("/proc/self/maps") as f:
        for line in f:
            m = re.search(r"(/\S*" + pattern + r"\S*)", line)
            if m:
                out[m.group(1)] = None
    return list(out)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def report(name, pattern):
    libs = loaded_libs(pattern)
    if not libs:
        print(f"  {name}: (none loaded)")
        return
    dirs = {os.path.dirname(p) for p in libs}
    for p in libs:
        print(f"    {p}")
    if len(dirs) > 1:
        FAIL.append(f"{name} is loaded from {len(dirs)} different directories: {sorted(dirs)}")
        print(f"  !! {name}: MIXED — {len(dirs)} source directories")
    else:
        print(f"  OK {name}: single source {dirs.pop()}")


def main():
    import jax
    from jax._src.lib import cuda_versions
    import jax.numpy as jnp
    import numpy as np
    import torch
    import torchcodec
    import torchvision

    import openpi.models.gemma as _gemma

    section("1. reported versions")
    rt, build = cuda_versions.cudnn_get_version(), cuda_versions.cudnn_build_version()
    cudnn_package_version = importlib.metadata.version("nvidia-cudnn-cu12")
    torch_rt = torch.backends.cudnn.version()
    print(f"  jax                    : {jax.__version__}")
    print(f"  torch                  : {torch.__version__}")
    print(f"  torchvision            : {torchvision.__version__}")
    print(f"  torchcodec             : {torchcodec.__version__}")
    print(f"  nvidia-cudnn-cu12      : {cudnn_package_version}")
    print(f"  torch CUDA             : {torch.version.cuda}")
    print(f"  torch cuDNN runtime    : {torch_rt}")
    print(f"  cuDNN runtime (reported): {rt}")
    print(f"  cuDNN build            : {build}")
    print(f"  LD_LIBRARY_PATH        : {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")

    actual = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchcodec": torchcodec.__version__,
        "nvidia-cudnn-cu12": cudnn_package_version,
        "cuda": torch.version.cuda,
    }
    for name, expected in EXPECTED.items():
        if major_minor(actual[name]) != expected:
            FAIL.append(f"{name} must be {expected}.x, got {actual[name]}")
    FAIL.extend(
        f"{name} must be a {CU_SUFFIX} wheel, got {actual[name]}"
        for name in CU_WHEELS
        if not str(actual[name]).endswith(CU_SUFFIX)
    )
    if (rt // 10000, rt // 100 % 100) != EXPECTED_CUDNN:
        FAIL.append(f"cuDNN runtime must be {EXPECTED_CUDNN[0]}.{EXPECTED_CUDNN[1]}.x, got {rt}")
    if torch_rt != rt:
        FAIL.append(f"PyTorch reports cuDNN {torch_rt}, while JAX reports {rt}")

    # Force the CUDA libraries to actually load before inspecting the memory map.
    dev = jax.devices()[0]
    jnp.ones((8, 8)).block_until_ready()
    print(f"  device                 : {dev} (compute {dev.compute_capability})")

    section("2. libraries actually mapped into this process")
    report("libcudnn", "libcudnn")
    report("libcublas", "libcublas")
    report("libnccl", "libnccl")

    # A version embedded in a filename that disagrees with the reported runtime is
    # the exact failure that produced a 9.10.2 dispatcher over 9.14 engines.
    vers = set()
    for p in loaded_libs("libcudnn"):
        m = re.search(r"\.so\.(\d+)\.(\d+)\.(\d+)", p)
        if m:
            vers.add(f"{m.group(1)}.{m.group(2)}.{m.group(3)}")
    if len(vers) > 1:
        FAIL.append(f"cuDNN filenames span multiple versions: {sorted(vers)}")
        print(f"  !! cuDNN file versions: MIXED {sorted(vers)}")
    elif vers:
        v = vers.pop()
        maj, minor, patch = (int(x) for x in v.split("."))
        want = maj * 10000 + minor * 100 + patch
        print(f"  cuDNN file version     : {v}")
        if want != rt:
            FAIL.append(f"cuDNN filename says {v} ({want}) but runtime reports {rt}")
            print(f"  !! filename {want} != reported runtime {rt}")

    section("3. production attention shape: forward + backward vs fp32 reference")
    # pi0.5 training shape: 3x256 image + 200 prompt + 50 action tokens, GQA 8:1,
    # head_dim 256, bf16, real block mask including fully-masked padding rows.
    batch_size, num_heads, num_kv_heads, head_dim = 2, 8, 1, 256
    seq_len = 768 + 200 + 50

    im = np.ones((batch_size, seq_len), dtype=bool)
    for b, plen in enumerate((120, 165)):
        im[b, 768 + plen : 768 + 200] = False
    ar = np.zeros((seq_len,), dtype=bool)
    ar[768 + 200] = True
    im_j = jnp.asarray(im)
    cs = jnp.broadcast_to(jnp.cumsum(jnp.asarray(ar), -1), im_j.shape)
    mask = jnp.logical_and(cs[:, None, :] <= cs[:, :, None], im_j[:, None, :] * im_j[:, :, None])[:, None]
    valid = np.asarray(jnp.any(mask, -1))[0, 0]
    n_empty = int((~np.asarray(jnp.any(mask, -1))).sum())
    print(
        f"  shape batch_size={batch_size} seq_len={seq_len} heads={num_heads} kv={num_kv_heads} head_dim={head_dim} bf16"
    )
    print(f"  fully-masked query rows in batch: {n_empty}")

    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.normal(0, 10 / np.sqrt(head_dim), (batch_size, seq_len, num_heads, head_dim)), jnp.bfloat16)
    k = jnp.asarray(rng.normal(0, 1, (batch_size, seq_len, num_kv_heads, head_dim)), jnp.bfloat16)
    v = jnp.asarray(rng.normal(0, 1, (batch_size, seq_len, num_kv_heads, head_dim)), jnp.bfloat16)
    cot = np.asarray(rng.normal(0, 1, (batch_size, seq_len, num_heads, head_dim)), np.float32)
    cot[:, : seq_len - 50] = 0.0  # only action tokens carry loss, as in Pi0
    cot = jnp.asarray(cot)

    def reference(q, k, v, m):  # explicit path, fp32 throughout
        q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
        gq = q.reshape(batch_size, seq_len, num_kv_heads, num_heads // num_kv_heads, head_dim)
        lg = jnp.einsum("BTKGH,BSKH->BKGTS", gq, k, preferred_element_type=jnp.float32)
        lg = jnp.where(m[:, :, None], lg, -2.3819763e38)
        return jnp.einsum("BKGTS,BSKH->BTKGH", jax.nn.softmax(lg, -1), v).reshape(
            batch_size, seq_len, num_heads, head_dim
        )

    def cudnn(q, k, v, m):  # what training runs with use_cudnn_attention=true
        has_key = jnp.any(m, -1)[:, 0, :, None, None]
        q = jnp.where(has_key, q, jax.lax.stop_gradient(q))
        return jax.nn.dot_product_attention(q, k, v, mask=m, scale=1.0, implementation="cudnn")

    def loss(fn):
        return lambda q, k, v, m: jnp.sum(fn(q, k, v, m).astype(jnp.float32) * cot)

    try:
        gr = [
            np.asarray(x.astype(jnp.float32), np.float64) for x in jax.grad(loss(reference), (0, 1, 2))(q, k, v, mask)
        ]
        gc = [np.asarray(x.astype(jnp.float32), np.float64) for x in jax.grad(loss(cudnn), (0, 1, 2))(q, k, v, mask)]
    except Exception as exc:
        FAIL.append(f"cuDNN attention raised {type(exc).__name__}: {exc}")
        print(f"  !! cuDNN attention FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        gr = gc = None

    if gc is not None:
        for i, nm in enumerate(("dQ", "dK", "dV")):
            a, b = gc[i][:, valid], gr[i][:, valid]
            rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
            nan = int(np.isnan(gc[i]).sum())
            status = "OK" if (nan == 0 and rel < 2e-2) else "!!"
            if nan or rel >= 2e-2:
                FAIL.append(f"{nm}: rel-L2={rel:.3e} NaN={nan}")
            print(f"  {status} {nm}: rel-L2 vs fp32 = {rel:.3e}   NaN count = {nan}   (bf16 noise is ~3e-3)")

    section("4. production FP16 custom VJP: forward + backward vs fp32 reference")

    # Section 3 exercises the raw bf16 kernel, which docs/training-optimization.md marks
    # unsafe for production. This is the path a production config actually runs:
    #   model: {use_cudnn_attention: true, cudnn_attention_dtype: float16}
    def cudnn_fp16(q, k, v, m):
        return _gemma._cudnn_attention_in_dtype(q, k, v, m, jnp.float16)

    gf = None
    try:
        gf = [
            np.asarray(x.astype(jnp.float32), np.float64) for x in jax.grad(loss(cudnn_fp16), (0, 1, 2))(q, k, v, mask)
        ]
    except Exception as exc:
        FAIL.append(f"FP16 cuDNN custom VJP raised {type(exc).__name__}: {exc}")
        print(f"  !! FP16 cuDNN custom VJP FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    if gf is not None and gr is not None:
        for i, nm in enumerate(("dQ", "dK", "dV")):
            a, b = gf[i][:, valid], gr[i][:, valid]
            rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
            nan = int(np.isnan(gf[i]).sum())
            status = "OK" if (nan == 0 and rel < 2e-2) else "!!"
            if nan or rel >= 2e-2:
                FAIL.append(f"FP16 {nm}: rel-L2={rel:.3e} NaN={nan}")
            print(f"  {status} {nm}: rel-L2 vs fp32 = {rel:.3e}   NaN count = {nan}   (expect below the bf16 row)")

        # The custom VJP promises a zero -- never NaN, never garbage -- q-gradient on the
        # fully masked rows that padding produces. Check that at the production shape.
        empty = ~valid
        if empty.any():
            nonzero = int(np.count_nonzero(gf[0][:, empty]))
            status = "OK" if nonzero == 0 else "!!"
            if nonzero:
                FAIL.append(f"FP16 dQ is nonzero on {nonzero} fully-masked query entries")
            print(f"  {status} dQ on {int(empty.sum())} fully-masked query rows: {nonzero} nonzero entries")

    section("verdict")
    if FAIL:
        print("  FAIL — do not start production training:")
        for f in FAIL:
            print(f"    - {f}")
        return 1
    print("  PASS — single-source cuDNN stack, production-shape backward finite and accurate,")
    print("         for both the raw bf16 kernel and the FP16 custom VJP.")
    print("  NOTE: this gates the kernel only. It does NOT prove training converges;")
    print("        the 2026-08-30 divergence passed every check of this kind and still")
    print("        diverged between step 1000 and 1200. Watch the loss curve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
