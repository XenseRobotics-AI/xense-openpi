"""CPU-only follow-up to attention_dterm_probe.py: anatomy of the peaked attention rows.

Loads the captured q/k/v/mask/d_out of every layer (dterm_capture.npz) and, in float64,
reconstructs the exact single-layer backward for the peaked rows to answer:

  * how big are K/V of the BOS sink key (position 768) relative to other keys?
  * which rows are peaked (maxP > 0.99): BOS-attending rows or other keys?
  * on BOS rows, how much cancellation does the true dQ have, and how large is the
    D-term error from rounding the stored O to bf16 (H1) relative to it -- with and
    without shifting V by V_sink?

Pure numpy; run anywhere the npz is:  python scripts/attention_dterm_cpu_analysis.py
"""

import sys

import ml_dtypes
import numpy as np

CAPTURE_FILE = sys.argv[1] if len(sys.argv) > 1 else "/root/localstorage/logs/dterm_capture.npz"
BIG_NEG = -2.3819763e38
SINK = 768
MAX_ROWS = 3000
rng = np.random.default_rng(0)


def region(pos):
    if pos < 768:
        return f"img{pos // 256}"
    if pos < 968:
        return "txt"
    return "act"


def bf16_round(x):
    return x.astype(ml_dtypes.bfloat16).astype(np.float64)


def load():
    data = np.load(CAPTURE_FILE)
    layers = {}
    for key in data.files:
        layer, name = key.split("/")
        arr = data[key]
        if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:  # bf16 saved by np.savez as raw bytes
            arr = arr.view(ml_dtypes.bfloat16)
        layers.setdefault(int(layer), {})[name] = arr
    return layers


def row_anatomy(q_rows, k, v, mask_rows, d_rows, shift=None):
    """q_rows/d_rows: R x H ; k, v: S x H (single kv head) ; mask_rows: R x S bool. float64."""
    s = q_rows @ k.T  # R S
    s = np.where(mask_rows, s, -np.inf)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    v_eff = v if shift is None else v - shift[None, :]
    o = p @ v_eff  # R H   (fp64 "true" output of the kernel-internal V)
    dp = d_rows @ v_eff.T  # R S
    d_true = np.sum(p * dp, axis=1)  # R
    ds = p * (dp - d_true[:, None])
    dq = ds @ k  # R H
    # H1: kernel D from bf16-rounded O
    delta = np.sum(d_rows * (bf16_round(o) - o), axis=1)  # R
    pk = p @ k  # R H
    err_h1 = -delta[:, None] * pk
    return dict(p=p, o=o, dp=dp, d_true=d_true, ds=ds, dq=dq, delta=delta, pk=pk, err_h1=err_h1)


def norms(x):
    return np.sqrt(np.sum(x * x, axis=-1))


def main():
    layers = load()
    print(f"loaded {len(layers)} layers from {CAPTURE_FILE}", flush=True)
    for layer in sorted(layers):
        e = layers[layer]
        q = e["q"].astype(np.float64)  # B T N H
        k = e["k"].astype(np.float64)[:, :, 0]  # B S H
        v = e["v"].astype(np.float64)[:, :, 0]
        mask = e["mask"][:, 0]  # B T S
        d = e["d_out"].astype(np.float64)  # B T N H
        B, T, N, H = q.shape
        valid_q = mask.any(-1)  # B T
        valid_k = mask.any(1)  # B S

        kn, vn = norms(k), norms(v)  # B S
        med_k = np.median(kn[valid_k])
        med_v = np.median(vn[valid_k])
        print(
            f"\n=== layer {layer:2d} ===  |K_768|={kn[:, SINK].mean():6.2f} (median valid |K|={med_k:6.2f})"
            f"   |V_768|={vn[:, SINK].mean():6.2f} (median valid |V|={med_v:6.2f})",
            flush=True,
        )

        # per-row softmax stats for all valid rows (float32 logits are exact enough for argmax/maxP)
        s = np.einsum("btnh,bsh->btns", q.astype(np.float32), k.astype(np.float32))
        s = np.where(mask[:, :, None, :], s, BIG_NEG)
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        max_p = p.max(-1)  # B T N
        arg_p = p.argmax(-1)
        del s, p
        vrow = np.broadcast_to(valid_q[:, :, None], max_p.shape)
        peaked = (max_p > 0.99) & vrow
        bos_rows = (arg_p == SINK) & vrow
        n_peaked = int(peaked.sum())
        n_peaked_bos = int((peaked & bos_rows).sum())
        hist = {}
        for pos in arg_p[peaked & ~bos_rows]:
            hist[region(int(pos))] = hist.get(region(int(pos)), 0) + 1
        print(
            f"  peaked rows (maxP>0.99): {n_peaked}  of which BOS(768): {n_peaked_bos}; other peaked keys by region: {hist}"
            f"   BOS-argmax rows: {int(bos_rows.sum())} ({bos_rows.sum() / vrow.sum():.1%} of valid rows),"
            f" P>0.9: {int((bos_rows & (max_p > 0.9)).sum())}, P>0.99: {n_peaked_bos}",
            flush=True,
        )

        def analyse(sel, label, shift_by_sink):
            idx = np.argwhere(sel)
            if len(idx) == 0:
                print(f"  {label}: no rows", flush=True)
                return
            if len(idx) > MAX_ROWS:
                idx = idx[rng.choice(len(idx), MAX_ROWS, replace=False)]
            out = {"plain": [], "shift": []}
            for b in np.unique(idx[:, 0]):
                rows = idx[idx[:, 0] == b]
                t_idx, n_idx = rows[:, 1], rows[:, 2]
                q_rows = q[b, t_idx, n_idx]
                d_rows = d[b, t_idx, n_idx]
                m_rows = mask[b, t_idx]
                for key, shift in (("plain", None), ("shift", v[b, SINK] if shift_by_sink else None)):
                    if key == "shift" and not shift_by_sink:
                        continue
                    r = row_anatomy(q_rows, k[b], v[b], m_rows, d_rows, shift=shift)
                    eps = 1.0 - r["p"].max(1)
                    dq_n = norms(r["dq"])
                    naive = eps * norms(d_rows) * med_v * med_k  # first-order scale of a peaked row's dQ
                    ds_sink = np.abs(np.take_along_axis(r["ds"], r["p"].argmax(1)[:, None], 1)[:, 0])
                    h1_n = norms(r["err_h1"])
                    k_arg = k[b][r["p"].argmax(1)]  # R H
                    cos_k = np.sum(r["err_h1"] * k_arg, 1) / np.maximum(h1_n * norms(k_arg), 1e-300)
                    out[key].append(
                        np.stack(
                            [eps, dq_n, naive, ds_sink * norms(k_arg), np.abs(r["d_true"]), np.abs(r["delta"]), h1_n, cos_k, norms(r["o"])],
                            1,
                        )
                    )
            for key, chunks in out.items():
                if not chunks:
                    continue
                a = np.concatenate(chunks)
                eps, dq_n, naive, dssink_k, d_true, delta, h1_n, cos_k, o_n = a.T
                gm = lambda x: float(np.exp(np.mean(np.log(np.maximum(x, 1e-300)))))
                print(
                    f"  {label:34s} [{key:5s}] rows={len(a):5d}  eps={gm(eps):.2e}  |dQ_true|={gm(dq_n):.2e}"
                    f"  naive eps|dO||V||K|={gm(naive):.2e}  |dS_sink||K|={gm(dssink_k):.2e}  |D|={gm(d_true):.2e}"
                    f"  |delta_H1|={gm(delta):.2e}  |O|={gm(o_n):.2e}"
                    f"  |errH1|/|dQ_true|: gm={gm(h1_n / np.maximum(dq_n, 1e-300)):.2e} med={np.median(h1_n / np.maximum(dq_n, 1e-300)):.2e}"
                    f"  cos(errH1,K_argmax)={np.mean(np.abs(cos_k)):.2f}",
                    flush=True,
                )

        analyse(bos_rows & (max_p > 0.99), "BOS rows P>0.99", True)
        analyse(bos_rows & (max_p > 0.9) & (max_p <= 0.99), "BOS rows 0.9<P<=0.99", True)
        analyse(bos_rows & (max_p <= 0.9), "BOS rows P<=0.9", True)
        analyse(peaked & ~bos_rows, "other peaked rows P>0.99", False)
        analyse(vrow & ~bos_rows & (max_p <= 0.9), "ordinary rows (argmax!=768,P<=0.9)", False)


if __name__ == "__main__":
    main()
