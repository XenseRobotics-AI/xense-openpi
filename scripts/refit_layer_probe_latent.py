"""Offline refit of saved layer-probe features against the latent target (what the LTP head trains on)."""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "test")
from tactile_counterfactual import layer_probe as lp

HOR = (10, 20, 30, 40, 50)
LABELS = "assets/tactile_future_labels/bottle-sorting-0810"
RUNS = {
    "0915_1a": "outputs/tactile_layer_probes/linear/10k_2048",
    "0917_ltp": "outputs/tactile_layer_probes/linear/ltp10k_2048",
    "0917_ltp30k": "outputs/tactile_layer_probes/linear/ltp30k_2048",
}
OUT = "outputs/tactile_layer_probes/linear/latent_refit"
# runs already fitted in OUT/results.json are reused unless named on the command line (forces a refit)
FORCE = set(sys.argv[1:])
ALPHAS = (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)

fr = json.load(open(RUNS["0917_ltp"] + "/frames.json"))
ep, frm, is_val = np.array(fr["episode"]), np.array(fr["frame"]), np.array(fr["is_val"])
contact = np.array(fr["contact"])
n = len(ep)
store = lp.FutureTactileStore(LABELS, HOR, target="latent")
Y = np.zeros((n, 5, 4, 256), np.float32)
for i in range(n):
    z, m = store.targets(int(ep[i]), int(frm[i]))
    assert m.all()
    Y[i] = z
y = Y.reshape(n, -1)
tr, va = ~is_val, is_val
zc = np.load(LABELS + "/z_tac.npy", mmap_mode="r")
off = np.load(LABELS + "/episode_offsets.npy")
Zcur = np.asarray(zc[off[ep] + frm], np.float32)  # [n,4,256] current latent
print("Y var", y.var(0).mean())


def fit(x):
    model, scores = lp.fit_ridge(x[tr], y[tr], x[va], y[va], ALPHAS)
    e = lp.evaluate_ridge(model, x[va], y[va], (5, 4, 256))
    return {
        "nmse": e["nmse"],
        "alpha": model.alpha,
        "per_horizon": e.get("per_horizon"),
        "per_pad": e.get("per_pad"),
        "contact": lp.evaluate_ridge(model, x[va & contact], y[va & contact], (5, 4, 256))["nmse"],
        "noncontact": lp.evaluate_ridge(model, x[va & ~contact], y[va & ~contact], (5, 4, 256))["nmse"],
    }


prev = json.load(open(OUT + "/results.json")) if pathlib.Path(OUT + "/results.json").exists() else {}
res = {"baselines": {}}
# persistence: copy current z to every horizon
pred = np.repeat(Zcur[:, None], 5, axis=1).reshape(n, -1)
base = ((y[va] - y[tr].mean(0)) ** 2).sum()
res["baselines"]["persistence_copy"] = float(((pred[va] - y[va]) ** 2).sum() / base)
# ridge from current z (1024 dims) -> future z
res["baselines"]["ridge_from_current_z"] = fit(Zcur.reshape(n, -1))
print("baselines", json.dumps(res["baselines"], default=float)[:400])

for name, d in RUNS.items():
    if name in prev and name not in FORCE:
        res[name] = prev[name]
        print(f"{name}: reused from {OUT}/results.json")
        continue
    assert json.load(open(d + "/frames.json"))["episode"] == fr["episode"], f"{name}: frame set differs"
    meta = json.load(open(d + "/results.json"))["meta"]
    conds, taus, layers = meta["conditions"], meta["taus"], meta["layers"]
    F = np.load(d + "/features.npy", mmap_mode="r")
    res[name] = {}
    t0 = time.time()
    for ci, c in enumerate(conds):
        if c == "tac-shuffle":
            continue
        res[name][c] = {}
        for ti, tau in enumerate(taus):
            per = {}
            for li, layer in enumerate(layers[1:]):
                per[layer] = fit(np.asarray(F[ci, ti, li], np.float32))
            res[name][c][str(tau)] = per
            best = min(per, key=lambda k: per[k]["nmse"])
            print(
                f"{name} {c} tau={tau}: best {best} nmse={per[best]['nmse']:.4f}  l10={per['l10']['nmse']:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
json.dump(res, open(OUT + "/results.json", "w"), indent=1, default=float)

# report
L = ["l%d" % i for i in range(19)]
lines = [
    "# Latent-target refit (same 2048 frames, same split)",
    "",
    f"baselines: persistence copy nMSE {res['baselines']['persistence_copy']:.4f}; ridge from current z {res['baselines']['ridge_from_current_z']['nmse']:.4f}",
    "",
]
for tau in ("0.25", "1.0"):
    lines += [
        f"## tau={tau}: nMSE real / null-real gain",
        "",
        "| layer | " + " | ".join(f"{name} real | {name} gain" for name in RUNS) + " |",
        "|" + "---|" * (1 + 2 * len(RUNS)),
    ]
    for l in L:
        r = []
        for name in RUNS:
            a = res[name]["real"][tau][l]["nmse"]
            g = res[name]["null"][tau][l]["nmse"] - a
            r += [f"{a:.4f}", f"{g:+.4f}"]
        lines.append(f"| {l} | " + " | ".join(r) + " |")
    lines.append("")
    for name in RUNS:
        b = res[name]["real"][tau]
        bl = min(b, key=lambda k: b[k]["nmse"])
        lines.append(
            f"- {name} best {bl} nmse {b[bl]['nmse']:.4f}; per_horizon {b[bl]['per_horizon']}; per_pad {b[bl]['per_pad']}; contact {b[bl]['contact']:.4f} / noncontact {b[bl]['noncontact']:.4f}"
        )
    lines.append("")
open(OUT + "/report.md", "w").write("\n".join(lines))
print("done")
