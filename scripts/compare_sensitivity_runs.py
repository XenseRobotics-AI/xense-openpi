"""Side-by-side table of layer-wise sensitivity probe runs (S_tac / S_vl / share at a few points)."""

import argparse
import json
import pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label=path/to/sensitivity/<run> pairs")
    p.add_argument("--tau", default="1.0")
    p.add_argument("--points", nargs="+", default=["l6", "l8", "l9", "l10", "l11", "l14", "l18", "v_t", "chunk"])
    a = p.parse_args()
    runs = {}
    for r in a.runs:
        lab, path = r.split("=", 1)
        runs[lab] = json.loads(pathlib.Path(path, "results.json").read_text())

    def get(d, point, var, sub="all"):
        blk = d["results"]["chunk"] if point == "chunk" else d["results"][a.tau]
        key = "chunk" if point == "chunk" else point
        return blk[key][var][sub]["mean"]

    labs = list(runs)
    print(f"tau={a.tau}; columns per run: S_tac | S_vl | share  (share = S_tac / S_x)")
    print("| point | " + " | ".join(f"{lab} S_tac | {lab} S_vl | {lab} share" for lab in labs) + " |")
    print("|---|" + "---|" * (3 * len(labs)))
    for pt in a.points:
        cells = []
        for lab in labs:
            d = runs[lab]
            st, sv, sx = get(d, pt, "tac-shuffle"), get(d, pt, "vl-swap"), get(d, pt, "x")
            cells += [f"{st:.2e}", f"{sv:.3g}", f"{st / sx:.2e}"]
        print(f"| {pt} | " + " | ".join(cells) + " |")
    print()
    print("share at v_t, contact / non-contact, per tau:")
    for lab in labs:
        v = runs[lab]["verdict"]["share_at_v_t"]
        print(f"- {lab}: " + "; ".join(f"tau {t}: {v[t]['contact']:.2e} / {v[t]['noncontact']:.2e}" for t in v))
    print()
    print("tactile-token S_x (collapse check) and pad/vl at v_t:")
    for lab in labs:
        vd = runs[lab]["verdict"]
        print(f"- {lab}: S_x {vd['tactile_token_cross_sample_S_x']:.3f}; pad/vl {vd['pad_pert_over_vl_at_v_t']}")


if __name__ == "__main__":
    main()
