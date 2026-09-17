#!/usr/bin/env python3
"""Layer-wise linear probe: which action-expert layer carries the future tactile field?

Step 1, item 2 of docs/action-conditioned-tactile-pretraining.md (RATG section 3.2). For
one frozen Pi0TactileFastVit checkpoint the script

1. draws frames across episodes (contact / non-contact stratified through the label
   store's pixel field), decodes them through the training data pipeline,
2. runs the training forward at fixed flow times ``tau`` and fixed noise, and pools the
   action-position residual stream after every action-expert block (plus the suffix input
   ``l0`` and the pooled VLM output ``vlm``) into one feature vector per frame,
3. fits a closed-form ridge from each layer's features to the future tactile field of
   section 4.2 (``pixel_delta``: 16x16x3 change over the next 10..50 frames, per pad,
   normalised by the store's RMS; or ``latent``: the PCA-whitened frozen-FastViT latents),
4. reports the validation nMSE (1 - R^2 against the train-mean predictor; zero
   prediction ~ 1) per layer, per tau, for the tactile input as recorded (``real``),
   masked out (``null``) and rolled inside the batch (``tac-shuffle``).

The layer with the smallest ``real`` nMSE is the candidate ``m`` for step 2; the
``null - real`` gap per layer says how much of that predictability came from the tactile
tokens rather than from vision / state / the (partly clean) action chunk.

Usage (from the repo root, training environment):

    python scripts/probe_future_tactile_layers.py \\
        --config-name pi05_base_bi_flexiv_bottle_sorting_0915_fastvit_h100 \\
        --checkpoint-dir checkpoints/pi05_base_bi_flexiv_bottle_sorting_0915_fastvit_h100/10000 \\
        --labels-dir assets/tactile_future_labels/bottle-sorting-0810 \\
        --num-frames 2048 --batch-size 16 --num-workers 8

    # quick smoke test
    python scripts/probe_future_tactile_layers.py ... --num-frames 64 --taus 1.0 --conditions real

Outputs (``--out``, default ``outputs/tactile_layer_probes/linear/<timestamp>``):
``results.json`` (every fit), ``report.md`` (tables + layer choice), ``frames.json`` (the
sampled frames, split and contact flags), ``features.npy`` (float16
``[cond, tau, layer, frame, dim]`` memmap; refit offline without a GPU) and
``targets.npy``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "packages" / "xense-client" / "src"))

import numpy as np  # noqa: E402

logger = logging.getLogger("probe_future_tactile_layers")

CONDITIONS = ("real", "null", "tac-shuffle")
FEATURE_MODES = ("mean", "meanpos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", required=True, help="training config name (configs/*.yaml or configs/_examples/)")
    p.add_argument("--checkpoint-dir", required=True, help="checkpoint step directory (holds params/ and assets/)")
    p.add_argument("--labels-dir", required=True, help="label store from scripts/compute_tactile_future_labels.py")
    p.add_argument("--target", default="pixel_delta", choices=("pixel_delta", "latent"))
    p.add_argument("--repo-id", default=None, help="override the data config's LeRobot repo id")
    p.add_argument("--num-frames", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--taus", type=float, nargs="+", default=None, help="flow times (default 0.25 0.5 0.75 1.0)")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument(
        "--features",
        default="mean",
        choices=FEATURE_MODES,
        help="mean: mean over the action positions (1024 dims). meanpos: mean plus the action token at "
        "step k-1 for every horizon k (6 x 1024 dims; the dual-form ridge keeps it cheap).",
    )
    p.add_argument("--val-fraction", type=float, default=0.25, help="fraction of the sampled episodes held out")
    p.add_argument("--alphas", type=float, nargs="+", default=None, help="ridge penalties relative to trace(X^T X)/d")
    p.add_argument("--contact-quantile", type=float, default=0.5)
    p.add_argument("--contact-proxy", default="delta", choices=("delta", "state"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--cudnn-attention", action="store_true", help="keep the config's cuDNN attention (default: off)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def pool_features(action_stream, mode: str, horizons, action_horizon: int):
    """``[L+1, B, ah, D]`` -> ``[L+1, B, d]`` (jax arrays)."""
    import jax.numpy as jnp

    mean = jnp.mean(action_stream, axis=2)
    if mode == "mean":
        return mean
    positions = [min(int(k) - 1, action_horizon - 1) for k in horizons]
    return jnp.concatenate([mean] + [action_stream[:, :, p] for p in positions], axis=-1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("openpi").setLevel(logging.WARNING)

    import jax
    import jax.numpy as jnp

    from test.tactile_counterfactual import layer_probe as lp

    taus = tuple(args.taus) if args.taus else lp.TAUS
    alphas = tuple(args.alphas) if args.alphas else lp.DEFAULT_ALPHAS
    conditions = tuple(args.conditions)
    out_dir = pathlib.Path(
        args.out or _ROOT / "outputs" / "tactile_layer_probes" / "linear" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out_dir)

    t0 = time.monotonic()
    setup = lp.load_setup(
        args.config_name, args.checkpoint_dir, repo_id=args.repo_id, cudnn_attention=args.cudnn_attention
    )
    mc = setup.model_config
    horizons = tuple(int(k) for k in mc.tactile_future_horizons)
    ah, ad = mc.action_horizon, mc.action_dim
    tactile_keys = setup.tactile_keys
    store = lp.FutureTactileStore(args.labels_dir, horizons, target=args.target)
    if store.num_episodes != setup.dataset.num_episodes:
        logger.warning(
            "label store has %d episodes, dataset %d; only episodes present in both are sampled",
            store.num_episodes,
            setup.dataset.num_episodes,
        )
    logger.info("loaded model + dataset in %.0f s", time.monotonic() - t0)

    # ---- frames, split, targets ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    min_tail = max(*horizons, ah)
    refs = lp.sample_frames(
        setup.dataset,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        rng=rng,
        min_tail=min_tail,
        store=store,
        contact_quantile=args.contact_quantile,
        contact_proxy=args.contact_proxy,
    )
    is_val = lp.split_episodes(refs.episode, args.val_fraction, rng)
    n = len(refs)
    logger.info(
        "%d frames from %d episodes; %d train / %d val rows (%d val episodes)",
        n,
        len(np.unique(refs.episode)),
        int((~is_val).sum()),
        int(is_val.sum()),
        len(np.unique(refs.episode[is_val])),
    )
    num_pads, zdim = store.num_pads, store.target_dim()
    target_shape = (len(horizons), num_pads, zdim)
    targets = np.zeros((n, *target_shape), dtype=np.float32)
    for i, (ep, fr) in enumerate(zip(refs.episode, refs.frame, strict=True)):
        z, mask = store.targets(int(ep), int(fr))
        if not mask.all():
            raise RuntimeError(f"frame ({ep}, {fr}) has an invalid future horizon although min_tail={min_tail}")
        targets[i] = z
    np.save(out_dir / "targets.npy", targets)
    (out_dir / "frames.json").write_text(json.dumps({**refs.to_dict(), "is_val": is_val.tolist()}, indent=1))

    # ---- forward passes --------------------------------------------------------
    forward = lp.LayerForward(setup.model)
    layers = forward.layer_names
    noise = lp.fixed_noise(args.seed, n, ah, ad)
    features = None
    vlm = None
    flow_loss = np.full((len(conditions), len(taus), n), np.nan, dtype=np.float32)
    self_checks: dict[str, object] = {}
    t_fwd = time.monotonic()
    for b, batch in enumerate(lp.iter_batches(setup.dataset, refs, num_workers=args.num_workers)):
        rows = slice(int(batch.index[0]), int(batch.index[-1]) + 1)
        if not np.array_equal(batch.index, np.arange(rows.start, rows.stop)):
            raise RuntimeError("batches must arrive in order")
        obs = batch.observation
        noise_b = noise[rows]
        variants = {c: lp.make_variant(obs, c, tactile_keys) for c in conditions}
        if b == 0:
            first = forward(obs, batch.actions, noise_b, taus[0])
            again = forward(obs, batch.actions, noise_b, taus[0])
            rep = float(jnp.max(jnp.abs(first["action_stream"] - again["action_stream"])))
            self_checks["repeat_max_abs_diff"] = rep
            if rep != 0.0:
                logger.warning("real forward repeated is not bit-identical (max |diff| = %.3e)", rep)
            if "tac-shuffle" in variants:
                lp.assert_variant_differs(obs, variants["tac-shuffle"], tactile_keys)
                self_checks["tac_shuffle_differs"] = True
            self_checks["prefix_tokens_pooled_dim"] = int(first["prefix_pooled"].shape[-1])
        for ti, tau in enumerate(taus):
            for ci, cond in enumerate(conditions):
                out = forward(variants[cond], batch.actions, noise_b, tau)
                feats = np.asarray(pool_features(out["action_stream"], args.features, horizons, ah))
                if features is None:
                    features = np.lib.format.open_memmap(
                        out_dir / "features.npy",
                        mode="w+",
                        dtype=np.float16,
                        shape=(len(conditions), len(taus), len(layers), n, feats.shape[-1]),
                    )
                    vlm = np.zeros((n, int(out["prefix_pooled"].shape[-1])), dtype=np.float32)
                features[ci, ti, :, rows] = feats.astype(np.float16)
                flow_loss[ci, ti, rows] = np.asarray(jnp.mean(jnp.square(out["v_t"] - out["u_t"]), axis=(1, 2)))
                if ci == 0 and ti == 0:
                    vlm[rows] = np.asarray(out["prefix_pooled"])
        if b % 10 == 0:
            done = rows.stop
            logger.info("[%d/%d frames] %.1f frames/s", done, n, done / max(time.monotonic() - t_fwd, 1e-9))
    assert features is not None
    assert vlm is not None
    features.flush()
    logger.info("forward passes done in %.0f s", time.monotonic() - t_fwd)

    # ---- ridge fits ------------------------------------------------------------
    t_fit = time.monotonic()
    y = targets.reshape(n, -1)
    tr, va = ~is_val, is_val
    contact = refs.contact
    subsets = {}
    if contact is not None:
        subsets = {"contact": va & contact, "noncontact": va & ~contact}

    def fit(x: np.ndarray) -> dict:
        model, scores = lp.fit_ridge(x[tr], y[tr], x[va], y[va], alphas)
        entry = {**lp.evaluate_ridge(model, x[va], y[va], target_shape), "alpha": model.alpha, "alpha_nmse": scores}
        for name, rows_ in subsets.items():
            entry[name] = lp.evaluate_ridge(model, x[rows_], y[rows_], target_shape) if rows_.any() else None
        return entry

    vlm_entry = fit(vlm)
    results: dict[str, dict[str, dict[str, dict]]] = {}
    for ci, cond in enumerate(conditions):
        results[cond] = {}
        for ti, tau in enumerate(taus):
            per_layer = {"vlm": vlm_entry}
            for li, layer in enumerate(layers):
                per_layer[layer] = fit(np.asarray(features[ci, ti, li], dtype=np.float32))
            results[cond][str(tau)] = per_layer
            logger.info(
                "fit cond=%s tau=%.2f: best expert layer %s",
                cond,
                tau,
                min(layers[1:], key=lambda name: per_layer[name]["nmse"]),
            )
    logger.info("ridge fits done in %.0f s", time.monotonic() - t_fit)

    # ---- layer selection -------------------------------------------------------
    expert_layers = layers[1:]
    selection = {}
    for tau in taus:
        key = str(tau)
        real = results.get("real", {}).get(key)
        if real is None:
            continue
        best = min(expert_layers, key=lambda name: real[name]["nmse"])
        entry = {"best_layer": best, "best_nmse": real[best]["nmse"], "vlm_nmse": real["vlm"]["nmse"]}
        if "null" in results:
            gain = {name: results["null"][key][name]["nmse"] - real[name]["nmse"] for name in expert_layers}
            entry["tactile_gain_null_minus_real"] = gain
            entry["best_layer_by_tactile_gain"] = max(expert_layers, key=lambda name: gain[name])
        if subsets:
            entry["best_layer_contact"] = min(
                expert_layers, key=lambda name: (real[name]["contact"] or {"nmse": np.inf})["nmse"]
            )
        selection[key] = entry

    meta = {
        "config_name": args.config_name,
        "checkpoint_dir": str(setup.checkpoint_dir),
        "repo_id": setup.dataset.repo_id,
        "labels_dir": str(store.labels_dir),
        "target": args.target,
        "target_shape": list(target_shape),
        "horizons": list(horizons),
        "taus": list(taus),
        "conditions": list(conditions),
        "features": args.features,
        "feature_dim": int(features.shape[-1]),
        "layers": ["vlm", *layers],
        "num_frames": n,
        "num_train": int(tr.sum()),
        "num_val": int(va.sum()),
        "val_fraction": args.val_fraction,
        "alphas": list(alphas),
        "seed": args.seed,
        "contact_threshold": refs.contact_threshold,
        "contact_proxy": refs.contact_proxy,
        "contact_quantile": args.contact_quantile,
        "self_checks": self_checks,
        "flow_loss_mean": {
            cond: {str(tau): float(np.nanmean(flow_loss[ci, ti])) for ti, tau in enumerate(taus)}
            for ci, cond in enumerate(conditions)
        },
        "jax_devices": [str(d) for d in jax.devices()],
        "elapsed_s": time.monotonic() - t0,
    }
    payload = {"meta": meta, "results": results, "selection": selection}
    (out_dir / "results.json").write_text(json.dumps(lp.to_jsonable(payload), indent=1))
    report = render_markdown(payload)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"\nwrote {out_dir}")


def render_markdown(payload: dict) -> str:
    from test.tactile_counterfactual import layer_probe as lp

    meta, results, selection = payload["meta"], payload["results"], payload["selection"]
    taus = [str(t) for t in meta["taus"]]
    layers = meta["layers"]
    lines = ["# Layer-wise linear probe: future tactile field", ""]
    lines.append(f"- config `{meta['config_name']}`, checkpoint `{meta['checkpoint_dir']}`")
    lines.append(
        f"- target `{meta['target']}` {tuple(meta['target_shape'])}, horizons {meta['horizons']}, "
        f"features `{meta['features']}` ({meta['feature_dim']} dims)"
    )
    lines.append(
        f"- {meta['num_frames']} frames ({meta['num_train']} train / {meta['num_val']} val rows, split by episode); "
        f"contact proxy `{meta['contact_proxy']}` threshold {meta['contact_threshold']}"
    )
    lines.append(f"- self checks: {meta['self_checks']}")
    lines.append("")
    lines.append(
        "nMSE = validation MSE / MSE of the train-mean predictor (1 - R^2). Zero prediction of the "
        "normalised field scores ~1; lower is better."
    )
    lines.append("")
    for cond, per_tau in results.items():
        lines.append(f"## condition `{cond}`: nMSE per layer (rows) and tau (columns)")
        lines.append("")
        rows = [[layer] + [per_tau[t][layer]["nmse"] for t in taus] for layer in layers]
        lines.append(lp.markdown_table(["layer", *[f"tau={t}" for t in taus]], rows))
        lines.append("")
    if "real" in results and "null" in results:
        lines.append("## tactile contribution: nMSE(null) - nMSE(real) per layer (positive = tactile helped)")
        lines.append("")
        rows = [
            [layer] + [results["null"][t][layer]["nmse"] - results["real"][t][layer]["nmse"] for t in taus]
            for layer in layers
        ]
        lines.append(lp.markdown_table(["layer", *[f"tau={t}" for t in taus]], rows))
        lines.append("")
    if "real" in results and "tac-shuffle" in results:
        lines.append("## shuffle control: nMSE(tac-shuffle) - nMSE(real) per layer")
        lines.append("")
        rows = [
            [layer] + [results["tac-shuffle"][t][layer]["nmse"] - results["real"][t][layer]["nmse"] for t in taus]
            for layer in layers
        ]
        lines.append(lp.markdown_table(["layer", *[f"tau={t}" for t in taus]], rows))
        lines.append("")
    if selection:
        lines.append("## layer choice (condition `real`, expert layers only)")
        lines.append("")
        rows = []
        for t in taus:
            sel = selection.get(t)
            if sel is None:
                continue
            rows.append(
                [
                    t,
                    sel["best_layer"],
                    sel["best_nmse"],
                    sel.get("best_layer_by_tactile_gain", "n/a"),
                    sel.get("best_layer_contact", "n/a"),
                    sel["vlm_nmse"],
                ]
            )
        lines.append(
            lp.markdown_table(
                ["tau", "best layer", "nMSE", "best by null-real gain", "best on contact subset", "vlm nMSE"], rows
            )
        )
        lines.append("")
        for t in taus:
            sel = selection.get(t)
            if sel is None:
                continue
            best = results["real"][t][sel["best_layer"]]
            lines.append(f"### tau={t}, layer `{sel['best_layer']}`: breakdown")
            lines.append("")
            hz = meta["horizons"]
            lines.append(lp.markdown_table(["horizon k", *hz], [["nMSE", *best["per_horizon"]]]))
            lines.append("")
            lines.append(lp.markdown_table(["pad", *range(len(best["per_pad"]))], [["nMSE", *best["per_pad"]]]))
            if best.get("contact") is not None:
                lines.append("")
                lines.append(
                    lp.markdown_table(
                        ["subset", "nMSE", "n"],
                        [
                            ["contact", best["contact"]["nmse"], best["contact"]["n"]],
                            ["non-contact", best["noncontact"]["nmse"], best["noncontact"]["n"]],
                        ],
                    )
                )
            lines.append("")
    lines.append("## flow loss on the sampled frames (mean over rows), by condition and tau")
    lines.append("")
    rows = [[cond] + [vals[t] for t in taus] for cond, vals in meta["flow_loss_mean"].items()]
    lines.append(lp.markdown_table(["condition", *[f"tau={t}" for t in taus]], rows))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
