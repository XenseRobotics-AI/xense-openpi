#!/usr/bin/env python3
"""Layer-wise causal sensitivity probe: where does tactile enter the action stream, and how much?

Section 6.1 of docs/action-conditioned-tactile-pretraining.md, the light-weight cousin of
``scripts/tactile_counterfactual_probe.py``. One training-style forward per variant and
per flow time ``tau`` on batches of real frames drawn across episodes; nothing is trained.

Variants (same batch, same noise, same tau):

    real          reference
    null          tactile ``image_mask=False``
    tac-shuffle   tactile images + masks rolled by one inside the batch   -> S_tac
    vl-swap       RGB views + prompt rolled by one (pi05: the state rides in the prompt) -> S_vl
    pad-pert      the last N valid prompt tokens masked (red herring; expect ~0)
  --extra adds the counterfactual conditions of section 6.2:
    tac-zero      tactile images zeroed, masks kept
    pad-swap      left / right jaw swapped
    tac-timeshift tactile taken from the same episode ``--time-shift`` frames later

Measurement points: the tactile tokens (collapse check only), the action-position
residual stream ``l0`` (suffix input) .. ``l18`` (block outputs), the velocity ``v_t``,
and (``--chunk-steps`` > 0) the final action chunk of the production sampler.

Per point and tau: ``S_* = 1 - cos_cent(real, variant)`` (centred on the batch mean of
``real``), ``S_x`` = the same between neighbouring samples, ``R = S_tac / S_vl``,
``share = S_tac / S_x``; all of them on every row, on the contact subset and on the
non-contact subset (label store proxy). Self checks: the real forward repeated is
bit-identical; every tactile variant changes the tactile pixels.

Usage:

    python scripts/probe_tactile_sensitivity_layers.py \\
        --config-name pi05_base_bi_flexiv_bottle_sorting_0915_fastvit_h100 \\
        --checkpoint-dir checkpoints/pi05_base_bi_flexiv_bottle_sorting_0915_fastvit_h100/10000 \\
        --labels-dir assets/tactile_future_labels/bottle-sorting-0810 \\
        --num-frames 512 --batch-size 16 --extra

Run the same command on the no-tactile baseline checkpoint to get the reference curve.
Outputs ``results.json``, ``report.md`` and ``frames.json`` under ``--out`` (default
``outputs/tactile_layer_probes/sensitivity/<timestamp>``).
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

logger = logging.getLogger("probe_tactile_sensitivity_layers")

# Which variant supplies which headline sensitivity.
S_TAC, S_VL, S_NULL, S_PAD = "tac-shuffle", "vl-swap", "null", "pad-pert"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--labels-dir", default=None, help="label store; enables the contact / non-contact split")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--num-frames", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--taus", type=float, nargs="+", default=None, help="flow times (default 0.25 0.5 0.75 1.0)")
    p.add_argument("--extra", action="store_true", help="also run tac-zero, pad-swap and tac-timeshift")
    p.add_argument("--variants", nargs="+", default=None, help="explicit variant list (overrides --extra)")
    p.add_argument("--pad-extra", type=int, default=2, help="pad-pert: valid prompt tokens masked at the tail")
    p.add_argument("--time-shift", type=int, default=30, help="tac-timeshift donor offset in frames")
    p.add_argument("--chunk-steps", type=int, default=10, help="denoising steps for the final-chunk point; 0 skips it")
    p.add_argument("--contact-quantile", type=float, default=0.5)
    p.add_argument("--contact-proxy", default="delta", choices=("delta", "state"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--cudnn-attention", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


class Accumulator:
    """Per-(tau, point, variant) lists of per-sample sensitivities, plus the sample flags."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str], list[np.ndarray]] = {}
        self.contact: list[np.ndarray] = []

    def add(self, tau: str, point: str, variant: str, per_sample: np.ndarray) -> None:
        self.values.setdefault((tau, point, variant), []).append(np.asarray(per_sample, dtype=np.float64))

    def get(self, tau: str, point: str, variant: str) -> np.ndarray | None:
        chunks = self.values.get((tau, point, variant))
        return None if not chunks else np.concatenate(chunks)


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
    if args.variants:
        variants = tuple(args.variants)
    else:
        variants = lp.CORE_VARIANTS + (lp.EXTRA_VARIANTS if args.extra else ())
    unknown = [v for v in variants if v not in lp.ALL_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {lp.ALL_VARIANTS}")
    variants = tuple(v for v in variants if v != "real")
    tactile_variants = [v for v in variants if v in ("tac-shuffle", "tac-zero", "pad-swap", "tac-timeshift")]
    out_dir = pathlib.Path(
        args.out or _ROOT / "outputs" / "tactile_layer_probes" / "sensitivity" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out_dir)

    t0 = time.monotonic()
    setup = lp.load_setup(
        args.config_name, args.checkpoint_dir, repo_id=args.repo_id, cudnn_attention=args.cudnn_attention
    )
    mc = setup.model_config
    ah, ad = mc.action_horizon, mc.action_dim
    tactile_keys = setup.tactile_keys
    store = None
    if args.labels_dir:
        store = lp.FutureTactileStore(args.labels_dir, tuple(mc.tactile_future_horizons))
    logger.info("loaded model + dataset in %.0f s", time.monotonic() - t0)

    rng = np.random.default_rng(args.seed)
    use_timeshift = "tac-timeshift" in variants
    min_tail = max(ah, args.time_shift if use_timeshift else 0)
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
    n = len(refs)
    (out_dir / "frames.json").write_text(json.dumps(refs.to_dict(), indent=1))

    forward = lp.LayerForward(setup.model)
    layers = forward.layer_names
    points = ["tactile_tokens", *layers, "v_t"]
    chunk_sampler = None
    if args.chunk_steps > 0:
        chunk_sampler = lp.ChunkSampler(setup.model, rtc=bool(mc.enable_training_time_rtc))
        points.append("chunk")
    noise = lp.fixed_noise(args.seed, n, ah, ad)
    acc = Accumulator()
    self_checks: dict[str, object] = {}

    batches = lp.iter_batches(setup.dataset, refs, num_workers=args.num_workers)
    donors = None
    if use_timeshift:
        donors = lp.iter_batches(setup.dataset, refs, num_workers=args.num_workers, frame=refs.frame + args.time_shift)
    t_fwd = time.monotonic()
    for b, batch in enumerate(batches):
        rows = slice(int(batch.index[0]), int(batch.index[-1]) + 1)
        if not np.array_equal(batch.index, np.arange(rows.start, rows.stop)):
            raise RuntimeError("batches must arrive in order")
        obs = batch.observation
        noise_b = noise[rows]
        donor_obs = None
        if donors is not None:
            donor = next(donors)
            if not np.array_equal(donor.index, batch.index):
                raise RuntimeError("donor batch out of sync")
            donor_obs = donor.observation
        variant_obs = {
            v: lp.make_variant(obs, v, tactile_keys, pad_extra=args.pad_extra, donor=donor_obs) for v in variants
        }
        if b == 0:
            first = forward(obs, batch.actions, noise_b, taus[0])
            again = forward(obs, batch.actions, noise_b, taus[0])
            rep = float(jnp.max(jnp.abs(first["action_stream"] - again["action_stream"])))
            self_checks["repeat_max_abs_diff"] = rep
            if rep != 0.0:
                logger.warning("real forward repeated is not bit-identical (max |diff| = %.3e)", rep)
            for v in tactile_variants:
                lp.assert_variant_differs(obs, variant_obs[v], tactile_keys)
            self_checks["tactile_variants_differ"] = tactile_variants
        acc.contact.append(np.ones(rows.stop - rows.start, dtype=bool) if refs.contact is None else refs.contact[rows])

        for tau in taus:
            key = str(tau)
            real = forward(obs, batch.actions, noise_b, tau)
            real_np = {k: np.asarray(v) for k, v in real.items()}
            acc.add(key, "tactile_tokens", "x", lp.cross_sample_sensitivity(real_np["tactile_tokens"]))
            layered_x = lp.layered_cross_sample(real_np["action_stream"])
            for li, layer in enumerate(layers):
                acc.add(key, layer, "x", layered_x[li])
            acc.add(key, "v_t", "x", lp.cross_sample_sensitivity(real_np["v_t"]))
            for v in variants:
                out = forward(variant_obs[v], batch.actions, noise_b, tau)
                out_np = {k: np.asarray(val) for k, val in out.items()}
                acc.add(key, "tactile_tokens", v, lp.sensitivity(real_np["tactile_tokens"], out_np["tactile_tokens"]))
                layered = lp.layered_sensitivity(real_np["action_stream"], out_np["action_stream"])
                for li, layer in enumerate(layers):
                    acc.add(key, layer, v, layered[li])
                acc.add(key, "v_t", v, lp.sensitivity(real_np["v_t"], out_np["v_t"]))

        if chunk_sampler is not None:
            real_chunk = np.asarray(chunk_sampler(obs, noise_b, args.chunk_steps))
            acc.add("chunk", "chunk", "x", lp.cross_sample_sensitivity(real_chunk))
            for v in variants:
                chunk = np.asarray(chunk_sampler(variant_obs[v], noise_b, args.chunk_steps))
                acc.add("chunk", "chunk", v, lp.sensitivity(real_chunk, chunk))
        if b % 5 == 0:
            logger.info("[%d/%d frames] %.1f frames/s", rows.stop, n, rows.stop / max(time.monotonic() - t_fwd, 1e-9))
    logger.info("forward passes done in %.0f s", time.monotonic() - t_fwd)

    # ---- aggregate -------------------------------------------------------------
    contact = np.concatenate(acc.contact)
    subsets = {"all": np.ones_like(contact)}
    if store is not None:
        subsets["contact"] = contact
        subsets["noncontact"] = ~contact

    def block(tau_key: str, point: str) -> dict:
        entry: dict = {}
        columns = ["x", *variants]
        for name in columns:
            vals = acc.get(tau_key, point, name)
            entry[name] = None if vals is None else {sub: lp.summarize(vals, mask) for sub, mask in subsets.items()}
        for sub in subsets:
            s_tac = _mean(entry, S_TAC, sub)
            s_vl = _mean(entry, S_VL, sub)
            s_x = _mean(entry, "x", sub)
            s_pad = _mean(entry, S_PAD, sub)
            entry.setdefault("ratios", {})[sub] = {
                "R": _ratio(s_tac, s_vl),
                "share": _ratio(s_tac, s_x),
                "pad_over_vl": _ratio(s_pad, s_vl),
            }
        return entry

    results: dict[str, dict[str, dict]] = {}
    for tau in taus:
        key = str(tau)
        results[key] = {point: block(key, point) for point in points if point != "chunk"}
    if chunk_sampler is not None:
        results["chunk"] = {"chunk": block("chunk", "chunk")}

    verdict = build_verdict(results, taus, layers, variants, has_subsets=store is not None)

    meta = {
        "config_name": args.config_name,
        "checkpoint_dir": str(setup.checkpoint_dir),
        "repo_id": setup.dataset.repo_id,
        "labels_dir": args.labels_dir,
        "num_frames": n,
        "num_episodes": len(np.unique(refs.episode)),
        "batch_size": args.batch_size,
        "taus": list(taus),
        "variants": list(variants),
        "pad_extra": args.pad_extra,
        "time_shift": args.time_shift if use_timeshift else None,
        "chunk_steps": args.chunk_steps,
        "chunk_sampler": None if chunk_sampler is None else chunk_sampler.mode,
        "points": points,
        "contact_threshold": refs.contact_threshold,
        "contact_proxy": refs.contact_proxy,
        "contact_quantile": args.contact_quantile,
        "num_contact": int(contact.sum()) if store is not None else None,
        "seed": args.seed,
        "self_checks": self_checks,
        "jax_devices": [str(d) for d in jax.devices()],
        "elapsed_s": time.monotonic() - t0,
    }
    payload = {"meta": meta, "results": results, "verdict": verdict}
    (out_dir / "results.json").write_text(json.dumps(lp.to_jsonable(payload), indent=1))
    report = render_markdown(payload)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"\nwrote {out_dir}")


def _mean(entry: dict, name: str, sub: str) -> float | None:
    stats = entry.get(name)
    if not stats or stats.get(sub) is None:
        return None
    return float(stats[sub]["mean"])


# Below this the denominator is float32 round-off (e.g. S_vl at l0, which cannot see the
# prefix at all), and a ratio would be 0/0 dressed up as a number.
_RATIO_FLOOR = 1e-7


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b < _RATIO_FLOOR:
        return None
    return a / b


def build_verdict(results: dict, taus, layers, variants, *, has_subsets: bool) -> dict:
    """The three checks of section 6.1, as numbers plus a one-line reading each."""
    out: dict = {}
    first = str(taus[0])
    # 1. tactile tokens must vary across samples; pad-pert must be a non-event.
    tok_x = _mean(results[first]["tactile_tokens"], "x", "all")
    out["tactile_token_cross_sample_S_x"] = tok_x
    out["tactile_tokens_collapsed"] = None if tok_x is None else bool(tok_x < 1e-3)
    pad = {t: results[str(t)]["v_t"]["ratios"]["all"]["pad_over_vl"] for t in taus}
    out["pad_pert_over_vl_at_v_t"] = pad
    out["pad_pert_is_negligible"] = (
        None if all(v is None for v in pad.values()) else all(v is not None and v < 0.1 for v in pad.values())
    )
    # 2. share at v_t, contact vs non-contact.
    share = {}
    for t in taus:
        r = results[str(t)]["v_t"]["ratios"]
        share[str(t)] = {sub: r[sub]["share"] for sub in r}
    out["share_at_v_t"] = share
    if has_subsets:
        out["share_contact_above_noncontact"] = {
            str(t): (
                share[str(t)]["contact"] is not None
                and share[str(t)]["noncontact"] is not None
                and share[str(t)]["contact"] > share[str(t)]["noncontact"]
            )
            for t in taus
        }
    # 3. S_tac curve over layers: shallowest layer reaching half of the curve's maximum.
    curves = {}
    for t in taus:
        curve = [_mean(results[str(t)][layer], S_TAC, "all") for layer in layers]
        vals = np.array([np.nan if c is None else c for c in curve])
        rise = None
        if np.isfinite(vals).any() and np.nanmax(vals) > 0:
            half = 0.5 * np.nanmax(vals)
            rise = layers[int(np.argmax(vals >= half))]
        curves[str(t)] = {"S_tac_by_layer": curve, "rise_layer": rise, "peak_layer": layers[int(np.nanargmax(vals))]}
    out["S_tac_curves"] = curves
    return out


def render_markdown(payload: dict) -> str:
    from test.tactile_counterfactual import layer_probe as lp

    meta, results, verdict = payload["meta"], payload["results"], payload["verdict"]
    variants = meta["variants"]
    lines = ["# Layer-wise tactile sensitivity probe", ""]
    lines.append(f"- config `{meta['config_name']}`, checkpoint `{meta['checkpoint_dir']}`")
    lines.append(
        f"- {meta['num_frames']} frames from {meta['num_episodes']} episodes, batch {meta['batch_size']}, "
        f"variants {variants}, pad_extra {meta['pad_extra']}, time_shift {meta['time_shift']}"
    )
    if meta["contact_proxy"]:
        lines.append(
            f"- contact proxy `{meta['contact_proxy']}` threshold {meta['contact_threshold']:.3f}: "
            f"{meta['num_contact']} contact rows"
        )
    lines.append(f"- self checks: {meta['self_checks']}")
    lines.append("")
    lines.append(
        "S_* = 1 - centred cosine between the real forward and the variant, per sample, averaged. "
        "S_x = the same between neighbouring samples (the natural variation). R = S_tac / S_vl, "
        "share = S_tac / S_x."
    )
    lines.append("")
    lines.append("## verdict")
    lines.append("")
    lines.append(
        f"- tactile tokens: cross-sample S_x = {verdict['tactile_token_cross_sample_S_x']}; "
        f"collapsed = {verdict['tactile_tokens_collapsed']}"
    )
    lines.append(
        f"- pad-pert / vl at v_t: { {k: _fmt(v) for k, v in verdict['pad_pert_over_vl_at_v_t'].items()} }; "
        f"negligible = {verdict['pad_pert_is_negligible']}"
    )
    for t, s in verdict["share_at_v_t"].items():
        lines.append(f"- tau={t}: share at v_t " + ", ".join(f"{sub}={_fmt(v)}" for sub, v in s.items()))
    if "share_contact_above_noncontact" in verdict:
        lines.append(f"- share(contact) > share(non-contact) at v_t: {verdict['share_contact_above_noncontact']}")
    for t, c in verdict["S_tac_curves"].items():
        lines.append(f"- tau={t}: S_tac rises at `{c['rise_layer']}` (half of peak), peaks at `{c['peak_layer']}`")
    lines.append("")

    extra_cols = [v for v in variants if v not in (S_TAC, S_VL, S_NULL, S_PAD)]
    header = ["point", "S_tac", "S_vl", "S_null", "S_pad", *[f"S[{v}]" for v in extra_cols], "S_x", "R", "share"]
    subs = ["all"]
    if meta["contact_proxy"]:
        header += ["share(contact)", "share(non-contact)"]
        subs = ["all", "contact", "noncontact"]
    for tau in meta["taus"]:
        key = str(tau)
        lines.append(f"## tau = {tau}")
        lines.append("")
        rows = [_row(point, entry, extra_cols, subs) for point, entry in results[key].items()]
        lines.append(lp.markdown_table(header, rows))
        lines.append("")
    if "chunk" in results:
        lines.append(f"## final action chunk ({meta['chunk_steps']} steps, {meta['chunk_sampler']} sampler)")
        lines.append("")
        lines.append(lp.markdown_table(header, [_row("chunk", results["chunk"]["chunk"], extra_cols, subs)]))
        lines.append("")
    return "\n".join(lines)


def _row(point: str, entry: dict, extra_cols: list[str], subs: list[str]) -> list:
    """One report row: the headline sensitivities, S_x, R and the share per subset."""
    row = [point, *(_mean(entry, name, "all") for name in (S_TAC, S_VL, S_NULL, S_PAD, *extra_cols))]
    row += [_mean(entry, "x", "all"), entry["ratios"]["all"]["R"]]
    row += [entry["ratios"][sub]["share"] for sub in subs]
    return row


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
