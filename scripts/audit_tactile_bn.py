"""Step 0: is the tactile encoder collapse caused by frozen ImageNet BN stats?

``tactile_encoders/fastvit.py`` hardcodes ``use_running_average=True`` at every
BatchNorm, and an earlier audit found all 160 BN mean/var buffers still
bit-identical to the ImageNet pretrained values. Gel images are low-contrast and
near-constant across frames, so normalising them by ImageNet statistics is a
candidate mechanism for the ~130x compression of the tactile manifold measured
in experiment B.

This script re-estimates the BN running statistics on real tactile frames
(a plain forward pass with batch statistics, which is what training would have
done with ``use_running_average=False``) and re-runs the separability metrics
before and after. No retraining, no gradient steps -- only the 160 buffers move.

Decision rule
-------------
separability recovers  -> the collapse is a normalisation bug; any retrain must
                          fix BN (re-estimate, or swap to GroupNorm) or it will
                          reproduce the collapse.
separability unchanged -> the collapse is representational; only better input
                          (differential frames) plus an auxiliary loss can fix it.

Usage:
    PYTHONPATH=.:src python scripts/audit_tactile_bn.py \
        --probe-config configs/probes/water_weight_counterfactual_inline_ep0based.yaml \
        --checkpoint-dir /path/to/checkpoints/<config>/59999
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import flax.linen as nn
import flax.nnx as nnx
import flax.traverse_util as tu
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.models.tactile_encoders import build_tactile_encoder
from openpi.models.tactile_encoders import fastvit as fastvit_mod
from test.tactile_counterfactual import probe_config as _probe_config
from test.tactile_counterfactual import runner as _runner

# ---------------------------------------------------------------------------
# checkpoint loading (partial restore: the tactile subtree is ~30 MB of 12 GB)
# ---------------------------------------------------------------------------


def _build(node, path: str = "", *, keep=None, as_args: bool = False):
    if isinstance(node, dict):
        return {k: _build(v, f"{path}/{k}" if path else str(k), keep=keep, as_args=as_args) for k, v in node.items()}
    if isinstance(node, list | tuple):
        return [_build(v, f"{path}/{i}" if path else str(i), keep=keep, as_args=as_args) for i, v in enumerate(node)]
    if as_args:
        return ocp.ArrayRestoreArgs(restore_type=np.ndarray)
    return node if keep(path) else ocp.PLACEHOLDER


def restore_tactile(checkpoint_dir: pathlib.Path) -> dict:
    path = (checkpoint_dir / "params").resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = dict(ckptr.metadata(path))
        restored = ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=_build(metadata, keep=lambda p: "tactile" in p),
                restore_args=_build(metadata, as_args=True),
            ),
        )
    flat = tu.flatten_dict(restored["params"])
    if all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    return tu.unflatten_dict(flat)


# ---------------------------------------------------------------------------
# nnx state <-> linen variables
#
# The encoder is a linen module behind nnx_bridge.ToNNX, so its nnx state is
# "module/<linen path>/<name>". Splitting mean/var off gives back exactly the
# two linen collections, which lets us call module.apply(..., mutable=...) and
# drive BatchNorm in batch-statistics mode without touching the source.
# ---------------------------------------------------------------------------

_BN_LEAVES = ("mean", "var")


def to_linen_variables(encoder) -> dict:
    pure = nnx.state(encoder).to_pure_dict()["module"]
    flat = tu.flatten_dict(pure)
    params = {k: v for k, v in flat.items() if k[-1] not in _BN_LEAVES}
    stats = {k: v for k, v in flat.items() if k[-1] in _BN_LEAVES}
    return {"params": tu.unflatten_dict(params), "batch_stats": tu.unflatten_dict(stats)}


class _TrainModeBN:
    """Proxy for the ``nn`` module global that forces BN into batch-stat mode.

    ``nn.BatchNorm(...)`` is looked up on the module global at call time, so
    swapping the global after import reaches every one of the 80 BN sites
    without editing the model source. Everything else proxies to flax.linen.
    """

    def __init__(self, momentum: float):
        self._momentum = momentum

    def __getattr__(self, name):
        return getattr(nn, name)

    def BatchNorm(self, **kwargs):  # noqa: N802 - mirrors flax.linen's name
        kwargs["use_running_average"] = False
        kwargs["momentum"] = self._momentum
        return nn.BatchNorm(**kwargs)


def forward_eval(module, variables, images: np.ndarray, batch: int) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        out.append(np.asarray(module.apply(variables, jnp.asarray(images[i : i + batch])), dtype=np.float32))
    return np.concatenate(out)


def reestimate_bn(module, variables, images: np.ndarray, *, batch: int, passes: int, momentum: float) -> dict:
    """Return variables with BN running stats re-estimated on `images`."""
    original_nn = fastvit_mod.nn
    fastvit_mod.nn = _TrainModeBN(momentum)
    try:
        stats = variables["batch_stats"]
        n_batches = 0
        for _ in range(passes):
            order = np.arange(len(images))
            for i in range(0, len(images) - batch + 1, batch):  # drop last partial batch
                _, updates = module.apply(
                    {"params": variables["params"], "batch_stats": stats},
                    jnp.asarray(images[order[i : i + batch]]),
                    mutable=["batch_stats"],
                )
                stats = updates["batch_stats"]
                n_batches += 1
        residual = momentum**n_batches
    finally:
        fastvit_mod.nn = original_nn
    print(f"  BN re-estimated over {n_batches} batches of {batch}; ImageNet residual weight = {residual:.2e}")
    return {"params": variables["params"], "batch_stats": jax.tree.map(np.asarray, stats)}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def tokens_of(module, variables, proj_kernel, proj_bias, images, batch) -> np.ndarray:
    feats = forward_eval(module, variables, images, batch)
    return feats @ proj_kernel + proj_bias


def separability(tok_a: np.ndarray, tok_b: np.ndarray) -> dict:
    """Doc experiment-B style: between-class distance vs within-class scatter."""

    def scatter(x):
        return float(np.linalg.norm(x - x.mean(0), axis=1).mean())

    ma, mb = tok_a.mean(0), tok_b.mean(0)
    between = float(np.linalg.norm(ma - mb))
    within = max(scatter(tok_a), scatter(tok_b))
    cos = float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb)))
    return {
        "between": between,
        "within": within,
        "ratio": between / within if within else float("inf"),
        "one_minus_cos": 1.0 - cos,
        "rel": between / float(np.linalg.norm(ma)),
    }



def ridge_probe(x_tr, y_tr, x_te, y_te, ratios=(1.0, 0.1, 0.01)) -> list[float]:
    """Linear readout accuracy, dual form (n << d), standardised on train stats.

    A class-mean distance is too crude a proxy for what the action expert can
    do: its readout of a tactile token is linear (W_V), so the decision-relevant
    question is whether a linear map separates the classes -- which it can even
    when the class means sit inside the within-class scatter.
    """
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    a = (x_tr - mu) / sd
    b = (x_te - mu) / sd
    k = a @ a.T
    kt = b @ a.T
    scale = float(np.trace(k)) / len(a)
    out = []
    for r in ratios:
        alpha = np.linalg.solve(k + r * scale * np.eye(len(a)), y_tr)
        out.append(float(np.mean(np.sign(kt @ alpha) == y_te)))
    return out


def whitened_loo_cos(deltas: np.ndarray, lam: float = 0.1) -> tuple[float, float]:
    """Leave-one-out alignment of the per-pair difference vectors (see _transmit.py)."""
    u = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
    g = u @ u.T
    out = []
    for i in range(len(u)):
        idx = np.delete(np.arange(len(u)), i)
        gi = g[np.ix_(idx, idx)]
        a = np.linalg.solve(gi + lam * np.eye(len(idx)), np.ones(len(idx)))
        out.append(float(g[i, idx] @ a / np.sqrt(a @ gi @ a)))
    out = np.array(out)
    return float(out.mean()), 1.0 / np.sqrt(deltas.shape[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-config", type=pathlib.Path, required=True)
    p.add_argument("--checkpoint-dir", type=pathlib.Path, default=None)
    p.add_argument("--calib-frames", type=int, default=384)
    p.add_argument("--eval-episodes", type=int, nargs="+", default=[1, 21])
    p.add_argument("--frames-per-class", type=int, default=12)
    p.add_argument("--pairs", type=int, default=60, help="heavy/light pairs for the direction test")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    probe = _probe_config.load_probe_config(args.probe_config)
    if args.checkpoint_dir is not None:
        probe = dataclasses.replace(
            probe, model=dataclasses.replace(probe.model, checkpoint_dir=str(args.checkpoint_dir))
        )
    ckpt = pathlib.Path(probe.model.checkpoint_dir).expanduser().resolve()
    print(f"checkpoint : {ckpt}")

    train_config = _runner.build_train_config(probe)
    data_config = _runner.resolve_data_config(train_config, probe, ckpt)
    dataset = _runner.build_probe_dataset(probe, train_config, data_config)

    tactile_keys = sorted(k for k in dataset.observation_from_sample(
        dataset.get_sample(probe.pairs[0].full.episode_index, probe.pairs[0].full.frame_index)
    ).images if "tactile" in k)
    print(f"tactile keys: {tactile_keys}")

    # ---- build encoder + proj, load trained weights -----------------------
    rngs = nnx.Rngs(0)
    encoder = build_tactile_encoder("fastvit_t12", rngs=rngs, pretrained_path=None, compute_dtype=jnp.float32)
    width = train_config.model.action_expert_width if hasattr(train_config.model, "action_expert_width") else 1024
    proj = nnx.Linear(encoder.feature_dim, width, rngs=rngs)
    params = restore_tactile(ckpt)
    for module_, pure in ((encoder, params["tactile_encoder"]), (proj, params["tactile_proj"])):
        state = nnx.state(module_)
        state.replace_by_pure_dict(pure)
        nnx.update(module_, state)
    encoder.eval()
    proj_kernel = np.asarray(proj.kernel.value, dtype=np.float32)
    proj_bias = np.asarray(proj.bias.value, dtype=np.float32) if proj.bias is not None else 0.0

    linen = fastvit_mod.FastVitT12Module(dtype=jnp.float32)
    variables = to_linen_variables(encoder)
    n_bn = len(tu.flatten_dict(variables["batch_stats"]))
    n_par = len(tu.flatten_dict(variables["params"]))
    print(f"loaded: {n_par} param leaves + {n_bn} BN buffers, tactile_proj -> {width}")

    # Gate: the linen path must reproduce the nnx path bit-for-bit.
    probe_img = np.zeros((2, 224, 224, 3), dtype=np.float32)
    gate = float(np.abs(forward_eval(linen, variables, probe_img, 2)
                        - np.asarray(encoder(jnp.asarray(probe_img)))).max())
    print(f"gate: linen vs nnx forward max|diff| = {gate:.3e}")
    if gate > 1e-4:
        raise SystemExit("linen/nnx forward mismatch -- variable split is wrong")

    # ---- collect frames ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    eval_eps = set(args.eval_episodes)

    def tactile_of(ep, fr):
        obs = dataset.observation_from_sample(dataset.get_sample(ep, fr))
        return np.stack([np.asarray(obs.images[k])[0] for k in tactile_keys]), np.asarray(obs.state)[0]

    all_eps = sorted({pr.full.episode_index for pr in probe.pairs} | {pr.empty.episode_index for pr in probe.pairs})
    calib_eps = [e for e in all_eps if e not in eval_eps]
    print(f"\ncalibration: {args.calib_frames} frames from {len(calib_eps)} episodes (eval episodes held out)")
    calib = []
    while len(calib) * len(tactile_keys) < args.calib_frames * len(tactile_keys) and calib_eps:
        ep = int(rng.choice(calib_eps))
        n = dataset.episode_length(ep)
        fr = int(rng.integers(0, n))
        if not dataset.has_sample(ep, fr):
            continue
        imgs, _ = tactile_of(ep, fr)
        calib.append(imgs)
        if len(calib) >= args.calib_frames:
            break
    calib_imgs = np.concatenate(calib)
    print(f"  calibration images: {calib_imgs.shape}")

    # grasp open vs closed inside a single episode (doc experiment B2)
    grasp = {}
    for ep in args.eval_episodes:
        n = dataset.episode_length(ep)
        frames = [f for f in range(0, n, max(1, n // 120)) if dataset.has_sample(ep, f)]
        rows = [(f, *tactile_of(ep, f)) for f in frames]
        gpos = np.array([s[19] for _, _, s in rows])
        openi = [i for i, g in enumerate(gpos) if g > 0.9]
        closedi = [i for i, g in enumerate(gpos) if g < 0.5]
        k = args.frames_per_class
        if len(openi) < k or len(closedi) < k:
            print(f"  ep{ep}: only {len(openi)} open / {len(closedi)} closed frames -- skipped")
            continue
        pick = lambda idx: np.stack([rows[i][1] for i in rng.choice(idx, k, replace=False)])
        grasp[ep] = (pick(openi), pick(closedi))
        print(f"  ep{ep}: {len(openi)} open / {len(closedi)} closed frames available")

    # heavy/light pairs (the task-relevant label)
    pairs = probe.pairs[: args.pairs]
    heavy = np.stack([tactile_of(pr.full.episode_index, pr.full.frame_index)[0] for pr in pairs])
    light = np.stack([tactile_of(pr.empty.episode_index, pr.empty.frame_index)[0] for pr in pairs])
    print(f"  heavy/light pairs: {len(pairs)}")

    noise = rng.uniform(-1.0, 1.0, size=(len(tactile_keys), 224, 224, 3)).astype(np.float32)

    # ---- report -----------------------------------------------------------
    def report(tag: str, var: dict) -> None:
        tok = lambda x: tokens_of(linen, var, proj_kernel, proj_bias, x, args.batch)
        print("\n" + "=" * 92)
        print(f"{tag}")
        print("=" * 92)

        print("  [B2] grasp open vs closed, within episode (token level)")
        print(f"    {'ep':>4s} {'cam':>4s} {'between':>10s} {'within':>10s} {'ratio':>8s} {'1-cos':>12s}")
        for ep, (op, cl) in grasp.items():
            for ci, key in enumerate(tactile_keys):
                m = separability(tok(op[:, ci]), tok(cl[:, ci]))
                print(f"    {ep:4d} {ci:4d} {m['between']:10.4f} {m['within']:10.4f} "
                      f"{m['ratio']:8.3f}x {m['one_minus_cos']:12.3e}")

        print("\n  [dynamic range] real tactile vs uniform-noise image")
        t_real = tok(heavy[0])
        t_noise = tok(noise)
        for ci in range(len(tactile_keys)):
            c = float(t_real[ci] @ t_noise[ci] / (np.linalg.norm(t_real[ci]) * np.linalg.norm(t_noise[ci])))
            print(f"    cam{ci}: 1-cos = {1 - c:.4e}")

        if len(grasp) >= 2:
            eps = list(grasp)
            print("\n  [B2b] linear readout of grasp state, TRAINED ON ONE EPISODE, TESTED ON THE OTHER")
            print(f"    {'train->test':>14s} {'cam':>10s} {'acc @ lam=1':>12s} {'0.1':>7s} {'0.01':>7s}")
            for tr_ep, te_ep in ((eps[0], eps[1]), (eps[1], eps[0])):
                for ci, key in enumerate(tactile_keys):
                    xtr = np.concatenate([tok(grasp[tr_ep][0][:, ci]), tok(grasp[tr_ep][1][:, ci])])
                    xte = np.concatenate([tok(grasp[te_ep][0][:, ci]), tok(grasp[te_ep][1][:, ci])])
                    y = np.concatenate([np.ones(args.frames_per_class), -np.ones(args.frames_per_class)])
                    acc = ridge_probe(xtr, y, xte, y)
                    print(f"    {f'{tr_ep}->{te_ep}':>14s} {key.replace('_rgb', ''):>10s} "
                          + "".join(f"{v:12.3f}" if i == 0 else f"{v:7.3f}" for i, v in enumerate(acc)))
            print("    (chance = 0.500)")

        print("\n  [heavy/light] token perturbation and reusable direction")
        print(f"    {'cam':>4s} {'||dt||/||t||':>13s} {'whitened LOO cos':>18s} {'chance':>9s} {'x chance':>9s}")
        for ci, key in enumerate(tactile_keys):
            th, tl = tok(heavy[:, ci]), tok(light[:, ci])
            d = th - tl
            rel = float(np.mean(np.linalg.norm(d, axis=1) / np.linalg.norm(th, axis=1)))
            cos, chance = whitened_loo_cos(d)
            print(f"    {key.replace('_rgb', ''):>4s} {rel * 100:12.3f}% {cos:+18.4f} "
                  f"{chance:9.4f} {cos / chance:8.2f}x")

        print("\n  [heavy/light] linear readout, EPISODES SPLIT INTO TWO BLOCKS")
        print(f"    {'cam':>10s} {'acc @ lam=1':>12s} {'0.1':>7s} {'0.01':>7s}")
        heavy_eps = np.array([pr.full.episode_index for pr in pairs])
        light_eps = np.array([pr.empty.episode_index for pr in pairs])
        cut = float(np.median(np.concatenate([heavy_eps, light_eps])))
        for ci, key in enumerate(tactile_keys):
            th, tl = tok(heavy[:, ci]), tok(light[:, ci])
            x = np.concatenate([th, tl])
            y = np.concatenate([np.ones(len(th)), -np.ones(len(tl))])
            blk = np.concatenate([heavy_eps, light_eps]) < cut
            if blk.sum() < 4 or (~blk).sum() < 4:
                print(f"    {key.replace('_rgb', ''):>10s}   (block split degenerate)")
                continue
            acc = ridge_probe(x[blk], y[blk], x[~blk], y[~blk])
            print(f"    {key.replace('_rgb', ''):>10s} "
                  + "".join(f"{v:12.3f}" if i == 0 else f"{v:7.3f}" for i, v in enumerate(acc)))
        print(f"    (chance = 0.500; train episodes < {cut:.0f}, test >= {cut:.0f})")

    report("BEFORE  --  BN running stats = ImageNet (as trained)", variables)

    print("\n" + "-" * 92)
    print("re-estimating BN running statistics on real tactile frames")
    new_vars = reestimate_bn(linen, variables, calib_imgs, batch=args.batch,
                             passes=args.passes, momentum=args.momentum)

    old_flat = tu.flatten_dict(variables["batch_stats"])
    new_flat = tu.flatten_dict(new_vars["batch_stats"])
    dm = [float(np.abs(new_flat[k] - old_flat[k]).mean() / (np.abs(old_flat[k]).mean() + 1e-12))
          for k in old_flat if k[-1] == "mean"]
    dv = [float(np.abs(new_flat[k] - old_flat[k]).mean() / (np.abs(old_flat[k]).mean() + 1e-12))
          for k in old_flat if k[-1] == "var"]
    print(f"  how far ImageNet stats were from the tactile data:")
    print(f"    mean: median rel. shift {np.median(dm):.3f}, p90 {np.quantile(dm, 0.9):.3f}, max {np.max(dm):.3f}")
    print(f"    var : median rel. shift {np.median(dv):.3f}, p90 {np.quantile(dv, 0.9):.3f}, max {np.max(dv):.3f}")

    report("AFTER   --  BN running stats re-estimated on tactile frames", new_vars)


if __name__ == "__main__":
    main()
