"""Compare attention implementations against an fp32 attention reference at a real Adam state.

The 2026-08-31 report showed that the full-18-layer cuDNN path diverges after
~step 1000 although its raw gradient is only ~1.5% away from the explicit bf16
path. This probe restores the real step-1000 train state (params + Adam moments),
computes the gradient and the actual AdamW update for the same real batch/rng under

  explicit_fp32ref  explicit attention with fp32 q/k/v/probs   (numerical reference)
  explicit_bf16     production explicit path                    (known stable)
  cudnn_bf16        historical cuDNN path                       (known to diverge)
  cudnn_fp16        cuDNN kernel in float16 + dynamic loss scaling (candidate)

and reports, for every variant, raw-gradient and Adam-update distance to the
fp32 reference and to the explicit bf16 path. A candidate is promising when its
update-space fingerprint (relative delta / sign-flip fraction) is at or below the
explicit bf16 path's own distance from the fp32 reference.
"""

import dataclasses
import functools
import logging
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

sys.path.insert(0, "scripts")
import train as train_mod

CONFIG = "diag_cudnn_strict_order"
EXP_NAME = "diag_cudnn_strict_order"
CHECKPOINT_STEP = 1000
BATCH = 32
WORKERS = 16
NUM_BATCHES = 2


def _loss_grads_update(model_def, trainable_filter, tx, opt_state, params, rng, obs, act):
    model = nnx.merge(model_def, params)
    model.train()

    def loss_fn(m, r, o, a):
        return jnp.mean(m.compute_loss(r, o, a, train=True))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter))(model, rng, obs, act)
    trainable_params = params.filter(trainable_filter)
    updates, _ = tx.update(grads, opt_state, trainable_params)
    return loss, grads, updates


def _flat(tree):
    pure = tree.to_pure_dict()
    out = []

    def rec(value, prefix):
        if isinstance(value, dict):
            for key, child in value.items():
                rec(child, f"{prefix}/{key}" if prefix else str(key))
        else:
            out.append((prefix, value))

    rec(pure, "")
    return out


def _to_host(tree):
    return {path: np.asarray(jax.device_get(value), np.float32) for path, value in _flat(tree)}


def _dot(a, b):
    return float(np.vdot(a.astype(np.float64), b.astype(np.float64)))


def _metrics(reference, candidate):
    totals = {"ALL": [0.0, 0.0, 0.0, 0.0, 0, 0]}
    for path, ref in reference.items():
        cand = candidate[path]
        delta = cand - ref
        group = path.split("/", 1)[0]
        totals.setdefault(group, [0.0, 0.0, 0.0, 0.0, 0, 0])
        for bucket in (totals["ALL"], totals[group]):
            bucket[0] += _dot(ref, ref)
            bucket[1] += _dot(cand, cand)
            bucket[2] += _dot(delta, delta)
            bucket[3] += _dot(ref, cand)
            nonzero = (ref != 0) | (cand != 0)
            bucket[4] += int(np.count_nonzero(nonzero & (np.signbit(ref) != np.signbit(cand))))
            bucket[5] += int(np.count_nonzero(nonzero))
    return totals


def _summary(reference, candidate):
    ref2, cand2, delta2, dot, flips, count = _metrics(reference, candidate)["ALL"]
    return (
        np.sqrt(delta2 / max(ref2, 1e-300)),
        dot / max(np.sqrt(ref2 * cand2), 1e-300),
        flips / max(count, 1),
    )


def _print_comparison(label, reference, candidate, per_group=False):
    metrics = _metrics(reference, candidate)
    ref2, cand2, delta2, dot, flips, count = metrics["ALL"]
    print(
        f"  {label:44s} rel_delta={np.sqrt(delta2 / max(ref2, 1e-300)):.4e} "
        f"cos={dot / max(np.sqrt(ref2 * cand2), 1e-300):+.7f} sign_flips={flips / max(count, 1):.4e}",
        flush=True,
    )
    if per_group:
        for group in sorted(key for key in metrics if key != "ALL"):
            ref2, cand2, delta2, dot, flips, count = metrics[group]
            print(
                f"      {group:22s} rel={np.sqrt(delta2 / max(ref2, 1e-300)):.4e} "
                f"cos={dot / max(np.sqrt(ref2 * cand2), 1e-300):+.7f} sign_flips={flips / max(count, 1):.4e}",
                flush=True,
            )


def _nonfinite(host):
    return sum(int(np.count_nonzero(~np.isfinite(v))) for v in host.values())


def main():
    logging.basicConfig(level=logging.WARNING)
    cfg = _config.get_config(CONFIG)
    cfg = dataclasses.replace(
        cfg,
        exp_name=EXP_NAME,
        batch_size=BATCH,
        num_workers=WORKERS,
        resume=True,
        overwrite=False,
        wandb_enabled=False,
    )

    def variant(**kw):
        return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, **kw))

    variant_cfgs = {
        "explicit_fp32ref": variant(use_cudnn_attention=False, explicit_attention_fp32=True),
        "explicit_bf16": variant(use_cudnn_attention=False),
        "cudnn_bf16": variant(use_cudnn_attention=True, cudnn_attention_dtype="bfloat16"),
        "cudnn_fp16": variant(use_cudnn_attention=True, cudnn_attention_dtype="float16"),
    }

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)

    cfg_ckpt = variant_cfgs["cudnn_bf16"]
    manager, resuming = _checkpoints.initialize_checkpoint_dir(
        cfg_ckpt.checkpoint_dir, keep_period=cfg_ckpt.keep_period, overwrite=False, resume=True
    )
    if not resuming or CHECKPOINT_STEP not in manager.all_steps():
        raise RuntimeError(f"checkpoint {CHECKPOINT_STEP} unavailable; found {manager.all_steps()}")
    state_shape, _ = train_mod.init_train_state(cfg_ckpt, init_rng, mesh, resume=True)
    state = _checkpoints.restore_state(manager, state_shape, None, step=CHECKPOINT_STEP)
    jax.block_until_ready(state)
    print(f"restored checkpoint step={CHECKPOINT_STEP}; train_state.step={int(state.step)}", flush=True)

    print("building loader ...", flush=True)
    loader = _data_loader.create_data_loader(variant_cfgs["explicit_bf16"], sharding=data_sharding, shuffle=True)
    it = iter(loader)
    batches = [next(it) for _ in range(NUM_BATCHES)]

    graphdefs = {}
    for name, vcfg in variant_cfgs.items():
        graphdefs[name] = nnx.graphdef(vcfg.model.create(init_rng))
    print("graphdefs built", flush=True)

    fns = {
        name: jax.jit(functools.partial(_loss_grads_update, graphdefs[name], cfg.trainable_filter, state.tx))
        for name in variant_cfgs
    }

    for bi, (obs, act) in enumerate(batches):
        step_rng = jax.random.fold_in(train_rng, state.step + bi)
        grads_host, updates_host, losses = {}, {}, {}
        for name, fn in fns.items():
            t0 = time.time()
            with sharding.set_mesh(mesh):
                loss, grads, updates = fn(state.opt_state, state.params, step_rng, obs, act)
                losses[name] = float(loss)
                grads_host[name] = _to_host(grads)
                updates_host[name] = _to_host(updates)
            del grads, updates
            print(
                f"batch {bi} {name:18s} loss={losses[name]:.8f} "
                f"nonfinite grads={_nonfinite(grads_host[name])} updates={_nonfinite(updates_host[name])} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

        for ref_name in ("explicit_fp32ref", "explicit_bf16"):
            print(f"\n=== batch {bi}: RAW GRADIENT vs {ref_name} ===", flush=True)
            for name in variant_cfgs:
                if name != ref_name:
                    _print_comparison(name, grads_host[ref_name], grads_host[name])
            print(f"\n=== batch {bi}: ADAM UPDATE vs {ref_name} ===", flush=True)
            for name in variant_cfgs:
                if name != ref_name:
                    _print_comparison(name, updates_host[ref_name], updates_host[name], per_group=(ref_name == "explicit_fp32ref"))
        print("", flush=True)


if __name__ == "__main__":
    main()
