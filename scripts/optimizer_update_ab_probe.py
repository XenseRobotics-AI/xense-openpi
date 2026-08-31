"""Compare optimizer-space updates at a saved training state.

Raw gradient L2 can hide errors in coordinates with a small Adam second moment.
This probe restores the real step-1000 optimizer state, computes gradients for
the same real batch/rng, and compares the actual AdamW updates produced by
explicit attention, cuDNN attention, and the prior multiplicative-noise control.
"""

import dataclasses
import functools
import logging
import sys

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


def _loss_and_update(model_def, trainable_filter, tx, noise_scale, opt_state, params, rng, obs, act):
    model = nnx.merge(model_def, params)
    model.train()

    def loss_fn(m, r, o, a):
        return jnp.mean(m.compute_loss(r, o, a, train=True))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter))(
        model, rng, obs, act
    )
    if noise_scale:
        noise_rng = jax.random.fold_in(rng, 0x6E6F6973)
        grads, _ = train_mod._add_relative_gradient_noise(grads, noise_rng, noise_scale)
    trainable_params = params.filter(trainable_filter)
    updates, _ = tx.update(grads, opt_state, trainable_params)
    return loss, updates


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


def _print_comparison(label, reference, candidate):
    metrics = _metrics(reference, candidate)
    ref2, cand2, delta2, dot, flips, count = metrics["ALL"]
    print(f"\n=== {label} ===", flush=True)
    print(
        f"|update_ref|={np.sqrt(ref2):.8e} |update_candidate|={np.sqrt(cand2):.8e} "
        f"rel_update_delta={np.sqrt(delta2 / ref2):.6e} "
        f"cos={dot / max(np.sqrt(ref2 * cand2), 1e-300):+.8f} "
        f"sign_flip_fraction={flips / max(count, 1):.6e}",
        flush=True,
    )
    for group in sorted(key for key in metrics if key != "ALL"):
        ref2, cand2, delta2, dot, flips, count = metrics[group]
        print(
            f"  {group:20s} rel={np.sqrt(delta2 / max(ref2, 1e-300)):.6e} "
            f"cos={dot / max(np.sqrt(ref2 * cand2), 1e-300):+.8f} "
            f"sign_flips={flips / max(count, 1):.6e}",
            flush=True,
        )


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
    cfg_on = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_cudnn_attention=True))
    cfg_off = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_cudnn_attention=False))

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)

    manager, resuming = _checkpoints.initialize_checkpoint_dir(
        cfg_on.checkpoint_dir,
        keep_period=cfg_on.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming or CHECKPOINT_STEP not in manager.all_steps():
        raise RuntimeError(f"checkpoint {CHECKPOINT_STEP} unavailable; found {manager.all_steps()}")
    state_shape, state_sharding = train_mod.init_train_state(cfg_on, init_rng, mesh, resume=True)
    state = _checkpoints.restore_state(manager, state_shape, None, step=CHECKPOINT_STEP)
    jax.block_until_ready(state)
    print(f"restored checkpoint directory step={CHECKPOINT_STEP}; train_state.step={int(state.step)}", flush=True)

    print("building one-batch loader ...", flush=True)
    loader = _data_loader.create_data_loader(cfg_off, sharding=data_sharding, shuffle=True)
    obs, act = next(iter(loader))
    step_rng = jax.random.fold_in(train_rng, state.step)

    gdef_on = state.model_def
    gdef_off = nnx.graphdef(cfg_off.model.create(init_rng))

    # The optimizer state structure is unchanged when only Adam epsilon changes.
    eps_values = (1e-8, 1e-6, 1e-5)
    tx_by_eps = {1e-8: state.tx}
    for eps in eps_values[1:]:
        cfg_eps = dataclasses.replace(cfg_on, optimizer=dataclasses.replace(cfg_on.optimizer, eps=eps))
        shape_eps, _ = train_mod.init_train_state(cfg_eps, init_rng, mesh, resume=True)
        tx_by_eps[eps] = shape_eps.tx

    host_updates = {}
    variants = [
        ("explicit_eps1e-8", gdef_off, tx_by_eps[1e-8], 0.0),
        ("cudnn_eps1e-8", gdef_on, tx_by_eps[1e-8], 0.0),
        ("explicit_noise015_eps1e-8", gdef_off, tx_by_eps[1e-8], 0.015),
        ("explicit_eps1e-6", gdef_off, tx_by_eps[1e-6], 0.0),
        ("cudnn_eps1e-6", gdef_on, tx_by_eps[1e-6], 0.0),
        ("explicit_eps1e-5", gdef_off, tx_by_eps[1e-5], 0.0),
        ("cudnn_eps1e-5", gdef_on, tx_by_eps[1e-5], 0.0),
    ]
    for name, graphdef, tx, noise_scale in variants:
        fn = jax.jit(functools.partial(_loss_and_update, graphdef, cfg.trainable_filter, tx, noise_scale))
        with sharding.set_mesh(mesh):
            loss, updates = fn(state.opt_state, state.params, step_rng, obs, act)
            loss = float(loss)
            host_updates[name] = _to_host(updates)
        del updates, fn
        print(f"computed {name}: loss={loss:.8f}", flush=True)

    _print_comparison("real cuDNN delta, Adam eps=1e-8", host_updates["explicit_eps1e-8"], host_updates["cudnn_eps1e-8"])
    _print_comparison(
        "prior multiplicative 1.5% noise, Adam eps=1e-8",
        host_updates["explicit_eps1e-8"],
        host_updates["explicit_noise015_eps1e-8"],
    )
    _print_comparison("real cuDNN delta, Adam eps=1e-6", host_updates["explicit_eps1e-6"], host_updates["cudnn_eps1e-6"])
    _print_comparison("real cuDNN delta, Adam eps=1e-5", host_updates["explicit_eps1e-5"], host_updates["cudnn_eps1e-5"])


if __name__ == "__main__":
    main()
