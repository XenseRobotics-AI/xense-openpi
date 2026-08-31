"""Full-model gradient A/B: explicit attention vs cuDNN fused attention.

Identical weights, identical batch, identical rng. The only difference is
Pi0Config.use_cudnn_attention. Reports per-parameter relative gradient error so a
systematic discrepancy can be localised to a module.
"""

import collections
import dataclasses
import functools
import logging
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax._src.lib import cuda_versions
import numpy as np
import optax

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

sys.path.insert(0, "scripts")
import train as train_mod

CONFIG = "pi05_base_bi_flexiv_cube_color_sorting_rtc_0829_h100"
BATCH = 32
WORKERS = 16
N_BATCHES = 3


def _grad_fn(model_def, trainable_filter, params, rng, obs, act):
    model = nnx.merge(model_def, params)
    model.train()

    def loss_fn(m, r, o, a):
        return jnp.mean(m.compute_loss(r, o, a, train=True))

    return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter))(model, rng, obs, act)


def _flat(state):
    pure = state.to_pure_dict()
    out = []

    def rec(d, prefix):
        if isinstance(d, dict):
            for k, v in d.items():
                rec(v, f"{prefix}/{k}" if prefix else str(k))
        else:
            out.append((prefix, d))

    rec(pure, "")
    return out


def main():
    logging.basicConfig(level=logging.WARNING)
    print("cuDNN runtime:", cuda_versions.cudnn_get_version())

    cfg = _config.get_config(CONFIG)
    cfg = dataclasses.replace(cfg, batch_size=BATCH, num_workers=WORKERS, wandb_enabled=False)
    cfg_off = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_cudnn_attention=False))
    cfg_on = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_cudnn_attention=True))

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)

    print("loading pi05_base weights ...")
    state, _ = train_mod.init_train_state(cfg_off, init_rng, mesh, resume=False)
    print("  param global norm:", float(optax.global_norm(state.params)))

    # Same parameter values, two attention implementations.
    gdef_off = state.model_def
    gdef_on = nnx.graphdef(cfg_on.model.create(init_rng))

    f_off = jax.jit(functools.partial(_grad_fn, gdef_off, cfg.trainable_filter))
    f_on = jax.jit(functools.partial(_grad_fn, gdef_on, cfg.trainable_filter))

    print("building data loader (this spawns workers) ...")
    dl = _data_loader.create_data_loader(cfg_off, sharding=data_sharding, shuffle=True)
    it = iter(dl)

    agg = collections.defaultdict(lambda: [0.0, 0.0])

    def to_host(state_tree):
        """Pull one gradient set to numpy so the device copy can be freed."""
        return {k: np.asarray(jax.device_get(v), np.float32) for k, v in _flat(state_tree)}

    for b in range(N_BATCHES):
        obs, act = next(it)
        step_rng = jax.random.fold_in(train_rng, b)

        # One gradient set at a time: 3B params in fp32 is ~12 GB, two live copies
        # plus the tree arithmetic does not fit alongside the activations.
        with sharding.set_mesh(mesh):
            loss_off, g_off = f_off(state.params, step_rng, obs, act)
            loss_off = float(loss_off)
            h_off = to_host(g_off)
        del g_off
        with sharding.set_mesh(mesh):
            loss_on, g_on = f_on(state.params, step_rng, obs, act)
            loss_on = float(loss_on)
            h_on = to_host(g_on)
        del g_on
        # Second identical cuDNN call. cuDNN's backward accumulates with atomics, so
        # this measures the RANDOM part of the cuDNN/explicit gap. Whatever is left
        # over is systematic, and only a systematic part can accumulate over steps.
        with sharding.set_mesh(mesh):
            _, g_on2 = f_on(state.params, step_rng, obs, act)
            h_on2 = to_host(g_on2)
        del g_on2

        sq_off = sq_on = sq_d = 0.0
        rows = []
        for path, av in h_off.items():
            av = av.astype(np.float64)
            bv = h_on[path].astype(np.float64)
            na = float(np.linalg.norm(av))
            nb = float(np.linalg.norm(bv))
            nd = float(np.linalg.norm(bv - av))
            sq_off += na**2
            sq_on += nb**2
            sq_d += nd**2
            rows.append((nd / max(na, 1e-30), path, na))
            key = path.split("/")[0]
            agg[key][0] += nd**2
            agg[key][1] += na**2

        sq_rand = 0.0
        for path, av in h_on.items():
            sq_rand += float(np.linalg.norm(h_on2[path].astype(np.float64) - av.astype(np.float64)) ** 2)
        n_off, n_on, n_d = np.sqrt(sq_off), np.sqrt(sq_on), np.sqrt(sq_d)
        n_rand = np.sqrt(sq_rand)
        print(f"\n=== batch {b} ===")
        print(f"  loss    explicit={loss_off:.6f}  cudnn={loss_on:.6f}  diff={loss_on - loss_off:+.3e}")
        print(f"  |grad|  explicit={n_off:.6f}  cudnn={n_on:.6f}  ratio={n_on / max(n_off, 1e-30):.4f}")
        rel_total = n_d / max(n_off, 1e-30)
        rel_rand = n_rand / max(n_off, 1e-30)
        # Total gap and the random part are orthogonal in expectation.
        rel_sys = np.sqrt(max(rel_total**2 - rel_rand**2, 0.0))
        print(f"  cuDNN vs explicit (total)   = {rel_total:.4e}")
        print(f"  cuDNN vs cuDNN    (random)  = {rel_rand:.4e}")
        print(f"  implied SYSTEMATIC component= {rel_sys:.4e}")
        rows.sort(reverse=True)
        print("  worst 10 parameters by relative gradient difference:")
        for rel, path, na in rows[:10]:
            print(f"    {rel:10.3e}  |g|={na:11.4e}  {path}")
        del h_off, h_on, h_on2

    print("\n=== aggregate by top-level module ===")
    for k, (dn, an) in sorted(agg.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1e-30))):
        print(f"  {np.sqrt(dn / max(an, 1e-30)):10.3e}  {k}")


if __name__ == "__main__":
    main()
