"""Measure whether cuDNN-vs-explicit gradient deltas align across real batches.

For each batch, one explicit-attention gradient is compared with the average of
three cuDNN-attention gradients. Averaging suppresses the known nondeterministic
atomic-accumulation component before measuring cross-batch directionality.
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
N_BATCHES = 8
CUDNN_REPEATS = 3


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
            for key, value in d.items():
                rec(value, f"{prefix}/{key}" if prefix else str(key))
        else:
            out.append((prefix, d))

    rec(pure, "")
    return out


def _to_host(state_tree):
    return {key: np.asarray(jax.device_get(value), np.float32) for key, value in _flat(state_tree)}


def _dot(a, b):
    # Accumulate each tensor in float64 without materializing a float64 copy of
    # the whole 3B-parameter model at once.
    return float(np.vdot(a.astype(np.float64), b.astype(np.float64)))


def _module(path):
    return path.split("/", 1)[0]


def _cos(dot, norm_a_sq, norm_b_sq):
    return dot / max(np.sqrt(norm_a_sq * norm_b_sq), 1e-300)


def main():
    logging.basicConfig(level=logging.WARNING)
    print("cuDNN runtime:", cuda_versions.cudnn_get_version(), flush=True)
    print(f"probe: batches={N_BATCHES}, cudnn_repeats={CUDNN_REPEATS}, local_batch={BATCH}", flush=True)

    cfg = _config.get_config(CONFIG)
    cfg = dataclasses.replace(cfg, batch_size=BATCH, num_workers=WORKERS, wandb_enabled=False)
    cfg_off = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_cudnn_attention=False))
    cfg_on = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_cudnn_attention=True))

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)

    print("loading pi05_base weights ...", flush=True)
    state, _ = train_mod.init_train_state(cfg_off, init_rng, mesh, resume=False)
    host_params = _to_host(state.params)
    param_norm_sq = sum(_dot(value, value) for value in host_params.values())
    print("  trainable param norm:", np.sqrt(param_norm_sq), flush=True)

    gdef_off = state.model_def
    gdef_on = nnx.graphdef(cfg_on.model.create(init_rng))
    f_off = jax.jit(functools.partial(_grad_fn, gdef_off, cfg.trainable_filter))
    f_on = jax.jit(functools.partial(_grad_fn, gdef_on, cfg.trainable_filter))

    print("building data loader ...", flush=True)
    dl = _data_loader.create_data_loader(cfg_off, sharding=data_sharding, shuffle=True)
    iterator = iter(dl)

    deltas = []
    delta_norm_sq = []
    pairwise_dot = np.zeros((N_BATCHES, N_BATCHES), np.float64)
    per_batch_summary = []
    per_module_summary = collections.defaultdict(list)

    for batch_idx in range(N_BATCHES):
        obs, act = next(iterator)
        step_rng = jax.random.fold_in(train_rng, batch_idx)

        with sharding.set_mesh(mesh):
            loss_off, g_off = f_off(state.params, step_rng, obs, act)
            loss_off = float(loss_off)
            host_off = _to_host(g_off)
        del g_off

        host_on_sum = None
        cudnn_losses = []
        repeat_diffs_sq = []
        first_on = None
        for repeat in range(CUDNN_REPEATS):
            with sharding.set_mesh(mesh):
                loss_on, g_on = f_on(state.params, step_rng, obs, act)
                cudnn_losses.append(float(loss_on))
                host_on = _to_host(g_on)
            del g_on
            if host_on_sum is None:
                host_on_sum = {key: value.copy() for key, value in host_on.items()}
                first_on = {key: value.copy() for key, value in host_on.items()}
            else:
                repeat_diffs_sq.append(sum(_dot(host_on[key] - first_on[key], host_on[key] - first_on[key]) for key in host_on))
                for key in host_on_sum:
                    host_on_sum[key] += host_on[key]
            del host_on

        host_on_avg = {key: value / CUDNN_REPEATS for key, value in host_on_sum.items()}
        del host_on_sum
        delta = {key: host_on_avg[key] - host_off[key] for key in host_off}

        totals = collections.defaultdict(lambda: collections.defaultdict(float))
        previous_dots = np.zeros(batch_idx, np.float64)
        for path, grad in host_off.items():
            cudnn_grad = host_on_avg[path]
            diff = delta[path]
            param = host_params[path]
            group = _module(path)
            for bucket in (totals["ALL"], totals[group]):
                bucket["g2"] += _dot(grad, grad)
                bucket["c2"] += _dot(cudnn_grad, cudnn_grad)
                bucket["d2"] += _dot(diff, diff)
                bucket["p2"] += _dot(param, param)
                bucket["dg"] += _dot(diff, grad)
                bucket["dp"] += _dot(diff, param)
            for previous_idx, previous_delta in enumerate(deltas):
                previous_dots[previous_idx] += _dot(diff, previous_delta[path])

        current_d2 = totals["ALL"]["d2"]
        pairwise_dot[batch_idx, batch_idx] = current_d2
        for previous_idx, dot in enumerate(previous_dots):
            pairwise_dot[batch_idx, previous_idx] = dot
            pairwise_dot[previous_idx, batch_idx] = dot

        deltas.append(delta)
        delta_norm_sq.append(current_d2)
        random_rel = np.mean(np.sqrt(repeat_diffs_sq)) / max(np.sqrt(totals["ALL"]["g2"]), 1e-300)
        all_stats = totals["ALL"]
        summary = {
            "rel_delta": np.sqrt(all_stats["d2"] / all_stats["g2"]),
            "cos_d_g": _cos(all_stats["dg"], all_stats["d2"], all_stats["g2"]),
            "proj_d_g": all_stats["dg"] / max(all_stats["g2"], 1e-300),
            "cos_d_p": _cos(all_stats["dp"], all_stats["d2"], all_stats["p2"]),
            "proj_d_p": all_stats["dp"] / max(all_stats["p2"], 1e-300),
            "random_rel": random_rel,
        }
        per_batch_summary.append(summary)

        print(f"\n=== batch {batch_idx} ===", flush=True)
        print(
            f"loss explicit={loss_off:.6f} cudnn_mean={np.mean(cudnn_losses):.6f} "
            f"loss_delta={np.mean(cudnn_losses) - loss_off:+.3e}",
            flush=True,
        )
        print(
            f"|g|={np.sqrt(all_stats['g2']):.6f} |delta_mean|/|g|={summary['rel_delta']:.4e} "
            f"repeat_random/|g|={random_rel:.4e}",
            flush=True,
        )
        print(
            f"delta vs grad: cos={summary['cos_d_g']:+.5f} projection/|g|={summary['proj_d_g']:+.4e}; "
            f"delta vs params: cos={summary['cos_d_p']:+.5f} projection/|p|={summary['proj_d_p']:+.4e}",
            flush=True,
        )
        if batch_idx:
            cosines = [
                _cos(pairwise_dot[batch_idx, previous_idx], current_d2, delta_norm_sq[previous_idx])
                for previous_idx in range(batch_idx)
            ]
            print("delta cosine vs prior batches: " + " ".join(f"{value:+.5f}" for value in cosines), flush=True)

        print("module metrics (rel_delta, cos(delta,grad), cos(delta,param)):", flush=True)
        for group in sorted(key for key in totals if key != "ALL"):
            stats = totals[group]
            values = (
                np.sqrt(stats["d2"] / max(stats["g2"], 1e-300)),
                _cos(stats["dg"], stats["d2"], stats["g2"]),
                _cos(stats["dp"], stats["d2"], stats["p2"]),
            )
            per_module_summary[group].append(values)
            print(f"  {group:20s} {values[0]:.4e} {values[1]:+.5f} {values[2]:+.5f}", flush=True)
        del host_off, host_on_avg, first_on

    cosine_matrix = np.eye(N_BATCHES, dtype=np.float64)
    off_diagonal = []
    for row in range(N_BATCHES):
        for col in range(row):
            value = _cos(pairwise_dot[row, col], delta_norm_sq[row], delta_norm_sq[col])
            cosine_matrix[row, col] = cosine_matrix[col, row] = value
            off_diagonal.append(value)

    print("\n=== cross-batch delta cosine matrix ===", flush=True)
    print("        " + " ".join(f"b{i:02d}" for i in range(N_BATCHES)), flush=True)
    for idx, row in enumerate(cosine_matrix):
        print(f"b{idx:02d} " + " ".join(f"{value:+.5f}" for value in row), flush=True)
    print(
        f"off-diagonal cosine: mean={np.mean(off_diagonal):+.6f} "
        f"std={np.std(off_diagonal, ddof=1):.6f} min={np.min(off_diagonal):+.6f} "
        f"max={np.max(off_diagonal):+.6f} n={len(off_diagonal)}",
        flush=True,
    )

    print("\n=== aggregate projections across batches (mean +/- sample std) ===", flush=True)
    for key in ("rel_delta", "random_rel", "cos_d_g", "proj_d_g", "cos_d_p", "proj_d_p"):
        values = np.asarray([row[key] for row in per_batch_summary])
        print(f"{key:12s} {values.mean():+.6e} +/- {values.std(ddof=1):.6e}", flush=True)

    print("\n=== aggregate module metrics (mean rel_delta, cos(delta,grad), cos(delta,param)) ===", flush=True)
    for group in sorted(per_module_summary):
        values = np.asarray(per_module_summary[group])
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=1)
        print(
            f"{group:20s} rel={means[0]:.4e}+/-{stds[0]:.2e} "
            f"cos_g={means[1]:+.5f}+/-{stds[1]:.2e} cos_p={means[2]:+.5f}+/-{stds[2]:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
