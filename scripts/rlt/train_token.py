"""RLT phase one, step 2: train the RL-token encoder-decoder on the prefix cache.

Reads the cache written by ``scripts/rlt/precompute_prefix.py``; the VLA
checkpoint is only fingerprinted, to refuse a cache built from different
weights, norm stats or data. Single process over all local GPUs, like
``scripts/train.py``: the batch is sharded across devices and
``token_training.fsdp_devices`` > 1 also shards params and optimizer state.

``--delete-cache-on-finish`` (on by default) deletes the cache once training
reaches ``num_train_steps``: phase two computes features live and never reads
it. Pass ``--no-delete-cache-on-finish`` to keep it, e.g. to train again with
other hyperparameters. An interrupted run never deletes it.

    mamba activate lerobot-xense
    python scripts/rlt/train_token.py <rlt_config> --exp-name <run> [--resume | --overwrite]

Checkpoints: ``<checkpoint_base_dir>/<config>/<exp-name>/token/<step>/`` with
``params/`` (the encoder-decoder), ``train_state/`` and ``assets/rlt_token.json``
(model config plus the cache identity it was trained against).
"""

import argparse
import dataclasses
import json
import logging
import pathlib

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

from openpi.rlt import config as _rlt_config
from openpi.rlt import prefix_cache
from openpi.rlt import vla as _vla
import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def _init_train_state(config: _rlt_config.RLTConfig, input_dim: int, mesh: jax.sharding.Mesh):
    tt = config.token_training
    tx = _optimizer.create_optimizer(
        # RLinf stage-one optimizer: AdamW(0.9, 0.95), weight decay 1e-10, global-norm clip 1.0.
        _optimizer.AdamW(b1=0.9, b2=0.95, weight_decay=1e-10, clip_gradient_norm=1.0),
        _optimizer.CosineDecaySchedule(
            warmup_steps=tt.warmup_steps, peak_lr=tt.peak_lr, decay_steps=tt.num_train_steps, decay_lr=tt.min_lr
        ),
    )

    def init(rng: jax.Array) -> training_utils.TrainState:
        model = config.model.create(input_dim, nnx.Rngs(rng))
        params = nnx.state(model)
        return training_utils.TrainState(
            step=0, params=params, model_def=nnx.graphdef(model), tx=tx, opt_state=tx.init(params), ema_decay=None
        )

    rng = jax.random.key(config.seed)
    shape = jax.eval_shape(init, rng)
    state_sharding = sharding.fsdp_sharding(shape, mesh, log=True)
    return jax.jit(init, out_shardings=state_sharding)(rng), state_sharding


def _train_step(state: training_utils.TrainState, batch: dict) -> tuple[training_utils.TrainState, dict]:
    model = nnx.merge(state.model_def, state.params)
    (loss, z_rl), grads = nnx.value_and_grad(lambda m: m(batch["hidden"], batch["mask"]), has_aux=True)(model)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(params),
        "z_rl_norm": jnp.linalg.norm(z_rl, axis=-1).mean(),
    }
    return dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="RLT config name (configs/rlt/<name>.yaml)")
    parser.add_argument("--exp-name", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete-cache-on-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete the prefix cache once training reaches num_train_steps (default: on)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    config = dataclasses.replace(
        _rlt_config.get_config(args.config), exp_name=args.exp_name, resume=args.resume, overwrite=args.overwrite
    )
    tt = config.token_training
    if tt.batch_size % jax.device_count():
        raise ValueError(f"batch_size {tt.batch_size} must be a multiple of {jax.device_count()} devices.")
    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))

    dataset = prefix_cache.PrefixCacheDataset(tt.prefix_cache_dir)
    identity = _vla.resolve(tt).cache_identity(tt)
    prefix_cache.check_identity(dataset.metadata, identity)
    if dataset.seq_len > config.model.prefix_seq_len:
        raise ValueError(f"Cached prefixes have {dataset.seq_len} tokens > model.prefix_seq_len.")
    input_dim = dataset.metadata["hidden_dim"]
    logging.info(
        "Prefix cache %s: %d frames of (%d, %d)", tt.prefix_cache_dir, len(dataset), dataset.seq_len, input_dim
    )

    mesh = sharding.make_mesh(tt.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.token_checkpoint_dir, keep_period=tt.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    if config.wandb_enabled:
        wandb.init(
            project=config.project_name,
            name=f"{config.name}/{config.exp_name}",
            config=dataclasses.asdict(config),
            resume="allow" if resuming else None,
        )
    else:
        wandb.init(mode="disabled")

    state, state_sharding = _init_train_state(config, input_dim, mesh)
    if resuming:
        state = _checkpoints.restore_state(manager, state, None)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    logging.info("RLT token model: %.1fM params, sharded over %s", param_count / 1e6, dict(mesh.shape))

    train_step = jax.jit(
        _train_step,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(0,),
    )
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=tt.batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=tt.num_workers,
        seed=config.seed,
    )
    run_metadata = {"model": dataclasses.asdict(config.model), "input_dim": input_dim, "prefix_cache": identity}

    def save_assets(directory) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rlt_token.json").write_text(json.dumps(run_metadata, indent=2))

    start_step = int(state.step)
    data_iter = iter(loader)
    infos = []
    for step in tqdm.tqdm(range(start_step, tt.num_train_steps), initial=start_step, total=tt.num_train_steps):
        with sharding.set_mesh(mesh):
            state, info = train_step(state, next(data_iter))
        infos.append(info)
        if (step + 1) % tt.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            logging.info("Step %d: %s", step + 1, ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step + 1)
            infos = []
        if (step + 1) % tt.save_interval == 0 or step + 1 == tt.num_train_steps:
            _checkpoints.save_train_state(manager, state, step + 1, save_assets)

    manager.wait_until_finished()
    wandb.finish()

    if int(state.step) >= tt.num_train_steps and args.delete_cache_on_finish:
        freed = prefix_cache.delete(tt.prefix_cache_dir)
        logging.info("Training complete; deleted prefix cache %s (%.1f GiB freed).", tt.prefix_cache_dir, freed / 2**30)


if __name__ == "__main__":
    main()
