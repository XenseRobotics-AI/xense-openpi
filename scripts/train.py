import dataclasses
import functools
import json
import logging
import os
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    with jax.named_scope("train_step/loss_and_grad"):
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    with jax.named_scope("train_step/optimizer_update"):
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    with jax.named_scope("train_step/apply_updates"):
        new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        with jax.named_scope("train_step/ema_update"):
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
                ),
            )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    with jax.named_scope("train_step/grad_norm"):
        grad_norm = optax.global_norm(grads)
    with jax.named_scope("train_step/param_norm"):
        param_norm = optax.global_norm(kernel_params)
    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    t_dl = time.monotonic()
    logging.info("[INIT] calling create_data_loader() ...")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    logging.info(f"[INIT] create_data_loader() done in {time.monotonic()-t_dl:.1f}s")

    t_it = time.monotonic()
    logging.info("[INIT] calling iter(data_loader) (this spawns workers) ...")
    data_iter = iter(data_loader)
    logging.info(f"[INIT] iter() done in {time.monotonic()-t_it:.1f}s — workers spawned")

    t_nb = time.monotonic()
    logging.info("[INIT] waiting for first batch via next(data_iter) ...")
    batch = next(data_iter)
    logging.info(f"[INIT] first batch received in {time.monotonic()-t_nb:.1f}s")

    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # --- timing setup ---
    timing_log_path = config.profile_timing_log_path
    if timing_log_path is None:
        timing_log_path = str(config.checkpoint_dir / "timing.jsonl")
    timing_log_dir = os.path.dirname(timing_log_path)
    if timing_log_dir:
        os.makedirs(timing_log_dir, exist_ok=True)
    timing_file = open(timing_log_path, "a", buffering=1)
    logging.info(f"[TIMING] per-step timing → {timing_log_path}")

    infos = []
    STALL_THRESHOLD_S = 3.0
    for step in pbar:
        t_loop_start = time.monotonic()

        # --- dispatch (Python -> XLA queue) ---
        # When profile_host_sync is False this is dominated by donate_argnums waiting
        # for the previous step's train_state to be ready (i.e. effectively device time).
        # When profile_host_sync is True we explicitly block below so this becomes ~pure
        # Python dispatch cost.
        t0 = time.monotonic()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        t_dispatch_only = time.monotonic() - t0

        # --- device wait (real on-device step time when host_sync is on) ---
        t_device_wait = 0.0
        if config.profile_host_sync:
            t0 = time.monotonic()
            jax.block_until_ready((train_state, info))
            t_device_wait = time.monotonic() - t0

        infos.append(info)

        # --- log reduce + wandb ---
        t_log = 0.0
        if step % config.log_interval == 0:
            t0 = time.monotonic()
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
            t_log = time.monotonic() - t0

        # --- next batch ---
        t0 = time.monotonic()
        batch = next(data_iter)
        t_next_batch = time.monotonic() - t0

        # --- checkpoint ---
        t_ckpt = 0.0
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            t0 = time.monotonic()
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            t_ckpt = time.monotonic() - t0

        t_total = time.monotonic() - t_loop_start

        # --- emit per-step timing record ---
        timing_record = {
            "step": int(step),
            "total_s": round(t_total, 4),
            "dispatch_only_s": round(t_dispatch_only, 4),
            "device_wait_s": round(t_device_wait, 4),
            "next_batch_s": round(t_next_batch, 4),
            "log_s": round(t_log, 4),
            "ckpt_s": round(t_ckpt, 4),
            "host_sync": bool(config.profile_host_sync),
        }
        timing_file.write(json.dumps(timing_record) + "\n")

        if config.profile_log_timing_to_wandb and config.wandb_enabled:
            wandb.log(
                {
                    "timing/total_s": t_total,
                    "timing/dispatch_only_s": t_dispatch_only,
                    "timing/device_wait_s": t_device_wait,
                    "timing/next_batch_s": t_next_batch,
                    "timing/log_s": t_log,
                    "timing/ckpt_s": t_ckpt,
                },
                step=step,
            )

        if (
            t_total > STALL_THRESHOLD_S
            or t_next_batch > STALL_THRESHOLD_S
            or t_dispatch_only > STALL_THRESHOLD_S
            or t_device_wait > STALL_THRESHOLD_S
            or t_ckpt > STALL_THRESHOLD_S
        ):
            pbar.write(
                f"[STALL step={step}] total={t_total:.2f}s "
                f"dispatch={t_dispatch_only:.2f}s device_wait={t_device_wait:.2f}s "
                f"next_batch={t_next_batch:.2f}s log={t_log:.2f}s ckpt={t_ckpt:.2f}s"
            )
        if step % 50 == 0:
            pbar.write(
                f"[TIMING step={step}] total={t_total:.2f}s "
                f"dispatch={t_dispatch_only:.2f}s device_wait={t_device_wait:.2f}s "
                f"next_batch={t_next_batch:.2f}s log={t_log:.2f}s ckpt={t_ckpt:.2f}s"
            )

    timing_file.close()
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
