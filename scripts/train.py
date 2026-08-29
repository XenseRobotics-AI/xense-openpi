import dataclasses
import functools
import logging
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
from jax._src.lib import cuda_versions
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


def _all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


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
    compute_metrics: bool = True,
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
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, candidate_opt_state = state.tx.update(grads, state.opt_state, params)
    candidate_params = optax.apply_updates(params, updates)

    # Never commit a non-finite update. This reduction is intentionally done on
    # every step: a single NaN update can poison every later checkpoint before a
    # periodic logging step notices it.
    update_is_finite = _all_finite((loss, grads, updates))
    new_params = jax.tree.map(lambda old, new: jnp.where(update_is_finite, new, old), params, candidate_params)
    new_opt_state = jax.tree.map(
        lambda old, new: jnp.where(update_is_finite, new, old), state.opt_state, candidate_opt_state
    )

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    if compute_metrics:
        # These whole-model reductions are useful for diagnostics but need not run on
        # every step when they are only logged every config.log_interval steps.
        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        grad_norm = optax.global_norm(grads)
        param_norm = optax.global_norm(kernel_params)
    else:
        # Keep a stable output pytree for both JIT variants. The host-side logging
        # reduction ignores these placeholders with nanmean.
        grad_norm = jnp.asarray(jnp.nan, dtype=loss.dtype)
        param_norm = jnp.asarray(jnp.nan, dtype=loss.dtype)

    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "update_is_finite": update_is_finite,
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    cudnn_runtime_version = cuda_versions.cudnn_get_version() if cuda_versions is not None else None
    logging.info(f"JAX cuDNN runtime version: {cudnn_runtime_version}")

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
    logging.info(f"[INIT] create_data_loader() done in {time.monotonic() - t_dl:.1f}s")

    t_it = time.monotonic()
    logging.info("[INIT] calling iter(data_loader) (this spawns workers) ...")
    data_iter = iter(data_loader)
    logging.info(f"[INIT] iter() done in {time.monotonic() - t_it:.1f}s — workers spawned")

    t_nb = time.monotonic()
    logging.info("[INIT] waiting for first batch via next(data_iter) ...")
    batch = next(data_iter)
    logging.info(f"[INIT] first batch received in {time.monotonic() - t_nb:.1f}s")

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
        static_argnums=(3,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    nonfinite_step = None
    xprof_active = False
    # --- stall diagnostics ---
    stall_threshold_s = 3.0  # log a warning when any phase exceeds this
    t_prev_loop_end = time.monotonic()
    for step in pbar:
        if config.xprof_trace_dir is not None and step == config.xprof_start_step:
            # Exclude work queued before the requested range and make the trace boundaries
            # correspond to complete training iterations.
            jax.block_until_ready((train_state, batch))
            logging.info(
                f"Starting JAX/XProf trace at step {step} for {config.xprof_num_steps} steps: "
                f"{config.xprof_trace_dir}"
            )
            jax.profiler.start_trace(
                config.xprof_trace_dir,
                create_perfetto_link=False,
                create_perfetto_trace=True,
            )
            xprof_active = True

        t_loop_start = time.monotonic()

        t0 = time.monotonic()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(
                train_rng,
                train_state,
                batch,
                step % config.log_interval == 0,
            )
        t_dispatch = time.monotonic() - t0

        # JAX dispatch is asynchronous. Fine-grained H2D profiling must first finish the
        # current train step; otherwise block_until_ready(batch) also measures time spent
        # queued behind model compute on the same devices.
        t_compute_sync = 0.0
        if config.profile_data_pipeline:
            t0 = time.monotonic()
            jax.block_until_ready((train_state, info))
            t_compute_sync = time.monotonic() - t0

        infos.append(info)

        t_log = 0.0
        if step % config.log_interval == 0:
            t0 = time.monotonic()
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
            t_log = time.monotonic() - t0
            if float(reduced_info["update_is_finite"]) < 1.0:
                nonfinite_step = step
                pbar.write(
                    f"[NONFINITE step={step}] At least one update since the previous log was skipped; "
                    "stopping training."
                )
                break

        t0 = time.monotonic()
        batch = next(data_iter)
        t_next_batch = time.monotonic() - t0
        data_profile = data_loader.profile_stats()

        t_ckpt = 0.0
        if (step % config.save_interval == 0 and step > start_step) or (
            config.save_final_checkpoint and step == config.num_train_steps - 1
        ):
            t0 = time.monotonic()
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            t_ckpt = time.monotonic() - t0

        t_total = time.monotonic() - t_loop_start
        # Warn on any slow phase OR any slow overall step
        if (
            t_total > stall_threshold_s
            or t_next_batch > stall_threshold_s
            or t_dispatch > stall_threshold_s
            or t_ckpt > stall_threshold_s
        ):
            pbar.write(
                f"[STALL step={step}] total={t_total:.2f}s "
                f"dispatch={t_dispatch:.2f}s next_batch={t_next_batch:.2f}s "
                f"log={t_log:.2f}s ckpt={t_ckpt:.2f}s"
            )
        # Also log every 50 steps a summary line so we can see trends even without stalls
        if step % 50 == 0:
            pbar.write(
                f"[TIMING step={step}] total={t_total:.2f}s "
                f"dispatch={t_dispatch:.2f}s next_batch={t_next_batch:.2f}s "
                f"log={t_log:.2f}s ckpt={t_ckpt:.2f}s"
            )
        if (
            config.profile_data_pipeline
            and data_profile is not None
            and step % config.profile_log_interval == 0
        ):
            pbar.write(
                f"[DATA_PROFILE step={step}] "
                f"dispatch={t_dispatch:.4f}s compute_sync={t_compute_sync:.4f}s "
                f"main_queue_wait={data_profile['main_queue_wait_s']:.4f}s "
                f"worker_getitem={data_profile['worker_getitem_s']:.4f}s "
                f"worker_collate={data_profile['worker_collate_s']:.4f}s "
                f"jax_array_construct={data_profile['jax_array_construct_s']:.4f}s "
                f"h2d_wait={data_profile['h2d_wait_s']:.4f}s "
                f"next_batch_total={t_next_batch:.4f}s"
            )

        if xprof_active and step + 1 == config.xprof_start_step + config.xprof_num_steps:
            jax.block_until_ready((train_state, info, batch))
            jax.profiler.stop_trace()
            xprof_active = False
            logging.info(f"Finished JAX/XProf trace at step {step}")

    if xprof_active:
        jax.block_until_ready((train_state, batch))
        jax.profiler.stop_trace()

    logging.info("Shutting down data loader")
    data_loader.close()
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if nonfinite_step is not None:
        raise FloatingPointError(f"Non-finite loss, gradients, or updates detected by step {nonfinite_step}")


if __name__ == "__main__":
    main(_config.cli())
