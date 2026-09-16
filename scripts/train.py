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
from jax._src.lib import cuda_versions
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


def _array_to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert one HWC image from model/data range to uint8 for W&B."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)

    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)

    image = image.astype(np.float32)
    image_min = float(np.nanmin(image))
    image_max = float(np.nanmax(image))
    if image_min >= -1.05 and image_max <= 1.05:
        image = (np.clip(image, -1.0, 1.0) + 1.0) * 127.5
    elif image_min >= -0.05 and image_max <= 1.05:
        image = np.clip(image, 0.0, 1.0) * 255.0
    else:
        image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8)


def _make_wandb_image_strip(
    images: dict[str, at.ArrayLike],
    *,
    image_keys: tuple[str, ...],
    sample_index: int,
    caption_extra: str = "",
) -> wandb.Image:
    strip = np.concatenate(
        [_array_to_uint8_image(np.asarray(jax.device_get(images[key][sample_index]))) for key in image_keys],
        axis=1,
    )
    caption = " | ".join(image_keys)
    if caption_extra:
        caption = f"{caption} ({caption_extra})"
    return wandb.Image(strip, caption=caption)


def _validate_and_log_first_batch_images(config: _config.TrainConfig, obs: _model.Observation) -> None:
    image_keys = tuple(obs.images.keys())
    batch_size = int(next(iter(obs.images.values())).shape[0])
    num_examples = min(5, batch_size)

    camera_views = [
        _make_wandb_image_strip(obs.images, image_keys=image_keys, sample_index=i) for i in range(num_examples)
    ]
    wandb_payload: dict[str, Any] = {"camera_views": camera_views}

    tactile_keys = tuple(getattr(config.model, "tactile_image_keys", ()))
    if not tactile_keys:
        wandb.log(wandb_payload, step=0)
        return

    missing_images = [key for key in tactile_keys if key not in obs.images]
    missing_masks = [key for key in tactile_keys if key not in obs.image_masks]
    if missing_images or missing_masks:
        raise RuntimeError(
            "Tactile batch validation failed: "
            f"missing image keys={missing_images}, missing mask keys={missing_masks}, "
            f"available image keys={image_keys}"
        )

    tactile_mask_arrays: dict[str, np.ndarray] = {}
    for key in tactile_keys:
        image = np.asarray(jax.device_get(obs.images[key]))
        mask = np.asarray(jax.device_get(obs.image_masks[key])).astype(bool)
        tactile_mask_arrays[key] = mask

        if image.ndim != 4 or image.shape[-1] != 3:
            raise RuntimeError(f"Tactile image {key} must be NHWC, got shape={image.shape}")
        if image.shape[1:3] != _model.IMAGE_RESOLUTION:
            raise RuntimeError(
                f"Tactile image {key} must be resized to {_model.IMAGE_RESOLUTION}, got {image.shape[1:3]}"
            )
        if mask.shape != image.shape[:1]:
            raise RuntimeError(f"Tactile mask {key} shape={mask.shape} does not match image batch={image.shape[:1]}")
        if not mask.any():
            raise RuntimeError(f"Tactile image {key} exists but all image_mask values are False")

        valid_image = image[mask]
        if not np.isfinite(valid_image).all():
            raise RuntimeError(f"Tactile image {key} contains NaN or Inf")

        valid_min = float(valid_image.min())
        valid_max = float(valid_image.max())
        valid_mean = float(valid_image.mean())
        valid_std = float(valid_image.std())
        if np.issubdtype(valid_image.dtype, np.floating) and (valid_min < -1.05 or valid_max > 1.05):
            raise RuntimeError(
                f"Tactile image {key} is outside expected [-1, 1] range: min={valid_min:.4f}, max={valid_max:.4f}"
            )
        if valid_std <= 1e-6:
            raise RuntimeError(f"Tactile image {key} appears constant: std={valid_std:.6g}")

        logging.info(
            "[TACTILE CHECK] %s shape=%s dtype=%s min=%.4f max=%.4f mean=%.4f std=%.4f mask_true=%d/%d",
            key,
            image.shape,
            image.dtype,
            valid_min,
            valid_max,
            valid_mean,
            valid_std,
            int(mask.sum()),
            int(mask.size),
        )

    tactile_views = []
    for i in range(num_examples):
        mask_summary = ", ".join(f"{key}={bool(tactile_mask_arrays[key][i])}" for key in tactile_keys)
        tactile_views.append(
            _make_wandb_image_strip(
                obs.images,
                image_keys=tactile_keys,
                sample_index=i,
                caption_extra=f"sample={i}, {mask_summary}",
            )
        )
    wandb_payload["tactile_views"] = tactile_views
    wandb.log(wandb_payload, step=0)


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
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None, lr_scales=config.param_lr_scales
    )

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


# Metrics that `train_step` fills with NaN on the steps it skips them on, and that the
# logging reduction therefore has to skip too. Everything else is reduced with a plain
# mean so that a NaN reaches the log instead of disappearing.
_NAN_PLACEHOLDER_METRICS = frozenset({"grad_norm", "param_norm", "aux/grad_ratio"})
# Per-bin auxiliary-loss metrics are NaN whenever a bin has no samples in a step.
_NAN_PLACEHOLDER_PREFIXES = ("tac/",)


def _reduce_with_nanmean(key: str) -> bool:
    return key in _NAN_PLACEHOLDER_METRICS or key.startswith(_NAN_PLACEHOLDER_PREFIXES)


def _aux_loss_weight(config: _config.TrainConfig, step: at.Array) -> at.Array:
    """lambda(step): linear warm-up from 0 to ``aux_loss_weight`` over ``aux_loss_warmup_steps``."""
    weight = jnp.asarray(config.aux_loss_weight, dtype=jnp.float32)
    if config.aux_loss_warmup_steps <= 0:
        return weight
    ramp = jnp.minimum(1.0, (step.astype(jnp.float32) + 1.0) / config.aux_loss_warmup_steps)
    return weight * ramp


def _masked_mean(values: at.Array, mask: at.Array) -> at.Array:
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _combine_losses(losses: dict[str, at.Array], aux_weight: at.Array) -> tuple[at.Array, dict[str, at.Array]]:
    """``flow + lambda * tac`` for the dict contract of ``Pi0TactileFastVit.compute_loss``."""
    flow = jnp.mean(losses["flow"])
    tac = _masked_mean(losses["tac"], losses["tac_mask"])
    metrics = {"loss/flow": flow, "loss/tac": tac, "loss/tac_weight": aux_weight}
    for index, value in enumerate(losses["tac_by_time"]):
        metrics[f"tac/time_bin{index}"] = value
    return flow + aux_weight * tac, metrics


def _model_returns_loss_dict(config: _config.TrainConfig) -> bool:
    """Static (trace-time) answer to "does compute_loss return the flow/tac dict?"."""
    return getattr(config.model, "tactile_future_layer", None) is not None


# Expert-side parameters of the stacked Gemma blocks (the ``_1`` suffix is the action
# expert). Every leaf under ``layers`` carries the block index on axis 0.
_ACTION_EXPERT_BLOCK_PARAMS = nnx_utils.PathRegex(r".*/layers/.*_1(/.*)?")


def _aux_grad_ratio(
    config: _config.TrainConfig,
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    total_grads: nnx.State,
    aux_weight: at.Array,
) -> at.Array:
    """r_g = |grad L_tac| / |grad L_flow| over the expert parameters of blocks 1..m.

    docs/action-conditioned-tactile-pretraining.md section 2.3.4 wants this in 0.1-0.5.
    Costs one extra forward/backward, so train_step only calls it on metric steps and
    only when ``log_aux_grad_ratio`` is set. The flow gradient is recovered as
    ``total - lambda * tac`` rather than with a third pass.
    """
    layer = config.model.tactile_future_layer
    shared_filter = nnx.All(config.trainable_filter, _ACTION_EXPERT_BLOCK_PARAMS)

    def tac_loss(model, rng, observation, actions):
        losses = model.compute_loss(rng, observation, actions, train=True)
        return _masked_mean(losses["tac"], losses["tac_mask"])

    tac_grads = nnx.grad(tac_loss, argnums=nnx.DiffState(0, shared_filter))(model, rng, observation, actions)
    total_subset = total_grads.filter(shared_filter)

    def first_blocks(grads: nnx.State) -> nnx.State:
        return jax.tree.map(lambda g: g[:layer], grads)

    tac_subset = first_blocks(tac_grads)
    flow_subset = jax.tree.map(lambda total, tac: total - aux_weight * tac, first_blocks(total_subset), tac_subset)
    return optax.global_norm(tac_subset) / jnp.maximum(optax.global_norm(flow_subset), 1e-12)


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

    returns_loss_dict = _model_returns_loss_dict(config)
    aux_weight = _aux_loss_weight(config, state.step)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        losses = model.compute_loss(rng, observation, actions, train=True)
        if returns_loss_dict:
            return _combine_losses(losses, aux_weight)
        return jnp.mean(losses), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    with jax.named_scope("train_step/loss_and_grad"):
        (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )

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
        with jax.named_scope("train_step/grad_norm"):
            grad_norm = optax.global_norm(grads)
        with jax.named_scope("train_step/param_norm"):
            param_norm = optax.global_norm(kernel_params)
        aux_grad_ratio = jnp.asarray(jnp.nan, dtype=loss.dtype)
        if returns_loss_dict and config.log_aux_grad_ratio:
            with jax.named_scope("train_step/aux_grad_ratio"):
                # `model` was updated in place above; rebuild the pre-update model so the
                # ratio is measured at the same parameters as `grads`.
                pre_update_model = nnx.merge(state.model_def, state.params)
                pre_update_model.train()
                aux_grad_ratio = _aux_grad_ratio(
                    config, pre_update_model, train_rng, observation, actions, grads, aux_weight
                )
    else:
        # Keep a stable output pytree for both JIT variants. The host-side logging
        # reduction drops these placeholders with nanmean; see _NAN_PLACEHOLDER_METRICS.
        grad_norm = jnp.asarray(jnp.nan, dtype=loss.dtype)
        param_norm = jnp.asarray(jnp.nan, dtype=loss.dtype)
        aux_grad_ratio = jnp.asarray(jnp.nan, dtype=loss.dtype)

    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        **loss_metrics,
    }
    if returns_loss_dict:
        info["aux/grad_ratio"] = aux_grad_ratio
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

    _validate_and_log_first_batch_images(config, batch[0])

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
            train_state, info = ptrain_step(train_rng, train_state, batch, step % config.log_interval == 0)
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
            # Only the norms carry NaN placeholders from the steps that skipped the
            # whole-model reductions, so only they may be reduced with nanmean. `loss`
            # keeps a plain mean: a NaN loss is a divergence signal and has to reach
            # the log rather than being silently dropped from the average.
            reduced_info = jax.device_get(
                {
                    key: jnp.nanmean(value) if _reduce_with_nanmean(key) else jnp.mean(value)
                    for key, value in stacked_infos.items()
                }
            )
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            # tqdm_loggable can drop pbar.write() lines when stdout is redirected.
            # Emit the same diagnostics through the normal logger so unattended
            # training runs always retain loss and gradient trends.
            logging.info("Step %d: %s", step, info_str)
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
    logging.info("Shutting down data loader")
    data_loader.close()
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
