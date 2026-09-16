from collections.abc import Mapping
import dataclasses
import re
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig,
    lr_schedule: LRScheduleConfig,
    weight_decay_mask: at.PyTree | None = None,
    lr_scales: Mapping[str, float] | None = None,
) -> optax.GradientTransformation:
    """Build the optimizer, optionally with per-parameter-group learning-rate multipliers.

    ``lr_scales`` maps a regex over the ``/``-joined parameter path (e.g.
    ``".*tactile_future_head.*"``) to a multiplier applied to that group's update after
    the base optimizer, i.e. an effective learning rate of ``peak_lr * scale``. A
    parameter matched by several patterns gets the product. Adam-type optimizers
    are invariant to gradient scale, so scaling the *update* is the only way to give
    a group its own LR without a second optimizer state.
    """
    lr = lr_schedule.create()
    tx = optimizer.create(lr, weight_decay_mask=weight_decay_mask)
    for pattern, scale in (lr_scales or {}).items():
        if scale <= 0:
            raise ValueError(f"lr scale for {pattern!r} must be positive, got {scale}")
        tx = optax.chain(tx, optax.masked(optax.scale(scale), _path_mask_fn(pattern)))
    return tx


def _path_mask_fn(pattern: str):
    """Return ``params -> bool pytree`` that is True where the joined param path fullmatches ``pattern``."""
    compiled = re.compile(pattern)

    def mask(tree: at.PyTree) -> at.PyTree:
        return jax.tree_util.tree_map_with_path(lambda path, _: compiled.fullmatch(join_path(path)) is not None, tree)

    return mask


def join_path(path) -> str:
    """``/``-join the dict/attr/sequence keys of a jax key path; the nnx ``.value`` leaf is dropped."""
    parts = []
    for entry in path:
        if isinstance(entry, jax.tree_util.DictKey):
            parts.append(str(entry.key))
        elif isinstance(entry, jax.tree_util.GetAttrKey):
            if entry.name != "value":
                parts.append(entry.name)
        elif isinstance(entry, jax.tree_util.SequenceKey):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return "/".join(parts)
