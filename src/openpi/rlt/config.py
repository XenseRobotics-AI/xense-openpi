"""RLT (RL-token) config schema and YAML loading.

Phase one trains the RL-token encoder-decoder offline on prefix hidden states
of a *frozen* pi0/pi05 SFT checkpoint. It runs in two steps, both driven by
the same YAML:

1. ``scripts/rlt/precompute_prefix.py`` runs the frozen VLA once over the
   dataset and writes the prefix hidden states to ``token_training.prefix_cache_dir``.
2. ``scripts/rlt/train_token.py`` trains the encoder-decoder on that cache.

Configs live in ``configs/rlt/<name>.yaml`` (the file stem is the config name).
``TrainConfig`` lookup only globs the top level of ``configs/``, so these files
never show up as VLA training configs. Example (``configs/rlt/_example.yaml``
lists every field)::

    token_training:
      vla_config: pi05_base_bi_flexiv_pick_up_cube_0824_h100
      vla_checkpoint: ./checkpoints/pi05_base_bi_flexiv_pick_up_cube_0824_h100/run01/59999
      prefix_cache_dir: /data/rlt_cache/pick_up_cube_0824
"""

from __future__ import annotations

import dataclasses
import difflib
import pathlib
from typing import Any, Literal

import flax.nnx as nnx
from omegaconf import OmegaConf

import openpi.training.yaml_loader as _yaml_loader


@dataclasses.dataclass(frozen=True)
class RLTModelConfig:
    """RL-token encoder-decoder architecture; mirrors RLinf's ``RLTTokenTransformer``.

    A single RL token, encoder and decoder of ``num_layers`` pre-LN self-attention
    blocks each. The input width (the VLM hidden size) is not configured here: it
    is read from the prefix cache.
    """

    embed_dim: int = 2048
    # Positional-table length. Must cover the (trimmed) prefix: pi05 with three
    # cameras is 3 * 256 image tokens + max_token_len prompt tokens.
    prefix_seq_len: int = 1024
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    # Compute dtype of the matmuls; params, residual stream and loss stay float32.
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    # Rematerialize each block in the backward pass (trades compute for memory).
    remat: bool = False

    def create(self, input_dim: int, rngs: nnx.Rngs):
        from openpi.rlt.token_model import RLTTokenTransformer

        return RLTTokenTransformer(self, input_dim=input_dim, rngs=rngs)


@dataclasses.dataclass(frozen=True)
class TokenTrainingConfig:
    """Phase one: prefix precompute plus offline encoder-decoder training."""

    # Name of the VLA TrainConfig (configs/<name>.yaml) defining the model and data pipeline.
    vla_config: str
    # Orbax checkpoint step dir of that config, containing params/ and assets/.
    vla_checkpoint: str
    # Where precompute writes the prefix cache and training reads it.
    prefix_cache_dir: str
    # Overrides the VLA config's data.repo_id when set.
    repo_id: str | None = None
    # Keep every n-th dataset frame in the cache. Adjacent 30 Hz frames give nearly
    # identical features, and every frame costs ~3.6 MB of cache on pi05.
    frame_stride: int = 1

    batch_size: int = 256
    num_train_steps: int = 20_000
    # AdamW + warmup-cosine, with RLinf's stage-one betas/clip; peak lr as in TacXense.
    peak_lr: float = 1e-4
    min_lr: float = 2.5e-5
    warmup_steps: int = 1_000
    num_workers: int = 8
    # FSDP shards params/optimizer state over this many devices; 1 is pure data parallel.
    fsdp_devices: int = 1

    checkpoint_base_dir: str = "./checkpoints_rlt"
    save_interval: int = 1_000
    # Checkpoints with step % keep_period == 0 are never deleted.
    keep_period: int | None = 5_000
    log_interval: int = 50

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}.")


@dataclasses.dataclass(frozen=True)
class RLConfig:
    """Phase two: online chunked-TD actor-critic on top of the frozen RL token.

    Defaults are TacXense's tuned real-robot values (its ``rlt_fast`` config).
    State and action widths are not configured: they come from the VLA's norm stats.
    """

    # Actor chunk length C: executed steps per transition, and the TD bootstrap horizon.
    num_action_chunks: int = 20
    # Length of the frozen VLA's reference chunk fed to the actor (only the first C steps are used).
    ref_num_action_chunks: int = 50

    # Heads: ReLU MLPs, LayerNorm on the critic only (TacXense architecture.md 4.51).
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    num_q_heads: int = 2
    actor_layer_norm: bool = False
    critic_layer_norm: bool = True

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # Global-norm clip, applied to actor and critic separately.
    max_grad_norm: float = 10.0
    # Per-step discount; the bootstrap is discounted by gamma**num_action_chunks.
    gamma: float = 0.99
    # Target-critic soft-update rate, applied after every critic update.
    tau: float = 0.005
    # Actor objective: -q_weight * Q1(s, pi(s)) + bc_weight * MSE(pi(s), reference or human action).
    q_weight: float = 0.1
    bc_weight: float = 5.0
    # Probability of zeroing the actor's reference input during training.
    reference_dropout_prob: float = 0.5
    # Fixed std of the Gaussian actor in normalized action space.
    fixed_std: float = 0.002
    # Critic updates per actor update.
    critic_actor_ratio: int = 2

    batch_size: int = 256
    # Replay capacity in transitions (one transition = one C-step window).
    buffer_size: int = 6000
    # Labeled critical phases are cut into C-step windows starting every `replay_stride` env steps.
    replay_stride: int = 2
    # Training (and actor execution) starts once replay holds this many transitions.
    warm_up: int = 250
    # Critic updates earned per committed transition.
    utd: int = 5

    # Online collection.
    # Phase-one token checkpoint (step dir or token/ run dir); None = this run's own token/ dir.
    token_checkpoint: str | None = None
    # Budget in operator rounds (reset -> round end).
    total_rounds: int = 300
    # Address the server listens on; the robot host dials in.
    listen: str = "0.0.0.0:8000"
    # Denoising steps of the frozen VLA's reference sampling.
    num_steps: int = 10
    # Frozen-VLA batch when a labeled phase computes its sliding-window features (wall clock only).
    replay_feature_batch_size: int = 16
    # A held Pico grip becomes a takeover once a controller moves this far from where it was pressed.
    takeover_position_m: float = 0.005
    takeover_rotation_deg: float = 3.0
    # Checkpoint (weights, optimizers, replay) every this many rounds, and at the end.
    save_interval: int = 10

    def __post_init__(self) -> None:
        for name in ("actor_hidden_dims", "critic_hidden_dims"):  # YAML gives lists
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.num_action_chunks > self.ref_num_action_chunks:
            raise ValueError("num_action_chunks must be <= ref_num_action_chunks.")
        if self.num_q_heads < 2:
            raise ValueError("num_q_heads must be >= 2 for clipped double-Q.")
        if self.fixed_std <= 0:
            raise ValueError("fixed_std must be positive.")
        if min(self.critic_actor_ratio, self.replay_stride, self.warm_up, self.utd) < 1:
            raise ValueError("critic_actor_ratio, replay_stride, warm_up and utd must be >= 1.")
        if self.buffer_size < self.warm_up:
            raise ValueError("buffer_size must cover warm_up transitions.")


@dataclasses.dataclass(frozen=True)
class RLTConfig:
    name: str
    token_training: TokenTrainingConfig
    model: RLTModelConfig = dataclasses.field(default_factory=RLTModelConfig)
    rl: RLConfig = dataclasses.field(default_factory=RLConfig)

    project_name: str = "openpi-rlt"
    # Supplied on the CLI.
    exp_name: str = ""
    seed: int = 42
    wandb_enabled: bool = True
    overwrite: bool = False
    resume: bool = False

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Run directory shared by both RLT phases."""
        if not self.exp_name:
            raise ValueError("--exp-name must be set")
        return (pathlib.Path(self.token_training.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def token_checkpoint_dir(self) -> pathlib.Path:
        return self.checkpoint_dir / "token"

    @property
    def rl_checkpoint_dir(self) -> pathlib.Path:
        return self.checkpoint_dir / "rl"

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "rlt"


def get_config(name: str) -> RLTConfig:
    """Load ``configs/rlt/<name>.yaml``."""
    path = _CONFIG_DIR / f"{name}.yaml"
    if path.is_file():
        return load(path)
    known = sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml") if not p.name.startswith("_"))
    closest = difflib.get_close_matches(name, known, n=1)
    hint = f" Did you mean '{closest[0]}'?" if closest else ""
    raise ValueError(f"RLT config '{name}' not found in {_CONFIG_DIR}.{hint}")


def load(path: pathlib.Path | str) -> RLTConfig:
    """Load an RLTConfig from YAML; the file stem becomes the config name."""
    path = pathlib.Path(path)
    return loads(path.read_text(), name=path.stem)


def loads(text: str, name: str) -> RLTConfig:
    raw: Any = OmegaConf.to_container(OmegaConf.create(text), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"RLT config root must be a mapping, got {type(raw).__name__}")
    return _yaml_loader.construct(RLTConfig, {**raw, "name": name})
