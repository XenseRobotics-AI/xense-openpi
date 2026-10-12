"""Phase-two learner: actor, twin-Q critic, target critic, replay, and the UTD schedule.

Update cadence (TacXense's tuned schedule): nothing trains until replay holds
``warm_up`` transitions; from then on every committed transition earns ``utd``
critic updates, and each ``train`` call spends the whole earned backlog - a
real robot has no separate phase to catch a deficit up later. The actor updates
after every ``critic_actor_ratio``-th critic update, and the target critic
soft-updates after every critic update.

The checkpoint holds everything a resumed run needs: weights, optimizer states,
counters, and the replay buffer with its sampling RNG.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import pickle
import shutil

import flax.nnx as nnx
import jax
import numpy as np
import optax

from openpi.rlt import action_space as _action_space
from openpi.rlt import config as _rlt_config
from openpi.rlt import mlp_policy
from openpi.rlt import q_gate as _q_gate
from openpi.rlt import replay as _replay
from openpi.rlt import td


@dataclasses.dataclass
class Counters:
    rounds: int = 0
    # Transitions ever committed (the UTD budget counts these, not the replay size).
    transitions: int = 0
    critic_updates: int = 0
    actor_updates: int = 0
    # Labeled phases committed and their executed steps (for the warm-up estimate).
    phases: int = 0
    phase_steps: int = 0
    # Chunks executed (the per-chunk logging axis).
    chunks: int = 0


class Learner:
    def __init__(
        self,
        config: _rlt_config.RLConfig,
        space: _action_space.ActionSpace,
        *,
        z_dim: int,
        seed: int = 0,
    ):
        self.config = config
        self.space = space
        dims = {"z_dim": z_dim, "state_dim": space.state_dim, "action_dim": space.action_dim}
        rngs = nnx.Rngs(seed)
        self._actor_def, self.actor = nnx.split(mlp_policy.Actor(config, **dims, rngs=rngs))
        self._critic_def, self.critic = nnx.split(mlp_policy.Critic(config, **dims, rngs=rngs))
        self.target_critic = jax.tree.map(lambda x: x, self.critic)
        self._actor_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.actor_lr))
        self._critic_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.critic_lr))
        self.actor_opt = self._actor_tx.init(self.actor)
        self.critic_opt = self._critic_tx.init(self.critic)
        self.replay = _replay.ReplayBuffer(
            config.buffer_size,
            **dims,
            num_action_chunks=config.num_action_chunks,
            ref_num_action_chunks=config.ref_num_action_chunks,
            seed=seed,
        )
        self.counters = Counters()
        self._rng = jax.random.key(seed + 1)
        self._critic_step = jax.jit(self._critic_step_impl)
        self._actor_step = jax.jit(self._actor_step_impl)
        self._act = jax.jit(self._act_impl)
        self.gate = _q_gate.QGate(config.q_gate)
        self._eval_rng = np.random.default_rng(seed + 2)
        self._gate_q_raw = jax.jit(self._gate_q_raw_impl)
        self._gate_q_canon = jax.jit(self._gate_q_canon_impl)

    # ----- acting -----

    def actor_module(self) -> mlp_policy.Actor:
        return nnx.merge(self._actor_def, self.actor)

    def act(self, features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Collection action: a sampled normalized chunk ``(C, A)`` and its absolute robot actions."""
        self._rng, rng = jax.random.split(self._rng)
        normalized, absolute = jax.device_get(self._act(self.actor, _obs(features), rng))
        return normalized, absolute

    def mean(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """The actor's deterministic normalized chunk (logging only; draws no randomness)."""
        return np.asarray(self._act(self.actor, _obs(features), None)[0])

    def _act_impl(self, actor_state, obs, rng):
        normalized = nnx.merge(self._actor_def, actor_state)(obs, noise_rng=rng)
        return normalized[0], self.space.decode(normalized, obs["state"])[0]

    # ----- training -----

    @property
    def warmed_up(self) -> bool:
        """Replay reached warm_up: training is earned, and open windows execute the actor."""
        return len(self.replay) >= self.config.warm_up

    def pending_updates(self) -> int:
        if not self.warmed_up:
            return 0
        return max(self.counters.transitions * self.config.utd - self.counters.critic_updates, 0)

    def commit(self, rows: list[dict]) -> None:
        """Add a round's transitions; all rows are validated before any is stored."""
        for row in rows:
            self.replay.prepare(row)
        for row in rows:
            self.replay.add(row)
        self.counters.transitions += len(rows)

    def train(self) -> list[dict[str, float]]:
        """Spend the whole earned update backlog; returns per-update metrics."""
        infos = []
        for _ in range(self.pending_updates()):
            batch = jax.tree.map(jax.numpy.asarray, self.replay.sample(self.config.batch_size))
            self._rng, critic_rng, actor_rng = jax.random.split(self._rng, 3)
            self.critic, self.target_critic, self.critic_opt, info = self._critic_step(
                self.critic, self.target_critic, self.critic_opt, self.actor, batch, critic_rng
            )
            self.counters.critic_updates += 1
            if self.counters.critic_updates % self.config.critic_actor_ratio == 0:
                bc_weight, q_weight = self._actor_weights()
                self.actor, self.actor_opt, actor_info = self._actor_step(
                    self.actor, self.actor_opt, self.critic, batch, actor_rng, bc_weight, q_weight
                )
                self.counters.actor_updates += 1
                info = {**info, **actor_info}
            infos.append({key: float(value) for key, value in jax.device_get(info).items()})
        return infos

    # ----- Q gate -----

    def _actor_weights(self) -> tuple[float, float]:
        """``(bc_weight, q_weight)`` for the next actor update.

        Default: the config's constant weights. With ``q_gate.enable``: warmup weights
        while the gate is closed; once open, a linear ramp to the online weights over
        ``q_gate.ramp_updates`` actor updates; a closed-again gate drops straight back.
        """
        cfg = self.config.q_gate
        if not cfg.enable:
            return self.config.bc_weight, self.config.q_weight
        if not self.gate.open:
            return cfg.warmup_bc_weight, cfg.warmup_q_weight
        opened_at = self.gate.opened_at_update
        if opened_at is None:
            opened_at = self.counters.actor_updates
        progress = (
            min(1.0, (self.counters.actor_updates - opened_at + 1) / cfg.ramp_updates)
            if cfg.ramp_updates > 0
            else 1.0
        )
        return (
            cfg.warmup_bc_weight + progress * (cfg.online_bc_weight - cfg.warmup_bc_weight),
            cfg.warmup_q_weight + progress * (cfg.online_q_weight - cfg.warmup_q_weight),
        )

    def _gate_q_raw_impl(self, critic, obs, actions_a, actions_b):
        """Q1 of two action sets on the same observations (Q1 is what drives the actor)."""
        module = nnx.merge(self._critic_def, critic)
        return module(obs, actions_a)[:, 0], module(obs, actions_b)[:, 0]

    def _gate_q_canon_impl(self, critic, obs, actions_a, actions_b):
        """Q1 with actions_b canonicalized first (random actions on clean windows)."""
        actions_b = self.space.canonicalize(actions_b, obs["state"])
        return self._gate_q_raw_impl(critic, obs, actions_a, actions_b)

    def evaluate_gate(self) -> tuple[dict[str, float] | None, str | None]:
        """One gate evaluation over recent replay windows: (``q_gate/`` metrics, event).

        Metrics are computed whenever replay holds enough windows of both classes,
        regardless of ``q_gate.enable`` - the flag only decides whether the state
        machine runs and the actor weights react. ``event`` is "opened"/"closed" on
        transitions. Intervened windows compare Q1 of the executed (human-corrected)
        chunk against the VLA proposal; clean windows compare the executed chunk
        against random actions.
        """
        cfg = self.config.q_gate
        if len(self.replay) == 0:
            return None, None
        intervened = self.replay.row_field("intervention_mask").any(axis=1)
        success_windows = int(self.replay.row_field("success").sum())
        round_ids = self.replay.row_field("round_id")
        max_round = int(round_ids.max())

        def pick(mask: np.ndarray) -> np.ndarray | None:
            idx = np.flatnonzero(mask)
            recent = idx[round_ids[idx] > max_round - cfg.recent_rounds]
            pool = recent if len(recent) >= cfg.min_windows else idx
            if len(pool) < cfg.min_windows:
                return None
            return self._eval_rng.choice(pool, size=min(cfg.eval_windows, len(pool)), replace=False)

        intervened_idx, clean_idx = pick(intervened), pick(~intervened)
        if intervened_idx is None or clean_idx is None:
            return {
                "q_gate/skipped": 1.0,
                "q_gate/success_windows": float(success_windows),
                "q_gate/open": float(self.gate.open),
            }, None

        num_chunks = self.config.num_action_chunks
        iv = self.replay.sample_indices(intervened_idx)
        obs = jax.tree.map(jax.numpy.asarray, iv["curr_obs"])
        proposal = iv["curr_obs"]["ref_chunk"][:, :num_chunks]
        q_exec, q_proposal = self._gate_q_raw(
            self.critic, obs, jax.numpy.asarray(iv["actions"]), jax.numpy.asarray(proposal)
        )
        cl = self.replay.sample_indices(clean_idx)
        clean_obs = jax.tree.map(jax.numpy.asarray, cl["curr_obs"])
        random = self._eval_rng.uniform(-1.0, 1.0, cl["actions"].shape).astype(np.float32)
        q_clean, q_random = self._gate_q_canon(
            self.critic, clean_obs, jax.numpy.asarray(cl["actions"]), jax.numpy.asarray(random)
        )
        q_exec, q_proposal, q_clean, q_random = (
            np.asarray(x, np.float64) for x in jax.device_get((q_exec, q_proposal, q_clean, q_random))
        )

        std = float(np.std(np.concatenate([q_exec, q_proposal]))) + 1e-8
        metrics = {
            "rank_accuracy": float(np.mean(q_exec > q_proposal)),
            "margin_std": float(np.median((q_exec - q_proposal) / std)),
            "clean_accuracy": float(np.mean(q_clean > q_random)),
            "success_windows": float(success_windows),
        }
        event = self.gate.observe(metrics, self.counters.actor_updates) if cfg.enable else None
        out = {f"q_gate/{key}": value for key, value in metrics.items()}
        out["q_gate/open"] = float(self.gate.open)
        out["q_gate/evals"] = float(self.gate.evals)
        return out, event

    def _critic_step_impl(self, critic, target, opt, actor, batch, rng):
        def loss_fn(critic):
            return td.critic_loss(
                nnx.merge(self._critic_def, critic),
                nnx.merge(self._critic_def, target),
                nnx.merge(self._actor_def, actor),
                self.space,
                batch,
                rng,
                gamma=self.config.gamma,
            )

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic)
        updates, opt = self._critic_tx.update(grads, opt, critic)
        critic = optax.apply_updates(critic, updates)
        target = optax.incremental_update(critic, target, self.config.tau)
        return critic, target, opt, {**info, "critic_grad_norm": optax.global_norm(grads)}

    def _actor_step_impl(self, actor, opt, critic, batch, rng, bc_weight, q_weight):
        def loss_fn(actor):
            return td.actor_loss(
                nnx.merge(self._actor_def, actor),
                nnx.merge(self._critic_def, critic),
                self.space,
                batch,
                rng,
                q_weight=q_weight,
                bc_weight=bc_weight,
                reference_dropout_prob=self.config.reference_dropout_prob,
                input_reference=self.config.input_reference,
            )

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor)
        # The BC and Q terms' gradients separately: how hard each pulls, and whether they agree.
        bc_grads = jax.grad(lambda actor: loss_fn(actor)[1]["weighted_bc"])(actor)
        q_grads = jax.grad(lambda actor: -loss_fn(actor)[1]["weighted_q"])(actor)
        updates, opt = self._actor_tx.update(grads, opt, actor)
        flat_bc, flat_q = (jax.numpy.concatenate([x.ravel() for x in jax.tree.leaves(g)]) for g in (bc_grads, q_grads))
        head_bias = grads["net"]["linears"][len(self.config.actor_hidden_dims)]["bias"]
        gripper = head_bias.value.reshape(self.config.num_action_chunks, -1)[:, list(self.space.gripper_dims)]
        return (
            optax.apply_updates(actor, updates),
            opt,
            {
                **info,
                "bc_weight": bc_weight,
                "q_weight": q_weight,
                "gate_open": float(self.gate.open),
                "actor_grad_norm": optax.global_norm(grads),
                "weighted_bc_grad_norm": jax.numpy.linalg.norm(flat_bc),
                "weighted_q_grad_norm": jax.numpy.linalg.norm(flat_q),
                "bc_q_grad_cosine": flat_bc
                @ flat_q
                / (jax.numpy.linalg.norm(flat_bc) * jax.numpy.linalg.norm(flat_q) + 1e-12),
                "gripper_head_bias_grad_norm": jax.numpy.linalg.norm(gripper),
            },
        )

    # ----- checkpointing -----

    def snapshot(self, path: pathlib.Path, extra: dict | None = None) -> None:
        """Weights only (actor, critic, target critic) plus counters and config, loadable by ``load_actor``."""
        state = {
            **(extra or {}),
            **jax.device_get({k: v for k, v in self._state().items() if k in ("actor", "critic", "target_critic")}),
            "counters": dataclasses.asdict(self.counters),
            "config": dataclasses.asdict(self.config),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def _state(self) -> dict:
        return {
            "actor": self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "rng": jax.random.key_data(self._rng),
            "gate": self.gate.state_dict(),
        }

    def save(self, directory: pathlib.Path, extra: dict | None = None) -> None:
        """Write ``directory`` atomically (via a sibling tmp dir)."""
        tmp = directory.with_name(directory.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        state = {
            **jax.device_get(self._state()),
            "counters": dataclasses.asdict(self.counters),
            "replay": self.replay.state_dict(),
            "config": dataclasses.asdict(self.config),
            **(extra or {}),
        }
        with open(tmp / "learner.pkl", "wb") as f:
            pickle.dump(state, f)
        shutil.rmtree(directory, ignore_errors=True)
        tmp.rename(directory)
        logging.info("Saved RL checkpoint %s (round %d)", directory, self.counters.rounds)

    def restore(self, directory: pathlib.Path) -> dict:
        """Load a checkpoint written by ``save``; returns it (for the caller's extra entries)."""
        with open(directory / "learner.pkl", "rb") as f:
            state = pickle.load(f)
        _check_contract(state["config"], self.config, directory)
        current = self._state()
        for key in ("actor", "critic", "target_critic", "actor_opt", "critic_opt"):
            restored = jax.tree.map(np.asarray, state[key])
            if jax.tree.structure(restored) != jax.tree.structure(current[key]):
                raise ValueError(f"RL checkpoint {key} does not match the current model structure.")
            setattr(self, key, jax.tree.map(jax.numpy.asarray, restored))
        self._rng = jax.random.wrap_key_data(state["rng"])
        if "gate" in state:  # checkpoints older than the gate have none; the gate starts closed
            self.gate.load_state_dict(state["gate"])
        self.counters = Counters(**state["counters"])
        self.replay.load_state_dict(state["replay"])
        return state


def _check_contract(saved: dict, config: _rlt_config.RLConfig, directory: pathlib.Path) -> None:
    """Refuse a checkpoint whose training-relevant settings differ; operational ones may change."""
    saved = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k not in config.OPERATIONAL_FIELDS}
    current = config.training_contract()
    if changed := sorted(k for k in current.keys() | saved.keys() if saved.get(k) != current.get(k)):
        raise ValueError(f"RL checkpoint {directory} was trained with different {', '.join(changed)}.")


def save_round(learner: Learner, rl_dir: pathlib.Path, extra: dict) -> None:
    """Save ``rl_dir/<round>`` and delete older rounds except multiples of ``keep_period``."""
    learner.save(rl_dir / str(learner.counters.rounds), extra)
    keep = learner.config.keep_period
    rounds = sorted((p for p in rl_dir.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    for old in rounds[:-1]:
        if keep is None or int(old.name) % keep:
            shutil.rmtree(old)


def _obs(features: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jax.numpy.asarray(features[key])[None] for key in ("z_rl", "state", "proprio", "ref_chunk")}


def load_actor(
    directory: pathlib.Path, config: _rlt_config.RLConfig, space: _action_space.ActionSpace, *, z_dim: int
) -> tuple[mlp_policy.Actor, dict]:
    """The trained actor of a ``Learner.save`` checkpoint dir or a ``snapshot`` file, plus its binding metadata."""
    with open(directory / "learner.pkl" if directory.is_dir() else directory, "rb") as f:
        state = pickle.load(f)
    _check_contract(state["config"], config, directory)
    actor = mlp_policy.Actor(
        config, z_dim=z_dim, state_dim=space.state_dim, action_dim=space.action_dim, rngs=nnx.Rngs(0)
    )
    graphdef, current = nnx.split(actor)
    restored = jax.tree.map(jax.numpy.asarray, state["actor"])
    if jax.tree.structure(restored) != jax.tree.structure(current):
        raise ValueError(f"RL checkpoint {directory} actor does not match the configured architecture.")
    return nnx.merge(graphdef, restored), state.get("binding", {})
