"""Phase-two learner: actor, twin-Q critic, target critic, replay, and the UTD schedule.

Update cadence (TacXense's tuned schedule): nothing trains until replay holds
``warm_up`` transitions; from then on every committed transition earns ``utd``
critic updates. The actor updates after every ``critic_actor_ratio``-th critic
update, and the target critic soft-updates after every critic update.

The earned budget is spent either synchronously (``train`` between rounds, the
robot idles) or, as in the RLT paper, concurrently with collection by a
``BackgroundTrainer`` thread. A single lock guards weights, replay, the RNG and
the update counters, so collection always acts with a complete set of weights
and checkpoints never see half an update.

The checkpoint holds everything a resumed run needs: weights, optimizer states,
counters, and the replay buffer with its sampling RNG.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import pickle
import shutil
import threading

import flax.nnx as nnx
import jax
import numpy as np
import optax

from openpi.rlt import action_space as _action_space
from openpi.rlt import config as _rlt_config
from openpi.rlt import mlp_policy
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
        # Guards weights, optimizer states, replay, the RNG and update counters (see module doc).
        self._lock = threading.RLock()
        # Set by a failed BackgroundTrainer; acting re-raises it so collection does not run on stale weights.
        self._background_error: BaseException | None = None

    # ----- acting -----

    def actor_module(self) -> mlp_policy.Actor:
        return nnx.merge(self._actor_def, self.actor)

    def act(self, features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Collection action: a sampled normalized chunk ``(C, A)`` and its absolute robot actions."""
        self.check_background()
        with self._lock:
            self._rng, rng = jax.random.split(self._rng)
            actor = self.actor
        normalized, absolute = jax.device_get(self._act(actor, _obs(features), rng))
        return normalized, absolute

    def mean(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """The actor's deterministic normalized chunk (logging only; draws no randomness)."""
        self.check_background()
        with self._lock:
            actor = self.actor
        return np.asarray(self._act(actor, _obs(features), None)[0])

    def check_background(self) -> None:
        if self._background_error is not None:
            raise RuntimeError("The background trainer failed.") from self._background_error

    def _act_impl(self, actor_state, obs, rng):
        normalized = nnx.merge(self._actor_def, actor_state)(obs, noise_rng=rng)
        return normalized[0], self.space.decode(normalized, obs["state"])[0]

    # ----- training -----

    @property
    def warmed_up(self) -> bool:
        """Replay reached warm_up: training is earned, and open windows execute the actor."""
        return len(self.replay) >= self.config.warm_up

    def pending_updates(self) -> int:
        with self._lock:
            if not self.warmed_up:
                return 0
            return max(self.counters.transitions * self.config.utd - self.counters.critic_updates, 0)

    def commit(self, rows: list[dict]) -> None:
        """Add a round's transitions; all rows are validated before any is stored."""
        for row in rows:
            self.replay.prepare(row)
        with self._lock:
            for row in rows:
                self.replay.add(row)
            self.counters.transitions += len(rows)

    def train(self) -> list[dict[str, float]]:
        """Spend the whole earned update backlog; returns per-update metrics."""
        infos = []
        while (update := self.train_step()) is not None:
            infos.append(update[1])
        return infos

    def train_step(self) -> tuple[int, dict[str, float]] | None:
        """One earned critic update (plus the actor's, when due): ``(critic update index, metrics)``.

        ``None`` when no update is earned. The gradient steps run outside the lock, so acting and
        committing are never blocked behind them; only one thread may call this at a time.
        """
        with self._lock:
            if self.pending_updates() == 0:
                return None
            batch = jax.tree.map(jax.numpy.asarray, self.replay.sample(self.config.batch_size))
            self._rng, critic_rng, actor_rng = jax.random.split(self._rng, 3)
            index = self.counters.critic_updates + 1
            actor, actor_opt = self.actor, self.actor_opt
        critic, target, critic_opt, info = self._critic_step(
            self.critic, self.target_critic, self.critic_opt, actor, batch, critic_rng
        )
        update_actor = index % self.config.critic_actor_ratio == 0
        if update_actor:
            actor, actor_opt, actor_info = self._actor_step(actor, actor_opt, critic, batch, actor_rng)
            info = {**info, **actor_info}
        info = {key: float(value) for key, value in jax.device_get(info).items()}
        with self._lock:
            self.critic, self.target_critic, self.critic_opt = critic, target, critic_opt
            self.counters.critic_updates = index
            if update_actor:
                self.actor, self.actor_opt = actor, actor_opt
                self.counters.actor_updates += 1
        return index, info

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

    def _actor_step_impl(self, actor, opt, critic, batch, rng):
        def loss_fn(actor):
            return td.actor_loss(
                nnx.merge(self._actor_def, actor),
                nnx.merge(self._critic_def, critic),
                self.space,
                batch,
                rng,
                q_weight=self.config.q_weight,
                bc_weight=self.config.bc_weight,
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
        with self._lock:
            state = self._snapshot_state(extra)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def _snapshot_state(self, extra: dict | None) -> dict:
        return {
            **(extra or {}),
            **jax.device_get({k: v for k, v in self._state().items() if k in ("actor", "critic", "target_critic")}),
            "counters": dataclasses.asdict(self.counters),
            "config": dataclasses.asdict(self.config),
        }

    def _state(self) -> dict:
        return {
            "actor": self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "rng": jax.random.key_data(self._rng),
        }

    def save(self, directory: pathlib.Path, extra: dict | None = None) -> None:
        """Write ``directory`` atomically (via a sibling tmp dir)."""
        tmp = directory.with_name(directory.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with self._lock:
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
        self.counters = Counters(**state["counters"])
        self.replay.load_state_dict(state["replay"])
        return state


class BackgroundTrainer:
    """Spends the learner's earned update budget on a worker thread, concurrently with collection.

    The RLT paper's asynchronous learner: the robot keeps collecting while the actor and critic
    train, and every chunk acts with the latest weights. Rows still reach replay once per round
    (``Learner.commit``), so a discard never has to be undone; the budget they earn is spent as
    fast as the GPU allows. A failure stops the thread and is re-raised by the next ``act``/``mean``
    and by ``drain``.
    """

    def __init__(self, learner: Learner, *, idle_poll_s: float = 0.5):
        self.learner = learner
        self._idle_poll_s = idle_poll_s
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._finish_backlog = False
        self._results: list[tuple[int, dict[str, float]]] = []
        self._results_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="rlt-background-trainer", daemon=True)

    def start(self) -> BackgroundTrainer:
        self._thread.start()
        return self

    def notify(self) -> None:
        """New budget was committed: stop idling."""
        self._wake.set()

    def drain(self) -> list[tuple[int, dict[str, float]]]:
        """The ``(critic update index, metrics)`` of every update finished since the last drain."""
        self.learner.check_background()
        with self._results_lock:
            results, self._results = self._results, []
        return results

    def stop(self, *, finish_backlog: bool) -> None:
        """Stop the thread; with ``finish_backlog``, after it has spent the whole earned budget."""
        self._finish_backlog = finish_backlog
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        try:
            while not (self._stop.is_set() and not self._finish_backlog):
                self._wake.clear()
                update = self.learner.train_step()
                if update is None:
                    if self._stop.is_set():
                        return
                    self._wake.wait(self._idle_poll_s)
                    continue
                with self._results_lock:
                    self._results.append(update)
        except BaseException as exc:
            logging.exception("Background trainer failed")
            self.learner._background_error = exc


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
