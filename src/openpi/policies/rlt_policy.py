"""Serve a trained RLT actor behind the ordinary policy websocket, for closed-loop evaluation.

Each request runs the frozen VLA and token encoder (``FeatureExtractor``). With
the actor switch on, the reply is the actor's deterministic chunk (its mean),
decoded to absolute robot actions ``(C, A)``. With it off, the reply is the VLA's
own chunk ``(R, A)``, the plain VLA policy. A request's
``rlt_switch`` kwarg (the websocket client forwards infer kwargs) sets the
switch for that and later requests.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from xense_client import base_policy as _base_policy

from openpi.rlt import config as _rlt_config
from openpi.rlt import features as _features
from openpi.rlt import learner as _learner
from openpi.rlt import mlp_policy
from openpi.rlt import vla as _vla


class RLTPolicy(_base_policy.BasePolicy):
    def __init__(self, extractor: _features.FeatureExtractor, actor: mlp_policy.Actor, *, use_actor: bool = True):
        self._extractor = extractor
        self._use_actor = use_actor
        space = extractor.space

        @jax.jit
        def act(obs):
            return space.decode(actor(obs), obs["state"])[0]

        self._act = act

    @override
    def infer(self, obs: dict, *, rlt_switch: bool | None = None, **kwargs) -> dict:  # type: ignore[misc]
        if rlt_switch is not None and bool(rlt_switch) != self._use_actor:
            self._use_actor = bool(rlt_switch)
            logging.info("RLT actor switch -> %s", self._use_actor)
        start = time.monotonic()
        features = self._extractor.extract(obs)
        if self._use_actor:
            batched = {key: jnp.asarray(features[key])[None] for key in ("z_rl", "state", "proprio", "ref_chunk")}
            actions = np.asarray(self._act(batched))
        else:
            actions = features["ref_exec"]
        return {
            "actions": actions,
            "state": features["state"],
            "rlt_actor": self._use_actor,
            "policy_timing": {"infer_ms": 1000 * (time.monotonic() - start)},
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {"rlt": True}


def create_rlt_policy(
    config: _rlt_config.RLTConfig, rl_checkpoint: pathlib.Path | str | None = None, *, use_actor: bool = True
) -> RLTPolicy:
    """Serve ``rl_checkpoint`` (default: the run's latest ``rl/<round>``) with the config's VLA and token model."""
    rl = config.rl
    if rl_checkpoint is None:
        rounds = [p for p in config.rl_checkpoint_dir.iterdir() if p.name.isdigit()]
        if not rounds:
            raise FileNotFoundError(f"No RL checkpoint under {config.rl_checkpoint_dir}.")
        rl_checkpoint = max(rounds, key=lambda p: int(p.name))
    rl_checkpoint = pathlib.Path(rl_checkpoint)
    frozen = _vla.resolve(config.token_training)
    token_checkpoint = _features.resolve_token_checkpoint(rl.token_checkpoint or config.token_checkpoint_dir)
    extractor = _features.FeatureExtractor.from_vla(frozen, token_checkpoint, rl, num_steps=rl.num_steps)
    actor, binding = _learner.load_actor(rl_checkpoint, rl, extractor.space, z_dim=extractor.z_dim)
    if binding.get("vla") != frozen.vla_identity():
        raise ValueError(f"{rl_checkpoint} was trained against a different VLA checkpoint.")
    if binding.get("token_checkpoint") != str(token_checkpoint):
        raise ValueError(f"{rl_checkpoint} was trained with token checkpoint {binding.get('token_checkpoint')}.")
    logging.info("Serving RLT actor %s (actor %s by default)", rl_checkpoint, "on" if use_actor else "off")
    return RLTPolicy(extractor, actor, use_actor=use_actor)
