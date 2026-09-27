"""Online RLT features: one jitted pass from a robot observation to what the MLP heads read.

For each observation the frozen VLA samples its reference chunk and, from the
same forward, the prefix hidden states; the frozen phase-one encoder turns
those into ``z_rl``. Outputs, all numpy:

- ``z_rl`` ``(Z,)``, ``state`` ``(S,)`` raw robot state, ``proprio`` ``(S,)`` normalized state,
- ``ref_chunk`` ``(R, A)``: the reference in the actor's normalized space,
- ``ref_exec`` ``(R, A)``: the same reference as absolute robot actions (what serving would execute).
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.rlt import action_space as _action_space
from openpi.rlt import config as _rlt_config
from openpi.rlt import token_model as _token_model
from openpi.rlt import vla as _vla
from openpi.shared import nnx_utils
import openpi.transforms as _transforms

TOKEN_METADATA = "rlt_token.json"
# Cache-identity keys that pin the VLA itself (the dataset keys do not matter online).
_VLA_IDENTITY_KEYS = ("vla_config", "params_fingerprint", "norm_stats_fingerprint")


def resolve_token_checkpoint(path: pathlib.Path | str) -> pathlib.Path:
    """A token step dir, or a ``token/`` run dir (its latest step)."""
    path = pathlib.Path(path).resolve()
    if (path / "assets" / TOKEN_METADATA).is_file():
        return path
    steps = sorted((p for p in path.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)) if path.is_dir() else []
    if not steps:
        raise FileNotFoundError(f"No RLT token checkpoint at {path}.")
    return steps[-1]


def load_token_model(path: pathlib.Path | str, frozen: _vla.FrozenVLA) -> _token_model.RLTTokenTransformer:
    """Load a phase-one checkpoint, refusing one trained on a different VLA checkpoint or norm stats."""
    step_dir = resolve_token_checkpoint(path)
    metadata = json.loads((step_dir / "assets" / TOKEN_METADATA).read_text())
    trained_on = metadata["prefix_cache"]
    current = frozen.vla_identity()
    if mismatched := [key for key in _VLA_IDENTITY_KEYS if trained_on.get(key) != current[key]]:
        raise ValueError(
            f"Token checkpoint {step_dir} was trained on a different VLA ({', '.join(mismatched)} differ)."
        )
    config = _rlt_config.RLTModelConfig(**metadata["model"])
    model = config.create(metadata["input_dim"], nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(_model.restore_params(step_dir / "params"))
    logging.info("Loaded RLT token model from %s", step_dir)
    return nnx.merge(graphdef, state)


class _FrozenNet(nnx.Module):
    def __init__(self, vla: _model.BaseModel, token: _token_model.RLTTokenTransformer, num_steps: int):
        self.vla = vla
        self.token = token
        self.num_steps = num_steps

    def __call__(self, rng: jax.Array, observation: _model.Observation) -> tuple[jax.Array, jax.Array]:
        actions, hidden, mask = self.vla.sample_actions_with_prefix(rng, observation, num_steps=self.num_steps)
        return actions, self.token.encode(hidden, mask)


class FeatureExtractor:
    def __init__(
        self,
        vla: _model.BaseModel,
        token: _token_model.RLTTokenTransformer,
        space: _action_space.ActionSpace,
        *,
        input_transform: _transforms.DataTransformFn,
        output_transform: _transforms.DataTransformFn,
        ref_num_action_chunks: int,
        num_steps: int = 10,
        seed: int = 0,
    ):
        self.space = space
        self.z_dim = token.rl_token.value.shape[-1]
        self._ref_len = ref_num_action_chunks
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._run = nnx_utils.module_jit(_FrozenNet(vla, token, num_steps).__call__)
        self._rng = jax.random.key(seed)

    @classmethod
    def from_vla(
        cls,
        frozen: _vla.FrozenVLA,
        token_checkpoint: pathlib.Path | str,
        rl: _rlt_config.RLConfig,
        *,
        num_steps: int = 10,
        default_prompt: str | None = None,
        seed: int = 0,
    ) -> FeatureExtractor:
        return cls(
            frozen.load_model(),
            load_token_model(token_checkpoint, frozen),
            _action_space.ActionSpace.from_data_config(frozen.data_config),
            input_transform=frozen.input_transforms(default_prompt),
            output_transform=frozen.output_transforms(),
            ref_num_action_chunks=rl.ref_num_action_chunks,
            num_steps=num_steps,
            seed=seed,
        )

    def extract(self, obs: dict) -> dict[str, np.ndarray]:
        return self.extract_batch([obs])[0]

    def extract_batch(self, observations: list[dict], *, pad_to: int | None = None) -> list[dict[str, np.ndarray]]:
        """Features of several observations in one forward; ``pad_to`` fixes the batch shape to avoid recompiles."""
        inputs = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations]
        inputs += [inputs[-1]] * max(0, (pad_to or 0) - len(inputs))
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *inputs)
        self._rng, rng = jax.random.split(self._rng)
        actions, z_rl = jax.device_get(self._run(rng, _model.Observation.from_dict(batch)))
        return [
            self._features(obs, model_state, model_actions, z)
            for obs, model_state, model_actions, z in zip(
                observations, np.asarray(batch["state"]), actions, z_rl, strict=False
            )
        ]

    def _features(self, obs: dict, model_state, model_actions, z_rl) -> dict[str, Any]:
        space = self.space
        ref_exec = self._output_transform({"state": model_state, "actions": model_actions})["actions"]
        if ref_exec.shape[0] < self._ref_len:
            raise ValueError(f"VLA chunk of {ref_exec.shape[0]} steps cannot fill a {self._ref_len}-step reference.")
        ref_exec = np.asarray(ref_exec[: self._ref_len, : space.action_dim], np.float32)
        state = np.asarray(obs["state"], np.float32)[: space.state_dim]
        return {
            "z_rl": np.asarray(z_rl, np.float32),
            "state": state,
            "proprio": np.asarray(space.normalize_state(state)),
            "ref_chunk": np.asarray(space.encode(ref_exec, state)),
            "ref_exec": ref_exec,
        }
