import dataclasses
import json
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.models import pi0_config
import openpi.models.model as _model
from openpi.policies import policy as _policy
from openpi.rlt import action_space as _action_space
from openpi.rlt import config as _rlt_config
from openpi.rlt import features as _features
from openpi.shared import normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

_IDENTITY = {"vla_config": "vla", "vla_checkpoint": "/ckpt", "params_fingerprint": "p", "norm_stats_fingerprint": "n"}
_TOKEN = _rlt_config.RLTModelConfig(embed_dim=32, num_heads=4, dtype="float32")


@dataclasses.dataclass
class _StubVLA:
    identity: dict

    def vla_identity(self):
        return self.identity


def _save_token(root: pathlib.Path, token) -> pathlib.Path:
    step = root / "token" / "7"
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(step / "params", {"params": nnx.state(token)})
    (step / "assets").mkdir()
    metadata = {"model": dataclasses.asdict(_TOKEN), "input_dim": 64, "prefix_cache": {**_IDENTITY, "repo_id": "r"}}
    (step / "assets" / _features.TOKEN_METADATA).write_text(json.dumps(metadata))
    return root / "token"


def test_token_checkpoint_roundtrip_and_identity_check(tmp_path):
    token = _TOKEN.create(64, nnx.Rngs(3))
    run_dir = _save_token(tmp_path, token)
    loaded = _features.load_token_model(run_dir, _StubVLA(_IDENTITY))
    prefix, mask = jnp.ones((2, 10, 64)), jnp.ones((2, 10), bool)
    np.testing.assert_allclose(loaded.encode(prefix, mask), token.encode(prefix, mask), atol=1e-6)
    with pytest.raises(ValueError, match="params_fingerprint"):
        _features.load_token_model(run_dir, _StubVLA({**_IDENTITY, "params_fingerprint": "other"}))


def test_features_match_serving_and_the_cache_path():
    model_config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_horizon=10, max_token_len=64
    )
    vla = model_config.create(jax.random.key(0))
    token = _TOKEN.create(_gemma_width(), nnx.Rngs(1))
    rng = np.random.default_rng(0)
    stats = _normalize.NormStats(
        mean=np.zeros(20), std=np.ones(20), q01=-np.ones(20) - rng.uniform(size=20), q99=np.ones(20)
    )
    stats.q01[18:], stats.q99[18:] = 0.0, 1.0
    data_config = _config.LeRobotBiFlexivDataConfig(repo_id="fake_rlt").create(
        pathlib.Path("/nonexistent"), model_config
    )
    data_config = dataclasses.replace(data_config, norm_stats={"state": stats, "actions": stats})
    inputs = [
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=True),
        *data_config.model_transforms.inputs,
    ]
    outputs = [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(data_config.norm_stats, use_quantiles=True),
        *data_config.data_transforms.outputs,
    ]
    extractor = _features.FeatureExtractor(
        vla,
        token,
        _action_space.ActionSpace.from_data_config(data_config),
        input_transform=_transforms.compose(inputs),
        output_transform=_transforms.compose(outputs),
        ref_num_action_chunks=8,
        num_steps=2,
    )
    state = rng.normal(size=20).astype(np.float32)
    state[18:] = 0.5
    image = rng.integers(0, 255, (3, 224, 224), dtype=np.uint8)  # CHW, as the robot sends it
    obs = {"images": {"head": image, "left_wrist": image}, "state": state, "prompt": "pick up the cube"}

    features = extractor.extract(obs)
    assert features["ref_chunk"].shape == features["ref_exec"].shape == (8, 20)
    assert (np.abs(features["ref_chunk"]) <= 1).all()
    # Same sampler, transforms and rng stream as the serving policy.
    served = _policy.Policy(vla, transforms=inputs, output_transforms=outputs, sample_kwargs={"num_steps": 2}).infer(
        obs
    )  # rng defaults to key(0), the extractor's seed
    np.testing.assert_allclose(features["ref_exec"], served["actions"][:8], atol=1e-5)
    # z_rl equals what the offline cache path (extract_prefix_hidden) feeds the encoder.
    batch = jax.tree.map(lambda x: jnp.asarray(x)[None], _transforms.compose(inputs)(dict(obs)))
    hidden, mask = vla.extract_prefix_hidden(_model.Observation.from_dict(batch))
    np.testing.assert_allclose(features["z_rl"], token.encode(hidden, mask)[0], atol=1e-4)


def _gemma_width() -> int:
    import openpi.models.gemma as _gemma

    return _gemma.get_config("dummy").width


def test_online_rlt_requires_a_default_prompt():
    from openpi.rlt import vla as _vla

    model_config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", pi05=True)
    for prompt in (None, "insert the cable"):
        data_config = _config.LeRobotBiFlexivDataConfig(repo_id="fake_rlt", default_prompt=prompt).create(
            pathlib.Path("/nonexistent"), model_config
        )
        frozen = _vla.FrozenVLA(_config.get_config("debug_pi05"), pathlib.Path("/ckpt"), data_config)
        if prompt is None:
            with pytest.raises(ValueError, match="default_prompt"):
                frozen.check_default_prompt()
        else:
            assert frozen.check_default_prompt() == prompt
