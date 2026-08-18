import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.gemma as _gemma
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_posemb_sincos_supports_tokenwise_positions():
    pos = jnp.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.5]], dtype=jnp.float32)
    emb = _pi0.posemb_sincos(pos, embedding_dim=8, min_period=4e-3, max_period=4.0)

    assert emb.shape == (2, 3, 8)


def test_rmsnorm_supports_tokenwise_adarms_conditioning():
    x = jnp.ones((2, 3, 4), dtype=jnp.float32)
    cond = jnp.ones((2, 3, 4), dtype=jnp.float32)

    rmsnorm = _gemma.RMSNorm()
    variables = rmsnorm.init(jax.random.key(0), x, cond)
    y, gate = rmsnorm.apply(variables, x, cond)

    assert y.shape == x.shape
    assert gate.shape == x.shape
