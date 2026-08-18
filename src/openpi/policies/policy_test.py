import pytest
from xense_client import action_chunk_broker

from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_droid")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_droid")

    example = droid_policy.make_droid_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 8)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_droid")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_droid")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = droid_policy.make_droid_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (8,)
