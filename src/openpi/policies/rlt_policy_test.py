import jax.numpy as jnp
import numpy as np

from openpi.policies import rlt_policy
from openpi.rlt import collector_test
from openpi.rlt import learner as _learner


def test_switch_routes_between_actor_and_vla(tmp_path):
    collector, robot, learner = collector_test._setup([])
    learner.save(tmp_path / "0", {"binding": {}})
    actor, _ = _learner.load_actor(tmp_path / "0", learner.config, learner.space, z_dim=collector_test.Z)
    obs = robot._obs()
    features = collector.extractor.extract(obs)
    collector.extractor.extract = lambda _: features
    policy = rlt_policy.RLTPolicy(collector.extractor, actor)

    on = policy.infer(obs)
    assert on["rlt_actor"]
    # The served chunk is the actor's mean (no exploration noise), decoded to absolute actions.
    batched = {key: jnp.asarray(features[key])[None] for key in ("z_rl", "state", "proprio", "ref_chunk")}
    expected = learner.space.decode(actor(batched), batched["state"])[0]
    np.testing.assert_allclose(on["actions"], expected, atol=1e-6)
    assert on["actions"].shape == (collector_test.C, 20)

    off = policy.infer(obs, rlt_switch=False)
    assert not off["rlt_actor"]
    assert off["actions"].shape == (collector_test.R, 20)
    assert not policy.infer(obs)["rlt_actor"]  # the switch latches
