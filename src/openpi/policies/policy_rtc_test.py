import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import bi_elite_policy
from openpi.policies import bi_flexiv_policy
from openpi.policies import policy as _policy


class _CaptureModel(_model.BaseModel):
    def __init__(self, action_dim: int = 20, action_horizon: int = 4):
        super().__init__(action_dim=action_dim, action_horizon=action_horizon, max_token_len=16)
        self.sample_kwargs = None

    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        raise NotImplementedError

    def sample_actions(self, rng, observation, **kwargs):
        self.sample_kwargs = kwargs
        batch = observation.state.shape[0]
        return np.zeros((batch, self.action_horizon, self.action_dim), dtype=np.float32)


@pytest.mark.parametrize("per_side_grouped", [False, True], ids=["bi_flexiv", "bi_rebot_b601"])
def test_rtc_prev_chunk_left_over_uses_training_action_space(per_side_grouped, monkeypatch):
    # Capture the sampler arguments directly: module_jit runs on a reconstructed
    # model, so Python-side writes do not update this test's model instance.
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda fn: fn)
    model = _CaptureModel()
    # Rebot reuses the Elite transforms: grippers are at 9 and 19, not 18 and 19.
    delta_mask = bi_elite_policy.DELTA_ACTION_MASK if per_side_grouped else _transforms.make_bool_mask(18, -1, -1)
    norm_stats = {
        "state": _transforms.NormStats(
            mean=np.zeros(20, dtype=np.float32),
            std=np.ones(20, dtype=np.float32),
            q01=np.zeros(20, dtype=np.float32),
            q99=np.ones(20, dtype=np.float32),
        ),
        "actions": _transforms.NormStats(
            mean=np.full(20, 10.0, dtype=np.float32),
            std=np.full(20, 2.0, dtype=np.float32),
            q01=np.zeros(20, dtype=np.float32),
            q99=np.ones(20, dtype=np.float32),
        ),
    }
    policy = _policy.Policy(
        model,
        transforms=[
            bi_elite_policy.BiEliteInputs() if per_side_grouped else bi_flexiv_policy.BiFlexivInputs(),
            _transforms.DeltaActions(delta_mask),
            _transforms.Normalize(norm_stats),
        ],
    )

    obs = {
        "state": np.arange(20, dtype=np.float32),
        "images": {
            "head": np.zeros((3, 4, 4), dtype=np.uint8),
            "left_wrist": np.zeros((3, 4, 4), dtype=np.uint8),
            "right_wrist": np.zeros((3, 4, 4), dtype=np.uint8),
        },
        # This test isolates action transforms; no tokenizer runs, so there
        # must be no raw string left when Policy converts inputs to JAX arrays.
    }
    prev_chunk_abs = np.stack(
        [
            np.arange(20, dtype=np.float32) + 100.0,
            np.arange(20, dtype=np.float32) + 200.0,
        ],
        axis=0,
    )
    prev_chunk_abs.setflags(write=False)

    policy.infer(obs, prev_chunk_left_over=prev_chunk_abs, inference_delay=2)

    transformed = model.sample_kwargs["prev_chunk_left_over"]
    assert transformed.shape == (1, 2, 20)

    expected = prev_chunk_abs.copy()
    if per_side_grouped:
        expected[:, :9] -= obs["state"][:9]
        expected[:, 10:19] -= obs["state"][10:19]
    else:
        expected[:, :18] -= obs["state"][:18]
    expected = (expected - 10.0) / 2.000001

    np.testing.assert_allclose(np.asarray(transformed[0]), expected, rtol=1e-5, atol=1e-5)
