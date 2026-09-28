import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlt import action_space as _action_space
from openpi.rlt import tacxense_reference
import openpi.transforms as _transforms


def bi_flexiv_space(seed: int = 0) -> _action_space.ActionSpace:
    rng = np.random.default_rng(seed)
    q01 = rng.uniform(-1.0, -0.2, 20).astype(np.float32)
    q99 = rng.uniform(0.2, 1.0, 20).astype(np.float32)
    q01[18:], q99[18:] = 0.0, 1.0  # gripper openings
    delta = np.asarray(_transforms.make_bool_mask(18, -1, -1))
    return _action_space.ActionSpace(
        state_q01=q01 - 0.5,
        state_q99=q99 + 0.5,
        action_q01=q01,
        action_q99=q99,
        delta_mask=delta,
        gripper_dims=(18, 19),
        rot6d_blocks=_action_space._rot6d_blocks(delta),
    )


def _state(rng, batch):
    state = rng.normal(size=(batch, 20)).astype(np.float32)
    state[:, 18:] = rng.uniform(0, 1, (batch, 2))
    return state


def test_layout_is_derived_from_the_delta_mask():
    space = bi_flexiv_space()
    assert space.gripper_dims == (18, 19)
    assert space.rot6d_blocks == ((3, 9), (12, 18))
    # XtacUmi keeps each gripper next to its arm.
    assert _action_space._rot6d_blocks(np.asarray(_transforms.make_bool_mask(9, -1, 9, -1))) == ((3, 9), (13, 19))


def test_decode_yields_valid_poses_and_gradients():
    space = bi_flexiv_space()
    rng = np.random.default_rng(1)
    state = jnp.asarray(_state(rng, 4))
    actions = jnp.asarray(rng.uniform(-1, 1, (4, 5, 20)).astype(np.float32))
    executed = np.asarray(space.decode(actions, state))
    for start, end in space.rot6d_blocks:
        c1, c2 = executed[..., start : start + 3], executed[..., start + 3 : end]
        np.testing.assert_allclose(np.linalg.norm(c1, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(c2, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.sum(c1 * c2, axis=-1), 0.0, atol=1e-5)
    assert (executed[..., 18:] >= 0).all()
    assert (executed[..., 18:] <= 1).all()
    assert (np.abs(space.canonicalize(actions, state)) <= 1).all()
    grad = jax.grad(lambda a: space.canonicalize(a, state).sum())(actions)
    assert np.isfinite(grad).all()


def test_degenerate_rotation_falls_back_to_state():
    space = bi_flexiv_space()
    state = _state(np.random.default_rng(2), 1)
    actions = np.array(space.decode(jnp.zeros((1, 1, 20)), state))
    actions[..., 3:9] = 0.0
    decoded = np.asarray(space._project(jnp.asarray(actions), jnp.asarray(state[:, None])))
    expected, _ = _action_space._gram_schmidt(jnp.asarray(state[:, 3:9]))
    np.testing.assert_allclose(decoded[0, 0, 3:9], expected[0], atol=1e-6)


def test_matches_tacxense_codec():
    torch = __import__("pytest").importorskip("torch")
    rlt = tacxense_reference.import_tacxense()
    space = bi_flexiv_space()
    codec = rlt.ActionCodec(
        rlt.ActionLayout.bi_flexiv(tuple(bool(x) for x in space.delta_mask)),
        state_q01=space.state_q01,
        state_q99=space.state_q99,
        action_q01=space.action_q01,
        action_q99=space.action_q99,
    )
    rng = np.random.default_rng(3)
    state = _state(rng, 6)
    normalized = rng.uniform(-1.2, 1.2, (6, 4, 20)).astype(np.float32)
    executed = np.asarray(space.decode(jnp.asarray(normalized), state))

    t_state = torch.from_numpy(state)
    np.testing.assert_allclose(executed, codec.decode(torch.from_numpy(normalized), t_state).numpy(), atol=1e-5)
    np.testing.assert_allclose(
        space.encode(jnp.asarray(executed), state), codec.encode(torch.from_numpy(executed), t_state).numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        space.canonicalize(jnp.asarray(normalized), state),
        codec.canonicalize(torch.from_numpy(normalized), t_state).numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(space.normalize_state(state), codec.normalize_proprio(t_state).numpy(), atol=1e-6)


def test_diagnose_counts_what_encode_corrects():
    space = bi_flexiv_space()
    state = np.zeros(20, np.float32)
    state[[3, 7, 12, 16]] = 1.0  # identity rotations
    state[18:] = 0.5
    actions = np.tile(state, (3, 1))  # hold the current pose: zero deltas, inside q01/q99
    assert space.diagnose(actions, state) == {"out_of_range": 0, "gripper_clips": 0, "rot6d_fallbacks": 0}
    actions[0, 18] = 1.5  # gripper beyond fully open
    actions[1, 3:9] = 0.0  # degenerate rotation
    actions[2, 0] = state[0] + 10 * (space.action_q99[0] - space.action_q01[0])  # far outside q01/q99
    counts = space.diagnose(actions, state)
    assert counts["gripper_clips"] == 1
    assert counts["rot6d_fallbacks"] == 1
    assert counts["out_of_range"] >= 1
    assert space.diagnose(np.full((2, 20), 3.0), state, normalized=True)["out_of_range"] > 0


def test_output_metrics_of_the_reference_itself_are_zero():
    from openpi.rlt import diagnostics

    space = bi_flexiv_space()
    rng = np.random.default_rng(5)
    state = _state(rng, 1)[0]
    reference = rng.uniform(-0.8, 0.8, (4, 20)).astype(np.float32)
    metrics = diagnostics.output_metrics(space, reference, reference, state)
    for key in ("residual_position_mm_max", "residual_rotation_rad_max", "residual_gripper_max"):
        assert metrics[key] < 1e-3, key
    shifted = reference.copy()
    shifted[:, 0] += 0.5
    assert diagnostics.output_metrics(space, shifted, reference, state)["residual_position_mm_max"] > 1
