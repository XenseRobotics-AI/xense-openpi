"""Tests that the deployment state vector is in the policy's dim order.

This example exists because the Flexiv driver's dim order and the XTac-UMI
recording rig's dim order differ, and this repo resolved that by building the
vector in the rig's order here rather than regrouping dims downstream. If that
ever silently drifts, a trained checkpoint would drive the wrong dims — so the
order is pinned against the policy module's own slice constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.policies import xtac_umi_policy

pytest.importorskip("lerobot", reason="robot client deps not installed")

from examples.xtac_umi_bi_flexiv_rizon4_rt import real_env

# The recording rig's feature order, per side in turn — see
# BiTaccapGripper.observation_features in xense-taccap-lerobot.
_RIG_ORDER = (
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "left_gripper.pos",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "right_gripper.pos",
)


def test_state_keys_match_the_recording_rig_order():
    assert real_env.STATE_KEYS == _RIG_ORDER


def test_state_keys_match_the_policy_slices():
    """Same agreement, stated against the constants the transforms actually use."""
    keys = np.asarray(real_env.STATE_KEYS)

    assert len(keys) == xtac_umi_policy.STATE_DIM
    assert all(key.startswith("left_tcp.") for key in keys[xtac_umi_policy.LEFT_TCP])
    assert list(keys[xtac_umi_policy.LEFT_GRIPPER]) == ["left_gripper.pos"]
    assert all(key.startswith("right_tcp.") for key in keys[xtac_umi_policy.RIGHT_TCP])
    assert list(keys[xtac_umi_policy.RIGHT_GRIPPER]) == ["right_gripper.pos"]


def test_get_qpos_reads_the_named_keys_in_order():
    obs = {key: float(index) for index, key in enumerate(real_env.STATE_KEYS)}
    obs["some.other.key"] = 999.0  # driver dicts carry more than the 20 dims

    qpos = real_env.XTacUmiBiFlexivRizon4RTRealEnv.get_qpos(obs)

    assert qpos.dtype == np.float32
    np.testing.assert_array_equal(qpos, np.arange(xtac_umi_policy.STATE_DIM, dtype=np.float32))


def test_build_action_dict_round_trips_get_qpos():
    """The state read and the action write must agree on which dim is which."""
    values = np.linspace(-0.5, 0.5, xtac_umi_policy.STATE_DIM, dtype=np.float32)
    # Gripper dims are clipped to [0, 1], so give them in-range values or the
    # round trip would be testing the clip rather than the ordering.
    values[9], values[19] = 0.25, 0.75

    action_dict = real_env.XTacUmiBiFlexivRizon4RTRealEnv.build_action_dict(values)

    assert set(action_dict) == set(real_env.STATE_KEYS)
    np.testing.assert_allclose(real_env.XTacUmiBiFlexivRizon4RTRealEnv.get_qpos(action_dict), values, atol=1e-6)


def test_build_action_dict_clips_only_the_grippers():
    values = np.full(xtac_umi_policy.STATE_DIM, 5.0)
    values[9] = 5.0  # left gripper, over-open
    values[19] = -2.0  # right gripper, over-closed

    action_dict = real_env.XTacUmiBiFlexivRizon4RTRealEnv.build_action_dict(values)

    assert action_dict["left_gripper.pos"] == pytest.approx(1.0)
    assert action_dict["right_gripper.pos"] == pytest.approx(0.0)
    # TCP values pass through verbatim, same convention as bi_flexiv_rizon4_rt.
    assert action_dict["left_tcp.x"] == pytest.approx(5.0)
    assert action_dict["right_tcp.r6"] == pytest.approx(5.0)


def test_build_action_dict_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Expected a 20D action"):
        real_env.XTacUmiBiFlexivRizon4RTRealEnv.build_action_dict(np.zeros(18))
