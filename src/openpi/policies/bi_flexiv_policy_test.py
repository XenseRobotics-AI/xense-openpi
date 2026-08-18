"""Pins the BiFlexiv 20D dim layout that its docstrings describe.

The docstrings used to say grippers sat at dims 9 and 19 (each next to its own
TCP) while the delta mask, the lerobot driver and the deployment example all put
them at 18 and 19. Nothing broke at runtime - these transforms pass state through
untouched - but the wrong layout is exactly the kind of thing a reader builds a
conversion layer against. So the layout gets a test, not just a comment.
"""

import numpy as np

from openpi.policies import bi_flexiv_policy
from openpi.policies import xtac_umi_policy
import openpi.transforms as _transforms


def test_delta_mask_puts_both_grippers_last():
    """18 TCP dims delta-encoded, the two trailing gripper dims absolute."""
    mask = _transforms.make_bool_mask(18, -1, -1)

    assert len(mask) == 20
    absolute = [index for index, is_delta in enumerate(mask) if not is_delta]
    assert absolute == [18, 19], "grippers are at 18/19, not per-side-grouped at 9/19"


def test_biflexiv_and_xtac_umi_layouts_are_deliberately_different():
    """A guard against 'harmonizing' the two into one order by mistake.

    They differ because the rigs differ: the Flexiv driver emits both TCPs then
    both grippers, the handheld XTac-UMI rig emits per side in turn. Neither is
    converted to the other anywhere in this repo.
    """
    bi_flexiv_absolute = [i for i, d in enumerate(_transforms.make_bool_mask(18, -1, -1)) if not d]
    xtac_umi_absolute = [i for i, d in enumerate(_transforms.make_bool_mask(9, -1, 9, -1)) if not d]

    assert bi_flexiv_absolute == [18, 19]
    assert xtac_umi_absolute == [9, 19]
    assert bi_flexiv_absolute != xtac_umi_absolute


def test_outputs_trim_to_20_dims():
    padded = np.random.rand(10, 32)
    result = bi_flexiv_policy.BiFlexivOutputs()({"actions": padded})

    assert result["actions"].shape == (10, 20)
    np.testing.assert_array_equal(result["actions"], padded[:, :20])


def test_inputs_keep_head_unmasked_and_fill_missing_wrists():
    """BiFlexiv's camera contract is the inverse of XTac-UMI's: head is the anchor."""
    example = bi_flexiv_policy.make_bi_flexiv_example()
    del example["images"]["left_wrist"]

    result = bi_flexiv_policy.BiFlexivInputs()(example)

    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.False_
    assert not result["image"]["left_wrist_0_rgb"].any()
    assert result["image_mask"]["right_wrist_0_rgb"] == np.True_
    # XTac-UMI does the opposite - wrists required, base masked out by default.
    assert xtac_umi_policy.XtacUmiInputs.EXPECTED_CAMERAS == ("left_wrist", "right_wrist")
