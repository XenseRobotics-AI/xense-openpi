"""Every shipped bench recipe must decode.

Recipes are read by a strict typed parser (draccus), which is the point — a
typo, or a knob belonging to the other gripper backend, fails the parse. Without
this test that failure surfaces the first time someone runs the recipe in front
of hardware. lerobot carries `tests/robots/test_recipe_gripper_blocks.py` for
the same reason.

Skipped where the Flexiv stack is not installed (the openpi training env);
runs in the `lerobot-xense` env the robot clients use.
"""

import pytest

pytest.importorskip("lerobot", reason="lerobot not installed")
pytest.importorskip("flexiv_rt", reason="flexiv_rt SDK not installed")

import examples.bi_flexiv_rizon4_rt.recipe as bi_recipe
import examples.flexiv_rizon4_rt.recipe as single_recipe


def test_bimanual_recipes_exist():
    assert bi_recipe.available_recipes(), "no recipes shipped for bi_flexiv_rizon4_rt"


def test_single_arm_recipes_exist():
    assert single_recipe.available_recipes(), "no recipes shipped for flexiv_rizon4_rt"


@pytest.mark.parametrize("name", bi_recipe.available_recipes())
def test_bimanual_recipe_decodes(name):
    config = bi_recipe.load_robot_config(name)

    assert config.left_robot_sn, f"{name}: no left arm SN"
    assert config.right_robot_sn, f"{name}: no right arm SN"
    assert "head" in config.cameras, f"{name}: head camera not pinned"

    # Both sides need a gripper: the 20D state/action vectors end in
    # left/right_gripper.pos, and lerobot only emits those for a side that has
    # one. real_env raises on a missing side, so catch it here instead.
    assert config.left_gripper is not None, f"{name}: no left gripper"
    assert config.right_gripper is not None, f"{name}: no right gripper"
    assert config.left_gripper.side == "left"
    assert config.right_gripper.side == "right"


@pytest.mark.parametrize("name", single_recipe.available_recipes())
def test_single_arm_recipe_decodes(name):
    config = single_recipe.load_robot_config(name)

    assert config.robot_sn, f"{name}: no arm SN"
    # 10D state/action ends in gripper.pos; lerobot omits the key without one.
    assert config.gripper is not None, f"{name}: no gripper block"
    # This driver does no USB-hub auto-discovery, so cameras must be pinned.
    assert config.cameras, f"{name}: no cameras pinned"


def test_cli_overrides_win_over_recipe():
    """A tuning flag beats the recipe — the documented precedence."""
    name = bi_recipe.available_recipes()[0]
    base = bi_recipe.load_robot_config(name)
    forced = bi_recipe.load_robot_config(name, stiffness_ratio=base.stiffness_ratio + 0.1)

    assert forced.stiffness_ratio == pytest.approx(base.stiffness_ratio + 0.1)


@pytest.mark.parametrize("name", bi_recipe.available_recipes())
def test_gripper_block_override_reaches_both_sides(name):
    """`enable_tactile` lives on the gripper block, but callers pass it flat.

    It is the knob every Flexiv inference run sets (tactile frames cost USB
    bandwidth the policy never reads), and it moved down a level upstream when
    the flat gripper_* fields became one typed block — so the routing, and the
    per-side clones __post_init__ rebuilds from it, are worth pinning.
    """
    base = bi_recipe.load_robot_config(name)
    flipped = not base.gripper.enable_tactile
    forced = bi_recipe.load_robot_config(name, enable_tactile=flipped)

    assert forced.gripper.enable_tactile is flipped
    assert forced.left_gripper.enable_tactile is flipped
    assert forced.right_gripper.enable_tactile is flipped
    # Routing one key must not disturb the rest of the block.
    assert forced.gripper.type == base.gripper.type
    assert forced.gripper.auto_discover_cameras == base.gripper.auto_discover_cameras


def test_unknown_override_is_rejected():
    """A name on neither level fails loudly, not by being ignored."""
    name = bi_recipe.available_recipes()[0]
    # The pre-rename spelling of `enable_tactile`: the exact silent no-op this
    # guards against, since dataclasses.replace() would have raised only because
    # the robot config happens to reject unknown kwargs.
    with pytest.raises(ValueError, match="name no field on"):
        bi_recipe.load_robot_config(name, enable_tactile_sensors=False)


def test_none_overrides_are_dropped():
    """`None` means "unset", not "force None" — the recipe's value survives."""
    name = bi_recipe.available_recipes()[0]
    base = bi_recipe.load_robot_config(name)
    passthrough = bi_recipe.load_robot_config(name, stiffness_ratio=None, log_level=None)

    assert passthrough.stiffness_ratio == base.stiffness_ratio
    assert passthrough.log_level == base.log_level


def test_unknown_recipe_name_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="Unknown robot recipe"):
        bi_recipe.load_robot_config("no-such-bench")


def test_wrong_robot_type_is_rejected():
    """A single-arm recipe must not load into the bimanual example."""
    single = single_recipe.resolve_recipe_path(single_recipe.available_recipes()[0])
    with pytest.raises(ValueError, match="but this example drives"):
        bi_recipe.load_robot_config(single)


def test_non_mapping_yaml_is_rejected(tmp_path):
    """A top-level list must fail with the friendly error, not AttributeError."""
    recipe = tmp_path / "listy.yaml"
    recipe.write_text("- robot:\n    type: bi_flexiv_rizon4_rt\n")
    with pytest.raises(ValueError, match="not a YAML mapping"):
        bi_recipe.load_robot_config(recipe)


def test_missing_robot_block_is_rejected(tmp_path):
    recipe = tmp_path / "empty.yaml"
    recipe.write_text("teleop:\n  type: bi_pico4\n")
    with pytest.raises(ValueError, match="no `robot:` block"):
        bi_recipe.load_robot_config(recipe)
