"""Source-level checks for the hardware-only client entry point.

Importing main.py loads the Flexiv SDK, which is intentionally unavailable in
the training/CI environment. These checks still catch wiring regressions that
do not require a robot, including the stale ``args.enable_tactile_sensors``
reference that previously failed only after connecting to the policy server.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

import examples.run_config as _run_config

_MAIN = Path(__file__).with_name("main.py")


def _main_tree() -> ast.Module:
    return ast.parse(_MAIN.read_text())


def _args_fields(tree: ast.Module) -> set[str]:
    args_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    return {
        node.target.id
        for node in args_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _load_args_class():
    """Execute only Args, avoiding main.py's robot-SDK imports."""
    tree = _main_tree()
    args_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    module = ast.fix_missing_locations(ast.Module(body=[args_class], type_ignores=[]))
    namespace = {"Annotated": Annotated, "dataclass": dataclass, "tyro": tyro}
    exec(compile(module, _MAIN, "exec"), namespace)
    return namespace["Args"]


def test_main_only_reads_declared_args_fields() -> None:
    tree = _main_tree()
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args"
    }

    assert referenced <= _args_fields(tree)


def test_tactile_cli_defaults_use_current_lerobot_finger_keys() -> None:
    tree = _main_tree()
    args_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    camera_fields = {
        "left_tactile_top_cam",
        "left_tactile_bottom_cam",
        "right_tactile_top_cam",
        "right_tactile_bottom_cam",
    }
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in args_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in camera_fields
    }

    assert defaults == {
        "left_tactile_top_cam": "left_tactile_left",
        "left_tactile_bottom_cam": "left_tactile_right",
        "right_tactile_top_cam": "right_tactile_left",
        "right_tactile_bottom_cam": "right_tactile_right",
    }


def test_tactile_flag_is_optional_recipe_override(tmp_path: Path) -> None:
    args_class = _load_args_class()

    inherited = _run_config.cli(lambda _: None, args_class, tmp_path, argv=[])
    enabled = _run_config.cli(lambda _: None, args_class, tmp_path, argv=["--args.enable-tactile"])
    disabled = _run_config.cli(lambda _: None, args_class, tmp_path, argv=["--args.no-enable-tactile"])

    assert inherited.enable_tactile is None
    assert enabled.enable_tactile is True
    assert disabled.enable_tactile is False
