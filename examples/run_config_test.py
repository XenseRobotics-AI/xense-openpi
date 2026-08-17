"""Tests for the run-YAML layer, and for every run file shipped in the repo.

The shipped-file test is the important one: a run file is only useful if it still
matches the `Args` of the example it drives, and the way that breaks is someone
renaming a flag. Catching it here beats catching it in front of hardware.
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib

import pytest

import examples.run_config as _run_config

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass
class _Args:
    """Stand-in for a real example's Args."""

    run: str | None = None
    robot_recipe: str | None = None
    host: str = "localhost"
    port: int = 8000
    rtc_enabled: bool = False
    subscribe_cameras: tuple[str, ...] = ("head",)


def _write(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(body)
    return path


def test_no_run_gives_plain_defaults(tmp_path: pathlib.Path):
    args = _run_config.load_defaults(_Args, tmp_path, argv=["--args.host", "1.2.3.4"])
    assert args == _Args()


def test_yaml_supplies_defaults(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "robot_recipe: forward-05\nhost: 192.168.2.100\nrtc_enabled: true\n")
    args = _run_config.load_defaults(_Args, tmp_path, argv=["--args.run", "demo"])
    assert (args.robot_recipe, args.host, args.rtc_enabled) == ("forward-05", "192.168.2.100", True)
    assert args.port == 8000, "untouched fields keep the dataclass default"
    assert args.run == "demo"


def test_cli_beats_yaml_beats_defaults(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "host: from-yaml\nport: 9001\n")
    args = _run_config.cli(
        lambda args: args,
        _Args,
        tmp_path,
        argv=["--args.run", "demo", "--args.host", "from-cli"],
    )
    assert args.host == "from-cli"  # CLI wins
    assert args.port == 9001  # YAML wins over the dataclass default
    assert args.rtc_enabled is False  # untouched by both


def test_equals_form_of_the_run_flag(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "host: from-yaml\n")
    args = _run_config.load_defaults(_Args, tmp_path, argv=["--args.run=demo"])
    assert args.host == "from-yaml"


def test_lists_become_tuples(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "subscribe_cameras: [head, left_wrist]\n")
    args = _run_config.load_defaults(_Args, tmp_path, argv=["--args.run", "demo"])
    assert args.subscribe_cameras == ("head", "left_wrist")


def test_unknown_key_is_rejected(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "hots: typo\n")
    with pytest.raises(ValueError, match="not run settings"):
        _run_config.load_defaults(_Args, tmp_path, argv=["--args.run", "demo"])


def test_run_key_inside_a_run_file_is_rejected(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "run: something\n")
    with pytest.raises(ValueError, match="which is the file itself"):
        _run_config.load_defaults(_Args, tmp_path, argv=["--args.run", "demo"])


def test_non_mapping_is_rejected(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "- just\n- a list\n")
    with pytest.raises(ValueError, match="not a YAML mapping"):
        _run_config.load_defaults(_Args, tmp_path, argv=["--args.run", "demo"])


def test_unknown_run_name_lists_what_exists(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "host: x\n")
    with pytest.raises(FileNotFoundError, match="demo"):
        _run_config.load_defaults(_Args, tmp_path, argv=["--args.run", "nope"])


def test_path_form_loads_from_outside_the_directory(tmp_path: pathlib.Path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = _write(elsewhere, "demo", "host: from-path\n")
    args = _run_config.load_defaults(_Args, tmp_path, argv=["--args.run", str(path)])
    assert args.host == "from-path"


def test_describe_names_the_file_and_the_overrides(tmp_path: pathlib.Path):
    _write(tmp_path, "demo", "host: from-yaml\n")
    argv = ["--args.run", "demo", "--args.port", "9999"]
    args = _run_config.cli(lambda args: args, _Args, tmp_path, argv=argv)
    described = _run_config.describe(args, _Args, tmp_path, argv=argv)
    assert "demo.yaml" in described
    assert "port" in described
    assert _run_config.describe(_Args(), _Args, tmp_path, argv=[]).startswith("No run file")


# --- the run files that ship with the repo ---------------------------------


def _shipped_runs() -> list[tuple[str, pathlib.Path]]:
    """(example name, run file) for every YAML under an example's runs/ dir."""
    return [
        (path.parent.parent.name, path)
        for path in sorted(_EXAMPLES_DIR.glob("*/runs/*.yaml"))
        if not path.name.startswith("_")
    ]


def _args_class(example: str) -> type:
    return importlib.import_module(f"examples.{example}.main").Args


@pytest.mark.parametrize(("example", "path"), _shipped_runs(), ids=lambda v: getattr(v, "stem", v))
def test_shipped_run_matches_its_example(example: str, path: pathlib.Path):
    pytest.importorskip("lerobot", reason="robot client deps not installed")
    args_cls = _args_class(example)
    args = _run_config.load_defaults(args_cls, path.parent, argv=["--args.run", path.stem])
    assert args.run == path.stem


def test_some_runs_are_shipped():
    assert _shipped_runs(), "no run files found under examples/*/runs/"
