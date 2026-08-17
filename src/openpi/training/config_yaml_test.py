"""Guards for the config YAMLs that ship with the repo.

`configs/` is now the only place training configs live (bar the generated
RoboArena baselines), so these tests stand in for the equivalence check that used
to compare each YAML against a Python twin in `_CONFIGS`.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

import openpi.training.config as _config
import openpi.training.yaml_loader as _yaml_loader

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "configs" / "_examples"


def _example_files() -> list[pathlib.Path]:
    """Shared example configs. Leading-underscore files are docs, not configs."""
    if not _EXAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in _EXAMPLES_DIR.glob("*.yaml") if not p.name.startswith("_"))


@pytest.mark.parametrize("yaml_path", _example_files(), ids=lambda p: p.stem)
def test_example_loads(yaml_path: pathlib.Path):
    """Every shared example parses into a TrainConfig named after its file."""
    config = _yaml_loader.load(yaml_path)
    assert config.name == yaml_path.stem


@pytest.mark.parametrize("yaml_path", _example_files(), ids=lambda p: p.stem)
def test_example_has_no_machine_local_paths(yaml_path: pathlib.Path):
    """A shared example must be runnable by everyone, so no `/home/<someone>/...`.

    This has bitten the repo before: a checkpoint path from one contributor's
    laptop was committed to the shared config file and nobody else could run it.
    Use a `gs://` URL, or keep the config in `configs/` (gitignored) instead.
    """
    # Values only - a comment recording where a path used to point is fine, and is
    # in fact how the two configs that had this problem document the swap.
    offenders = [
        value
        for value in _string_values(yaml.safe_load(yaml_path.read_text()))
        if value.startswith(("/home/", "/Users/"))
    ]
    assert not offenders, f"{yaml_path.name} points at machine-local paths: {offenders}"


def _string_values(node: object) -> list[str]:
    """Every string leaf in a parsed YAML tree."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [value for child in node.values() for value in _string_values(child)]
    if isinstance(node, list):
        return [value for child in node for value in _string_values(child)]
    return []


def test_examples_dir_is_populated():
    assert _example_files(), f"No example configs found in {_EXAMPLES_DIR}."


def test_config_names_are_unique():
    """A YAML shadowing a generated config would silently win in `get_config`."""
    generated = set(_config._generated_configs())
    from_yaml = set(_config._yaml_config_paths())
    assert not (generated & from_yaml), (
        f"These YAML files shadow generated configs: {sorted(generated & from_yaml)}. "
        "Rename the YAML, or drop the generator entry."
    )


def test_all_configs_load():
    """`all_configs()` is what the train CLI enumerates - none of it may be broken."""
    configs = _config.all_configs()
    assert set(_config._known_config_names()) == set(configs), (
        "A config is listed by name but failed to load; see the warning logged above."
    )
    for name, config in configs.items():
        assert config.name == name


def test_full_reference_yaml_parses():
    """The full-reference YAML documents every field, so it must never rot."""
    reference = _EXAMPLES_DIR / "_FULL_REFERENCE.yaml"
    if not reference.is_file():
        pytest.skip(f"{reference.name} not present; skipping reference parse test.")
    config = _yaml_loader.load(reference)
    assert type(config.model).__name__ in _yaml_loader._registry.MODELS
    assert type(config.weight_loader).__name__ in _yaml_loader._registry.WEIGHT_LOADERS
    assert type(config.lr_schedule).__name__ in _yaml_loader._registry.LR_SCHEDULES
    assert type(config.optimizer).__name__ in _yaml_loader._registry.OPTIMIZERS
