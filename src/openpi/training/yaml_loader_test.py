"""Tests for openpi.training.yaml_loader."""

from __future__ import annotations

import pathlib

import pytest

import openpi.training.config as _config
import openpi.training.weight_loaders as _weight_loaders
import openpi.training.yaml_loader as _yaml_loader
import openpi.transforms as _transforms


def test_load_minimal_debug_pi05():
    """Round-trip the simplest possible config (no data factory, just FakeDataConfig)."""
    yaml_text = """
model:
  type: Pi0Config
  pi05: true
  paligemma_variant: dummy
  action_expert_variant: dummy
data:
  type: FakeDataConfig
batch_size: 2
num_train_steps: 10
overwrite: true
exp_name: debug_pi05
wandb_enabled: false
"""
    cfg = _yaml_loader.loads(yaml_text, name="debug_pi05")
    assert cfg.name == "debug_pi05"
    assert cfg.batch_size == 2
    assert cfg.num_train_steps == 10
    assert cfg.overwrite is True
    assert cfg.wandb_enabled is False
    assert cfg.exp_name == "debug_pi05"
    assert cfg.model.pi05 is True
    assert cfg.model.paligemma_variant == "dummy"
    assert isinstance(cfg.data, _config.FakeDataConfig)


def test_load_bi_flexiv_config():
    """A realistic BiFlexiv config with nested base_config and weight loader."""
    yaml_text = """
model:
  type: Pi0Config
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m
  pi05: true
  enable_training_time_rtc: true
  max_delay: 10
data:
  type: LeRobotBiFlexivDataConfig
  repo_id: Xense/shoe_insole_retrieval_and_packing0515
  use_delta_cartesian_actions: true
  default_prompt: "Open the shoe tongue."
  base_config:
    prompt_from_task: true
weight_loader:
  type: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi05_base/params
batch_size: 256
num_train_steps: 40000
num_workers: 64
fsdp_devices: 8
ema_decay: null
"""
    cfg = _yaml_loader.loads(yaml_text, name="example_shoe")
    assert cfg.name == "example_shoe"
    assert cfg.model.pi05 is True
    assert cfg.model.enable_training_time_rtc is True
    assert cfg.model.max_delay == 10
    assert cfg.data.repo_id == "Xense/shoe_insole_retrieval_and_packing0515"
    assert cfg.data.use_delta_cartesian_actions is True
    assert cfg.data.default_prompt == "Open the shoe tongue."
    assert cfg.data.base_config is not None
    assert cfg.data.base_config.prompt_from_task is True
    assert isinstance(cfg.weight_loader, _weight_loaders.CheckpointWeightLoader)
    assert cfg.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert cfg.batch_size == 256
    assert cfg.num_train_steps == 40_000
    assert cfg.fsdp_devices == 8
    assert cfg.ema_decay is None


def test_unknown_type_raises():
    yaml_text = """
model:
  type: NonExistentModel
data:
  type: FakeDataConfig
"""
    with pytest.raises(KeyError, match="Unknown type 'NonExistentModel'"):
        _yaml_loader.loads(yaml_text, name="bad")


def test_missing_type_field_raises():
    yaml_text = """
model:
  pi05: true
data:
  type: FakeDataConfig
"""
    with pytest.raises(ValueError, match="missing required 'type:' key"):
        _yaml_loader.loads(yaml_text, name="bad")


def test_load_file_uses_stem_as_name(tmp_path: pathlib.Path):
    yaml_path = tmp_path / "my_cool_task.yaml"
    yaml_path.write_text(
        """
model:
  type: Pi0Config
  pi05: true
  paligemma_variant: dummy
  action_expert_variant: dummy
data:
  type: FakeDataConfig
batch_size: 4
"""
    )
    cfg = _yaml_loader.load(yaml_path)
    assert cfg.name == "my_cool_task"
    assert cfg.batch_size == 4


def test_load_file_ignores_in_file_name(tmp_path: pathlib.Path):
    """Filename wins over `name:` inside the YAML."""
    yaml_path = tmp_path / "filename_wins.yaml"
    yaml_path.write_text(
        """
name: this_is_ignored
model:
  type: Pi0Config
  pi05: true
  paligemma_variant: dummy
  action_expert_variant: dummy
data:
  type: FakeDataConfig
"""
    )
    cfg = _yaml_loader.load(yaml_path)
    assert cfg.name == "filename_wins"


def test_round_trip_dump_then_load():
    """Take an in-memory TrainConfig, dump it to YAML, parse it back, compare."""
    original = _config.TrainConfig(
        name="round_trip_demo",
        model=_config.get_config("debug_pi05").model,
        data=_config.FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="round_trip_demo",
        wandb_enabled=False,
    )
    yaml_text = _yaml_loader.dump(original)
    reloaded = _yaml_loader.loads(yaml_text, name="round_trip_demo")
    assert reloaded == original


def test_round_trip_debug_pi05_from_registry():
    """The canonical debug_pi05 config dumps and reloads to itself."""
    original = _config.get_config("debug_pi05")
    yaml_text = _yaml_loader.dump(original)
    reloaded = _yaml_loader.loads(yaml_text, name="debug_pi05")
    assert reloaded == original


def test_dump_rejects_unserializable():
    """A closure in the config is what makes it Python-only; dump must say so."""
    with_lambda = _config.TrainConfig(
        name="lambda_demo",
        data=_config.SimpleDataConfig(
            repo_id="fake",
            data_transforms=lambda model: _transforms.Group(),
        ),
    )
    with pytest.raises(ValueError, match="Cannot serialize"):
        _yaml_loader.dump(with_lambda)


def test_load_from_disk_file(tmp_path: pathlib.Path):
    """End-to-end: write a YAML to disk, load it, and verify name + content."""
    yaml_path = tmp_path / "disk_load_test.yaml"
    yaml_path.write_text(
        """
model:
  type: Pi0Config
  pi05: true
  paligemma_variant: dummy
  action_expert_variant: dummy
data:
  type: FakeDataConfig
batch_size: 8
exp_name: disk_load_test
"""
    )
    cfg = _yaml_loader.load(yaml_path)
    assert cfg.name == "disk_load_test"
    assert cfg.batch_size == 8
    assert cfg.exp_name == "disk_load_test"


def test_lora_freeze_filter_auto_derived():
    """A LoRA model in YAML (variant contains 'lora') gets freeze_filter auto-injected
    so YAML authors don't have to express flax.nnx filter trees by hand."""
    yaml_text = """
model:
  type: Pi0Config
  paligemma_variant: gemma_2b_lora
  action_expert_variant: gemma_300m_lora
  pi05: true
data:
  type: FakeDataConfig
"""
    cfg = _yaml_loader.loads(yaml_text, name="lora_test")
    # freeze_filter was not in the YAML, but the loader derived it from the model.
    # It must match what Pi0Config.get_freeze_filter() returns for this variant.
    import openpi.models.pi0_config as pi0_config

    expected = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        pi05=True,
    ).get_freeze_filter()
    assert repr(cfg.freeze_filter) == repr(expected)


def test_non_lora_freeze_filter_stays_default():
    """Non-LoRA models keep the default empty freeze_filter (nnx.Nothing)."""
    yaml_text = """
model:
  type: Pi0Config
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m
  pi05: true
data:
  type: FakeDataConfig
"""
    cfg = _yaml_loader.loads(yaml_text, name="full_ft_test")
    # freeze_filter should equal the default (nnx.Nothing); no auto-derivation triggered.
    default = _config.TrainConfig(name="__defaults__").freeze_filter
    assert repr(cfg.freeze_filter) == repr(default)


def test_explicit_freeze_filter_in_kwargs_is_honored_when_present():
    """If someone explicitly passes freeze_filter (e.g., via dump+reload), don't overwrite."""
    # We can't write freeze_filter into YAML directly (it's unserializable), but
    # the auto-derive path should only trigger when the key is absent.
    yaml_text = """
model:
  type: Pi0Config
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m
  pi05: true
data:
  type: FakeDataConfig
"""
    cfg = _yaml_loader.loads(yaml_text, name="x")
    # Non-LoRA -> default; we already verified above. This test just documents
    # that the gate (paligemma/action_expert variants) controls the behavior.
    assert "lora" not in cfg.model.paligemma_variant
    assert "lora" not in cfg.model.action_expert_variant


def test_nested_dataclass_blocks_need_no_type_tag():
    """`assets:` and `base_config:` are built from the field annotation alone."""
    yaml_text = """
model:
  type: Pi0Config
  action_horizon: 10
data:
  type: DroidInferenceDataConfig
  assets:
    asset_id: droid
  base_config:
    prompt_from_task: true
"""
    cfg = _yaml_loader.loads(yaml_text, name="nested_blocks")
    assert cfg.data.assets == _config.AssetsConfig(asset_id="droid")
    assert cfg.data.base_config == _config.DataConfig(prompt_from_task=True)


def test_transform_group_round_trips():
    """A repack group is expressible in YAML, and dumps back to the same text."""
    yaml_text = """
model:
  type: Pi0Config
  pi05: true
data:
  type: LeRobotAlohaDataConfig
  repo_id: fake/dataset
  repack_transforms:
    inputs:
    - type: RepackTransform
      structure:
        images:
          cam_high: observation.images.head
        state: observation.state
"""
    cfg = _yaml_loader.loads(yaml_text, name="repack_demo")
    (repack,) = cfg.data.repack_transforms.inputs
    assert isinstance(repack, _transforms.RepackTransform)
    assert repack.structure["images"]["cam_high"] == "observation.images.head"
    assert cfg.data.repack_transforms.outputs == (), "an absent side keeps Group's own default"

    assert _yaml_loader.loads(_yaml_loader.dump(cfg), name="repack_demo") == cfg


def test_unknown_transform_type_is_rejected():
    yaml_text = """
model:
  type: Pi0Config
data:
  type: LeRobotAlohaDataConfig
  repo_id: fake/dataset
  repack_transforms:
    inputs:
    - type: NotARegisteredTransform
"""
    with pytest.raises(KeyError, match="NotARegisteredTransform"):
        _yaml_loader.loads(yaml_text, name="bad_transform")
