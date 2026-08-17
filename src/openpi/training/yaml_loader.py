"""Load TrainConfig from a YAML file.

Each YAML file is a single, flat representation of one TrainConfig. The
filename's stem (e.g. `pi05_base_xense_open_lock.yaml` -> `pi05_base_xense_open_lock`)
becomes the config name, eliminating the need for a `name:` field inside the file.

Polymorphic fields (model, data, weight_loader, lr_schedule, optimizer) use a
`type:` discriminator that resolves through `openpi.training.registry`.

Example YAML:

    model:
      type: Pi0Config
      pi05: true
      paligemma_variant: gemma_2b
    data:
      type: LeRobotBiFlexivDataConfig
      repo_id: Xense/shoe_insole_retrieval_and_packing0515
      default_prompt: "Open the shoe tongue..."
      prompt_from_task: true
    weight_loader:
      type: CheckpointWeightLoader
      params_path: gs://openpi-assets/checkpoints/pi05_base/params
    batch_size: 256
    num_train_steps: 40000
    fsdp_devices: 8

Nested blocks that are not polymorphic (`assets:`, `base_config:`, a transform
group's `inputs:`/`outputs:`) need no `type:` - the field's own annotation says
which class to build. Transforms inside a group do carry `type:`, resolved
through `registry.TRANSFORMS`.

Limits:
- Lambdas / closures and bare classes (e.g. a tokenizer class passed as a value)
  cannot round-trip through YAML. Classes carrying them - SimpleDataConfig, the
  RoboArena baselines - are deliberately not registered and stay in Python; see
  `config._generated_configs`.
- `exp_name` is left unset in YAML; the user supplies it on the CLI.
- A LoRA `freeze_filter` is neither written nor read: it is re-derived from the
  model's `*_lora` variants on load, and skipped on dump for the same reason.
"""

from __future__ import annotations

import dataclasses
import pathlib
import types
import typing
from typing import Any

from omegaconf import OmegaConf

import openpi.training.registry as _registry
import openpi.transforms as _transforms

_CONFIG_FIELD_TO_REGISTRY: dict[str, str] = {
    "model": "MODELS",
    "data": "DATA_CONFIGS",
    "weight_loader": "WEIGHT_LOADERS",
    "lr_schedule": "LR_SCHEDULES",
    "optimizer": "OPTIMIZERS",
}


def load(yaml_path: pathlib.Path | str) -> TrainConfig:  # noqa: F821  (forward ref)
    """Load a TrainConfig from a YAML file. The file stem becomes the config name."""
    path = pathlib.Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YAML config not found: {path}")

    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}: {path}")

    name = path.stem
    return _build_train_config(name, raw)


def loads(yaml_text: str, name: str) -> TrainConfig:  # noqa: F821
    """Load a TrainConfig from a YAML string. Caller supplies the name."""
    raw = OmegaConf.to_container(OmegaConf.create(yaml_text), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return _build_train_config(name, raw)


def _build_train_config(name: str, raw: dict[str, Any]) -> TrainConfig:  # noqa: F821
    # Import here to avoid circular import (config.py imports registry, registry imports
    # config.py lazily, and yaml_loader is called from config.py).
    import openpi.training.config as _config

    _registry._populate_data_configs()

    kwargs: dict[str, Any] = {"name": name}

    for key, value in raw.items():
        if key == "name":
            # Filename is the source of truth; ignore in-file name to avoid mismatch.
            continue
        if key in _CONFIG_FIELD_TO_REGISTRY:
            kwargs[key] = _build_polymorphic(key, value)
        else:
            kwargs[key] = value

    # Auto-derive freeze_filter for LoRA model variants. Pi0Config.get_freeze_filter()
    # produces the correct flax.nnx filter tree from the variant strings alone, so YAML
    # authors don't need to express filter trees by hand. Only kicks in when:
    #   1) the user didn't specify freeze_filter explicitly, AND
    #   2) the model is Pi0Config with a "lora" substring in either variant.
    _maybe_inject_lora_freeze_filter(kwargs)

    return _config.TrainConfig(**kwargs)


def _maybe_inject_lora_freeze_filter(kwargs: dict[str, Any]) -> None:
    """If the YAML omitted freeze_filter and the model is LoRA, derive it from the model.

    Saves YAML authors from expressing a flax filter tree by hand; `dump` skips the
    field for the same reason, so the round-trip stays symmetric.
    """
    if "freeze_filter" in kwargs:
        return  # user provided one explicitly; respect that
    model = kwargs.get("model")
    if model is None:
        return
    # Only Pi0Config defines get_freeze_filter today.
    if not hasattr(model, "get_freeze_filter"):
        return
    paligemma = getattr(model, "paligemma_variant", "") or ""
    action_expert = getattr(model, "action_expert_variant", "") or ""
    if "lora" not in paligemma and "lora" not in action_expert:
        return
    kwargs["freeze_filter"] = model.get_freeze_filter()


def _build_polymorphic(field_name: str, spec: Any) -> Any:
    """Instantiate the right class for a `type:`-tagged dict."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"Field '{field_name}' must be a mapping with a 'type:' key, got {type(spec).__name__}")
    if "type" not in spec:
        raise ValueError(f"Field '{field_name}' is missing required 'type:' key. Got keys: {sorted(spec.keys())}")

    registry_name = _CONFIG_FIELD_TO_REGISTRY[field_name]
    registry = _registry.all_registries()[registry_name]
    cls = _registry.resolve(registry, spec["type"])

    # Strip the discriminator and pass the rest as kwargs.
    return _construct(cls, {k: v for k, v in spec.items() if k != "type"})


def _construct(cls: type, body: dict[str, Any]) -> Any:
    """Instantiate `cls` from a YAML mapping, building nested dataclasses as it goes.

    A field whose annotation is a dataclass (e.g. `assets: AssetsConfig`,
    `base_config: DataConfig | None`) and whose YAML value is a mapping gets
    constructed recursively, so nested blocks need no `type:` tag - the field's
    own annotation already says which class it is.
    """
    annotations = _field_annotations(cls)
    return cls(**{name: _coerce(annotations.get(name), value) for name, value in body.items()})


def _field_annotations(cls: type) -> dict[str, Any]:
    """Field name -> annotation. Falls back to the raw `f.type` if resolution fails."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:  # unresolvable forward refs shouldn't break loading
        hints = {}
    return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(cls)}


def _nested_dataclass(annotation: Any) -> type | None:
    """The dataclass a field holds, looking through Annotated / Optional / Union.

    Returns None for anything else (scalars, sequences, unresolved string
    annotations) - the loader's signal to pass the YAML value through untouched.
    """
    if annotation is None or isinstance(annotation, str):
        return None
    # tyro.conf.Suppress[X] and friends are Annotated[X, ...].
    if hasattr(annotation, "__metadata__"):
        return _nested_dataclass(annotation.__origin__)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if arg is type(None):
                continue
            if (found := _nested_dataclass(arg)) is not None:
                return found
        return None
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    return None


def _coerce(annotation: Any, value: Any) -> Any:
    """Turn a YAML value into whatever the annotated field expects."""
    if not isinstance(value, dict):
        return value
    target = _nested_dataclass(annotation)
    if target is None:
        return value
    if target is _transforms.Group:
        return _build_group(value)
    return _construct(target, value)


def _build_group(spec: dict[str, Any]) -> _transforms.Group:
    """Build a `transforms.Group` from `{inputs: [...], outputs: [...]}`.

    Only the keys present are passed on: `Group`'s own defaults are empty tuples,
    and passing an empty list instead would break dataclass equality with a config
    that left the side unset.
    """
    if unknown := set(spec) - {"inputs", "outputs"}:
        raise ValueError(f"Transform group has unknown keys {sorted(unknown)}; expected 'inputs' and/or 'outputs'.")
    return _transforms.Group(
        **{key: [_build_transform(item) for item in spec[key]] for key in ("inputs", "outputs") if key in spec}
    )


def _build_transform(spec: Any) -> Any:
    """Build one transform from a `type:`-tagged mapping."""
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"Each transform must be a mapping with a 'type:' key, got {spec!r}")
    cls = _registry.resolve(_registry.TRANSFORMS, spec["type"])
    return _construct(cls, {k: v for k, v in spec.items() if k != "type"})


def dump(config: TrainConfig, yaml_path: pathlib.Path | str | None = None) -> str:  # noqa: F821
    """Serialize a TrainConfig back to YAML. Used by the one-shot migration script.

    Returns the YAML text; if `yaml_path` is given, also writes to disk.
    Raises ValueError if the config contains unserializable fields (lambdas, etc.).
    """
    raw = _train_config_to_dict(config)
    text = OmegaConf.to_yaml(OmegaConf.create(raw), sort_keys=False)
    if yaml_path is not None:
        pathlib.Path(yaml_path).write_text(text)
    return text


def _train_config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a TrainConfig dataclass to a plain dict suitable for YAML.

    Fields that equal the TrainConfig default are skipped (to keep YAML noise low).
    Fields that don't equal the default and aren't serializable raise loudly - that
    config cannot round-trip through YAML and has to be built in Python.
    """
    import openpi.training.config as _config

    known_classes = _registry.all_known_classes()
    defaults = _config.TrainConfig(name="__defaults__")

    import tyro

    out: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        if f.name == "name":
            # Encoded via the YAML filename, not in-file.
            continue
        value = getattr(config, f.name)
        # Skip tyro MISSING sentinels (e.g. exp_name) - user supplies via CLI.
        if value is tyro.MISSING:
            continue
        # A LoRA freeze filter is re-derived from the model variants on load, so it
        # doesn't need to be written - which is just as well, since nnx filter trees
        # don't serialize. This is the dump-side mirror of _maybe_inject_lora_freeze_filter.
        if f.name == "freeze_filter" and _is_derivable_freeze_filter(config, value):
            continue
        default_value = getattr(defaults, f.name, _MISSING)
        # If the value equals the default, skip it - no need to serialize.
        if default_value is not _MISSING and _equal_for_yaml(value, default_value):
            continue
        # Otherwise serialize it; this can raise if value is not YAML-friendly.
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out[f.name] = _dataclass_to_yaml_dict(value, known_classes)
        else:
            out[f.name] = _scalar_to_yaml(value)
    return out


def _is_derivable_freeze_filter(config: Any, value: Any) -> bool:
    """True when `value` is exactly what the loader would rebuild from the model alone."""
    model = getattr(config, "model", None)
    if model is None or not hasattr(model, "get_freeze_filter"):
        return False
    variants = f"{getattr(model, 'paligemma_variant', '') or ''}{getattr(model, 'action_expert_variant', '') or ''}"
    if "lora" not in variants:
        return False
    return _equal_for_yaml(value, model.get_freeze_filter())


class _MissingSentinel:
    pass


_MISSING = _MissingSentinel()


def _equal_for_yaml(a: Any, b: Any) -> bool:
    """Compare two values for 'effectively the same as defaults'.

    Catches uncooperative types (no __eq__ / unhashable) by falling back to repr.
    """
    try:
        return bool(a == b) or (type(a) is type(b) and repr(a) == repr(b))
    except Exception:
        try:
            return repr(a) == repr(b)
        except Exception:
            return False


def _dataclass_to_yaml_dict(obj: Any, known_classes: dict[type, str]) -> dict[str, Any]:
    """Convert a dataclass instance to a {'type': '<name>', **fields} dict.

    Fields that equal the dataclass's own default are skipped to keep YAML compact.
    tyro.MISSING sentinels are skipped (treated as 'unset, user provides on CLI').
    Fields that can't be serialized AND don't equal default raise loudly.
    """
    import tyro

    cls = type(obj)
    type_name = known_classes.get(cls)

    # Build a fresh default instance for diffing field-by-field. If the class
    # requires args (e.g. CheckpointWeightLoader needs params_path), no defaults
    # available - then we serialize everything.
    default_instance: Any | None = None
    try:
        default_instance = cls()
    except TypeError:
        default_instance = None

    body: dict[str, Any] = {}
    if type_name is not None:
        body["type"] = type_name

    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)

        # Skip tyro MISSING sentinels - the user provides these on the CLI.
        if value is tyro.MISSING:
            continue

        # Skip fields equal to their class default (less YAML noise).
        if default_instance is not None:
            default_value = getattr(default_instance, f.name, _MISSING)
            if default_value is not _MISSING and _equal_for_yaml(value, default_value):
                continue

        # Serialize.
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            body[f.name] = _dataclass_to_yaml_dict(value, known_classes)
        else:
            body[f.name] = _scalar_to_yaml(value)
    return body


def _scalar_to_yaml(value: Any) -> Any:
    """Convert leaf values to YAML-friendly Python primitives."""
    import enum

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    # Nested dataclasses (transform groups, the transforms inside them, AssetsConfig,
    # ...) serialize the same way top-level ones do; unregistered classes simply come
    # out without a `type:` key, which is what the loader expects for a field whose
    # annotation already names the class.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_yaml_dict(value, _registry.all_known_classes())
    if isinstance(value, (list, tuple)):
        return [_scalar_to_yaml(v) for v in value]
    if isinstance(value, dict):
        return {k: _scalar_to_yaml(v) for k, v in value.items()}
    if isinstance(value, pathlib.Path):
        return str(value)
    # Anything else (lambdas, transform groups with closures, etc.) is unserializable.
    raise ValueError(
        f"Cannot serialize value of type {type(value).__name__} to YAML: {value!r}. "
        "This config cannot round-trip through YAML and has to be built in Python "
        "(see config._generated_configs)."
    )
