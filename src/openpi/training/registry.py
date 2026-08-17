"""Type registry for YAML config loading.

Maps string type names (as written in YAML) to dataclass types so that
yaml_loader can dispatch on the `type:` field. Add a new class to the
appropriate dict to make it loadable from YAML.

Only YAML-friendly classes are registered here. Classes that take lambdas
or other non-serializable values (e.g. SimpleDataConfig used by pi0_droid)
are intentionally not registered and remain Python-only; see
`config._generated_configs`.
"""

import openpi.models.pi0_config as _pi0_config
import openpi.models.pi0_fast as _pi0_fast
import openpi.models.pi0_tactile_config as _pi0_tactile_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

# Model configs (TrainConfig.model)
MODELS: dict[str, type] = {
    "Pi0Config": _pi0_config.Pi0Config,
    "Pi0FASTConfig": _pi0_fast.Pi0FASTConfig,
    "Pi0TactileConfig": _pi0_tactile_config.Pi0TactileConfig,
}


# Weight loaders (TrainConfig.weight_loader)
WEIGHT_LOADERS: dict[str, type] = {
    "NoOpWeightLoader": _weight_loaders.NoOpWeightLoader,
    "CheckpointWeightLoader": _weight_loaders.CheckpointWeightLoader,
    "PaliGemmaWeightLoader": _weight_loaders.PaliGemmaWeightLoader,
}


# LR schedules (TrainConfig.lr_schedule)
LR_SCHEDULES: dict[str, type] = {
    "CosineDecaySchedule": _optimizer.CosineDecaySchedule,
    "RsqrtDecaySchedule": _optimizer.RsqrtDecaySchedule,
}


# Optimizers (TrainConfig.optimizer)
OPTIMIZERS: dict[str, type] = {
    "AdamW": _optimizer.AdamW,
    "SGD": _optimizer.SGD,
}


# Data transforms that can appear inside a `Group` (e.g. a data config's
# `repack_transforms:`). Only transforms whose fields are plain data belong here:
# RepackTransform carries a nested str->str mapping and nothing else. A transform
# that takes arrays, masks built by helper functions, or a model config is not
# YAML-friendly and stays in Python.
TRANSFORMS: dict[str, type] = {
    "RepackTransform": _transforms.RepackTransform,
}


# Data config factories (TrainConfig.data). Populated lazily to avoid circular import
# with openpi.training.config (which defines these classes).
DATA_CONFIGS: dict[str, type] = {}


def _populate_data_configs() -> None:
    """Lazy import of data config classes to break circular import with config.py."""
    if DATA_CONFIGS:
        return
    import openpi.training.config as _config

    DATA_CONFIGS.update(
        {
            "FakeDataConfig": _config.FakeDataConfig,
            "DroidInferenceDataConfig": _config.DroidInferenceDataConfig,
            "LeRobotAlohaDataConfig": _config.LeRobotAlohaDataConfig,
            "LeRobotDROIDDataConfig": _config.LeRobotDROIDDataConfig,
            "RLDSDroidDataConfig": _config.RLDSDroidDataConfig,
            "LeRobotXenseFlareDataConfig": _config.LeRobotXenseFlareDataConfig,
            "LeRobotBiFlexivDataConfig": _config.LeRobotBiFlexivDataConfig,
            # SimpleDataConfig deliberately omitted: it carries lambdas (data_transforms)
            # that cannot be serialized to YAML.
        }
    )


def resolve(category: dict[str, type], type_name: str) -> type:
    """Look up a class by its registered string name, with a helpful error otherwise."""
    if type_name not in category:
        available = sorted(category.keys())
        raise KeyError(f"Unknown type '{type_name}'. Known: {available}")
    return category[type_name]


def all_registries() -> dict[str, dict[str, type]]:
    """Return all registries by name, for introspection/testing."""
    _populate_data_configs()
    return {
        "MODELS": MODELS,
        "WEIGHT_LOADERS": WEIGHT_LOADERS,
        "LR_SCHEDULES": LR_SCHEDULES,
        "OPTIMIZERS": OPTIMIZERS,
        "DATA_CONFIGS": DATA_CONFIGS,
        "TRANSFORMS": TRANSFORMS,
    }


def class_to_type_name(cls: type) -> str | None:
    """Reverse lookup: given a class, return its registered string name (any registry)."""
    for registry in all_registries().values():
        for name, klass in registry.items():
            if klass is cls:
                return name
    return None


def all_known_classes() -> dict[type, str]:
    """All registered classes mapped to their type-name (for serialization)."""
    return {klass: name for registry in all_registries().values() for name, klass in registry.items()}
