"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.bi_flexiv_policy as bi_flexiv_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.xense_flare_policy as xense_flare_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    """
    Generic data config for ALOHA-like robot platforms (including BiARX5).

    This config provides a flexible base for dual-arm manipulation robots with similar
    observation/action spaces. Used by Xense BiARX5 and other ALOHA-compatible platforms.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # Adapt to specific platforms
    adapt_to_pi: bool = False
    adapt_to_arx5: bool = False
    # Repack transforms - typically overridden by specific configs
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.use_delta_joint_actions:
            # Standard dual-arm manipulation: 6 joints per arm + 1 gripper per arm = 14 dims
            # Apply delta to joints (first 6 and middle 6), keep grippers absolute (7th and 14th)
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotXenseFlareDataConfig(DataConfigFactory):
    """
    Example data config for custom Xense Flare dataset in LeRobot format.
    """

    use_delta_cartesian_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"observation/wrist_image_left": "observation.images.wrist_cam"},
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[xense_flare_policy.XenseFlareInputs()],
            outputs=[xense_flare_policy.XenseFlareOutputs()],
        )

        if self.use_delta_cartesian_actions:
            delta_action_mask = _transforms.make_bool_mask(9, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotBiFlexivDataConfig(DataConfigFactory):
    """
    Data config for BiFlexiv Rizon4 RT dual-arm robot in LeRobot format.

    State/action format (20D, Cartesian with 6D rotation):
        left_tcp.{x, y, z, r1-r6} (9D) + left_gripper.pos (1D) = 10D
        right_tcp.{x, y, z, r1-r6} (9D) + right_gripper.pos (1D) = 10D

    Cameras: head, left_wrist, right_wrist.
    Compatible with Xense/pack_6_cosmetic_bottles_into_carton and similar bi_flexiv datasets.
    """

    use_delta_cartesian_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms: map dataset column names to policy expected format.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "head": "observation.images.head",
                            "left_wrist": "observation.images.left_wrist",
                            "right_wrist": "observation.images.right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[bi_flexiv_policy.BiFlexivInputs()],
            outputs=[bi_flexiv_policy.BiFlexivOutputs()],
        )

        if self.use_delta_cartesian_actions:
            # Dual-arm Cartesian: 18 TCP dims (left 0-8 + right 9-17, all delta) + 2 gripper dims (absolute)
            # Dataset ordering: [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]
            delta_action_mask = _transforms.make_bool_mask(18, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class DynamicLeRobotBiFlexivDataConfig(LeRobotBiFlexivDataConfig):
    """BiFlexiv config that trains from a dynamic root of local LeRobot datasets."""

    dynamic_root: str = tyro.MISSING

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return super().create(assets_dirs, model_config)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If set for dynamic datasets, refresh the loader every N steps.
    refresh_interval_steps: int | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="tie_shoes_50_episodes_lora_no_adjust_training_time_rtc_0202",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head",
                                "cam_left_wrist": "observation.images.left_wrist",
                                "cam_right_wrist": "observation.images.right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/ubuntu/openpi/checkpoints/tie_shoes_50_episodes_lora_no_adjust_1101/tie_shoes_50_episodes_lora_no_adjust_1103_40000/16000/params"
        ),
        num_train_steps=40_000,  # 20000
        num_workers=2,  # default 2
        fsdp_devices=1,  # refer line 359
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/pack_6_cosmetic_bottles_into_carton",
            use_delta_cartesian_actions=True,
            default_prompt="Pick up six cosmetic bottles one by one and pack them into the carton box. The box is narrow, so align each bottle carefully and insert it precisely.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        batch_size=64,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        num_workers=2,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/assemble_box_with_phone_stand0430_merged",
            use_delta_cartesian_actions=True,
            default_prompt="Assemble the packaging by folding the flat box into shape, placing the metal phone stand inside, and closing the box properly.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/assemble_box_with_phone_stand0410_merged_fixed",
            use_delta_cartesian_actions=True,
            default_prompt="Assemble the packaging by folding the flat box into shape, placing the metal phone stand inside, and closing the box properly.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80_000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head",
                                "cam_left_wrist": "observation.images.left_wrist",
                                "cam_right_wrist": "observation.images.right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        save_interval=2000,
        keep_period=10000,
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0520_a100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/earbud_case_insertion_teleop_0515",
            use_delta_cartesian_actions=True,
            default_prompt="Pick up each earbud case from the left stands, insert the matching earbuds, close the lid, and place the case in the box.",
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        save_interval=5000,
        keep_period=20000,
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("/home/li/hubo/xense-openpi/checkpoints/20000/params"),
        num_train_steps=20000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/shoe_insole_retrieval_and_packing0515",
            use_delta_cartesian_actions=True,
            default_prompt="Open the shoe tongue, take the insole out of the shoe, put the insole back into the shoe, and pack the shoe into the shoebox.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_newbalacne_shoe_insole_retrieval_and_packing_0616_h100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/newbalance_shoe_insole_retrieval_and_packing_0611",
            use_delta_cartesian_actions=True,
            default_prompt="Take the shoe out of the shoebox, open the shoe tongue, remove and reinsert the insole, then place the shoe into the shoebox.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_bag_inspection_0611_h100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/dewu_bag_inspection_0611",
            use_delta_cartesian_actions=True,
            default_prompt="Pick up the bag, open it, inspect its contents, close it, and place it on the opposite side.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_bag_inspection_0611_h100_dagger",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=DynamicLeRobotBiFlexivDataConfig(
            repo_id="Xense/dewu_bag_inspection_0611",
            dynamic_root="/wangsl/data/earbud_case_sequential_insertion_teleop/sc/xense-openpi/dynamic_data/bag_inspection",
            use_delta_cartesian_actions=True,
            default_prompt="Pick up the bag, open it, inspect its contents, close it, and place it on the opposite side.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_optical_module_insertion_0810_rtc_h100_dagger",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=DynamicLeRobotBiFlexivDataConfig(
            repo_id="Xense/optical-module-insertion-0731",
            dynamic_root="/wangsl/data/earbud_case_sequential_insertion_teleop/sc/dynamic_datasets",
            use_delta_cartesian_actions=True,
            default_prompt="Pick up the optical module with the left hand, and insert it into the network port with the right hand.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        save_interval=10000,
        ema_decay=None,
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        num_workers=64,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_base_bi_flexiv_optical_module_insertion_0810_rtc_h100",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
        ),
        data=LeRobotBiFlexivDataConfig(
            repo_id="Xense/optical-module-insertion-0731",
            use_delta_cartesian_actions=True,
            default_prompt="Pick up the optical module with the left hand, and insert it into the network port with the right hand.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        save_interval=10000,
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        num_workers=64,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


# YAML lookup paths, in priority order. First match wins.
# - configs/<name>.yaml         per-user, gitignored, for your in-flight experiments
# - configs/_examples/<name>.yaml  checked into git, canonical examples for the team
_YAML_SEARCH_DIRS: tuple[pathlib.Path, ...] = (
    pathlib.Path("configs"),
    pathlib.Path("configs") / "_examples",
)


def _find_yaml_config(config_name: str) -> pathlib.Path | None:
    """Look for configs/<name>.yaml first, then configs/_examples/<name>.yaml."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    for relative in _YAML_SEARCH_DIRS:
        candidate = repo_root / relative / f"{config_name}.yaml"
        if candidate.is_file():
            return candidate
    return None


def _known_config_names() -> list[str]:
    """All names available via either _CONFIGS or YAML files. For 'did you mean' hints."""
    names = set(_CONFIGS_DICT.keys())
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    for relative in _YAML_SEARCH_DIRS:
        directory = repo_root / relative
        if directory.is_dir():
            names.update(p.stem for p in directory.glob("*.yaml"))
    return sorted(names)


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name.

    Lookup order:
      1. configs/<name>.yaml          (per-user, gitignored)
      2. configs/_examples/<name>.yaml (shared examples in git)
      3. _CONFIGS_DICT[name]           (legacy Python registry)
    """
    yaml_path = _find_yaml_config(config_name)
    if yaml_path is not None:
        import openpi.training.yaml_loader as _yaml_loader

        return _yaml_loader.load(yaml_path)

    if config_name in _CONFIGS_DICT:
        return _CONFIGS_DICT[config_name]

    closest = difflib.get_close_matches(config_name, _known_config_names(), n=1, cutoff=0.0)
    closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
    raise ValueError(f"Config '{config_name}' not found.{closest_str}")
