# openpi - Xense Robotics Fork

> **Note:** This is a fork of [Physical Intelligence's openpi repository](https://github.com/Physical-Intelligence/openpi), adapted and extended for **Xense Robotics platforms** (BiARX5, BiFlexiv and XTac-UMI dual-arm robots).

## 🎯 Our Contributions

This fork focuses on adapting OpenPI models to Xense Robotics platforms with the following key contributions:

- **Xense Platform Support**: Complete integration for BiARX5, BiFlexiv and XTac-UMI dual-arm robot platforms
- **Custom Training Configurations**: Fine-tuned configs for various manipulation tasks (tie shoes, pick-and-place, open lock, wipe vase, etc.)
- **Platform-Specific Policies**: `bi_flexiv_policy.py`, `xtac_umi_policy.py` and optimized data processing pipelines for Xense robots
- **Real-World Deployment**: Production-ready inference and training commands for Xense platforms
- **Streamlined Codebase**: Removed ALOHA and LIBERO dependencies to focus on DROID and Xense platforms

---

## About OpenPI

openpi holds open-source models and packages for robotics, originally published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

This repository contains three types of models:

- **[π₀ model](https://www.physicalintelligence.company/blog/pi0)**: A flow-based vision-language-action model (VLA)
- **[π₀-FAST model](https://www.physicalintelligence.company/research/fast)**: An autoregressive VLA based on the FAST action tokenizer
- **[π₀.₅ model](https://www.physicalintelligence.company/blog/pi05)**: An upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation)

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

## Requirements

**⚠️ IMPORTANT: This project requires NVIDIA GPU with CUDA support. CPU-only execution is not supported.**

### Hardware Requirements

**GPU:** NVIDIA GPU with CUDA 12 support and **minimum 24GB VRAM** is required for training and inference. The models use JAX with CUDA 12 and PyTorch with CUDA 12.8.

| Mode               | Minimum VRAM | Recommended GPU    |
| ------------------ | ------------ | ------------------ |
| Inference          | 24 GB        | RTX 4090 (24GB)    |
| Fine-Tuning (LoRA) | 24 GB        | RTX 4090 (24GB)    |
| Fine-Tuning (Full) | 70+ GB       | A100 (80GB) / H100 |

**Note:** These estimations assume a single GPU. You can use multiple GPUs with model parallelism (FSDP) to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Multi-node training is not currently supported.

**Operating System:** Ubuntu 22.04 or later (other Linux distributions may work but are not officially tested). Windows and macOS are not supported.

## Installation

We build on the `lerobot-xense` mamba environment, then install the client package followed by the main openpi package.

```bash
# Clone with submodules
git clone git@github.com:XenseRobotics-AI/xense-openpi.git openpi
cd openpi

# Activate the base environment
mamba activate lerobot-xense

# 1. Install the client package (xense-client, used by robot runtimes)
cd packages/xense-client
GIT_LFS_SKIP_SMUDGE=1 pip install -e .

# 2. Install the main openpi package
cd ../..
GIT_LFS_SKIP_SMUDGE=1 pip install -e .
```

## Model Checkpoints

### Base Models

We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                          | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$  | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning                       | `gs://openpi-assets/checkpoints/pi05_base`     |

### Fine-Tuned Models

We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with the DROID Franka setup, they might not generalize to your particular setup, though we found the DROID checkpoint to generalize quite broadly in practice.

| Model              | Use Case                | Description                                                                                                                                                                                                                           | Checkpoint Path                                 |
| ------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| $\pi_0$-FAST-DROID | Inference               | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform                              | `gs://openpi-assets/checkpoints/pi0_fast_droid` |
| $\pi_0$-DROID      | Fine-Tuning             | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                                             | `gs://openpi-assets/checkpoints/pi0_droid`      |
| $\pi_{0.5}$-DROID  | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`     |

By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.

## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```

You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: The client package (`xense-client`) provides a websocket-based policy client so that model inference can run on a more powerful remote GPU server while the robot runtime streams observations and receives action chunks in real time. See `examples/simple_client/` for a minimal reference implementation.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.

## Fine-Tuning Base Models on Your Own Data

We will explain how to fine-tune a base model on your own data using examples from the DROID dataset and Xense platform. We will explain three steps:

1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

For training, we use the LeRobot dataset format. You can convert your own data to this format by following the LeRobot documentation. We provide example conversion scripts for the DROID dataset in the [`examples/droid`](examples/droid) directory.

### 2. Defining training configs and running training

A training config is **one YAML file per task**, resolved by name through
`get_config(name)`. Per-user configs in `configs/<name>.yaml` are gitignored;
shared templates live in `configs/_examples/<name>.yaml`. See
[`configs/README.md`](configs/README.md) for the schema and
[`docs/yaml_config_changelog.md`](docs/yaml_config_changelog.md) for design notes.

Lookup order: `configs/<name>.yaml` → `configs/_examples/<name>.yaml` →
generated configs. First match wins. The only configs still built in Python are
the RoboArena baselines (`paligemma_*_droid`), which pass tokenizer classes and
lambdas that can't be serialized; see `config._generated_configs`.

Building blocks:

- Data transforms: Define the data mapping from your environment to the model (see [`droid_policy.py`](src/openpi/policies/droid_policy.py) or [`xtac_umi_policy.py`](src/openpi/policies/xtac_umi_policy.py) for examples)
- `DataConfig`: Defines how to process raw data from LeRobot dataset for training
- `TrainConfig`: Defines fine-tuning hyperparameters, data config, and weight loader

#### Defining a new training config in YAML

Copy the closest example from `configs/_examples/` and edit it. For a personal
in-flight experiment, put the new file directly under `configs/` (it will be
gitignored). For something you want to share with the team, put it under
`configs/_examples/` and open a PR.

**Reference:** [`configs/_examples/_FULL_REFERENCE.yaml`](configs/_examples/_FULL_REFERENCE.yaml)
lists every `TrainConfig` field, every registered class for each polymorphic
slot (`model.type`, `data.type`, `weight_loader.type`, etc.), and the current
default for each value. Open it alongside your real config when you need to
look up "what does field X do" or "what else can I put in `data.type`".

**LoRA fine-tuning:** set `model.paligemma_variant` and/or
`model.action_expert_variant` to a `*_lora` value (e.g. `gemma_2b_lora`,
`gemma_300m_lora`). The YAML loader detects the `lora` substring and
auto-derives the correct `freeze_filter` from
`Pi0Config.get_freeze_filter()` — you don't need to (and can't) write a
flax filter tree in YAML. See
[`pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora.yaml`](configs/_examples/pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora.yaml)
for a complete LoRA example.

```yaml
# configs/my_task.yaml  (filename stem = config name; do not put `name:` inside)

model:
  type: Pi0Config # registered in src/openpi/training/registry.py
  pi05: true
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m
  enable_training_time_rtc: true
  # Required when launching with the cuDNN 9.14 setup shown below.
  use_cudnn_attention: true
  max_delay: 10

data:
  type: LeRobotBiFlexivDataConfig
  repo_id: Xense/<your_dataset>
  use_delta_cartesian_actions: true
  default_prompt: "Describe the task here."
  base_config:
    prompt_from_task: true

weight_loader:
  type: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi05_base/params

batch_size: 256
num_train_steps: 40000
num_workers: 64
fsdp_devices: 8
```

Then use it exactly like any other config:

```bash
python scripts/compute_norm_stats.py --config-name my_task
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py my_task \
    --exp-name=my_exp \
    --overwrite
python scripts/serve_policy.py policy:checkpoint --policy.config=my_task --policy.dir=checkpoints/my_task/my_exp/<step>
```

The training command above selects cuDNN from the currently active Conda
environment without hard-coding a machine-specific path. It must be paired
with the following model setting (already shown in the example config):

```yaml
model:
  use_cudnn_attention: true
```

At startup, `scripts/train.py` logs `JAX cuDNN runtime version: <version>`;
verify that it reports `91400`. Use `--overwrite` only for a new run that may
replace an existing experiment directory; use `--resume` to preserve and
continue an existing run.

⚠️ Before committing a YAML into `configs/_examples/`, scrub any machine-local
absolute paths from `weight_loader.params_path` (e.g. `/home/<you>/...`). Use
upstream URLs (`gs://openpi-assets/...`) or paths that every contributor can
resolve.

If you need to add a brand-new model class or data factory, register its string
name in [`src/openpi/training/registry.py`](src/openpi/training/registry.py)
first — then any YAML can reference it via `type: <YourClass>`.

#### Checking a config you just wrote

```bash
pytest src/openpi/training/config_yaml_test.py   # parses, and carries no machine-local paths
python scripts/train.py --help                   # your config name should appear in the list
```

To turn a config that only exists in Python (a RoboArena baseline, or one you
assembled in a REPL) into a YAML file:

```bash
python scripts/dump_config_to_yaml.py <name> --output-dir configs
```

#### Running training

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config (e.g., for Xense):

```bash
python scripts/compute_norm_stats.py --config-name pi05_base_xtac_umi_pick_up_cube_0807_h200
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py pi05_base_xtac_umi_pick_up_cube_0807_h200 \
    --exp-name=my_experiment \
    --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for _reloading_ normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from your robot runtime. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_base_xtac_umi_pick_up_cube_0807_h200 --policy.dir=checkpoints/pi05_base_xtac_umi_pick_up_cube_0807_h200/my_experiment/19999
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For more detailed examples of running inference on specific platforms, see:

- [DROID README](examples/droid/README.md) for DROID platform
- [BiARX5 README](examples/bi_arx5_real/README.md) for Xense platform

If you want to embed a policy server call in your own robot runtime, take a look at the `xense-client` package at `packages/xense-client/` and the minimal reference in `examples/simple_client/`.

### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the DROID benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup

`pip install -e .` is sufficient. The PyTorch path uses `transformers>=5.5,<5.6`; Pi0-specific behavior (AdaRMS, activation precision, read-only KV cache) lives in `src/openpi/models_pytorch/transformers_compat/` and is imported directly by `gemma_pytorch.py`. The old `transformers_replace/` copy-into-site-packages workflow has been removed.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:

   ```bash
   python examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
python scripts/train_pytorch.py debug --exp_name pytorch_test
python scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example using debug config:
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**

1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**

1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo.

| Issue                                  | Resolution                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dependency conflicts                   | Make sure you are in the `lerobot-xense` mamba environment, then run `GIT_LFS_SKIP_SMUDGE=1 pip install -e .` to install openpi.                                                                                                                                                                                                                                                                                                                                                                                    |
| Training runs out of GPU memory        | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may way to consider disabling EMA. |
| Policy server connection errors        | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                                                                                                                                                                                                                                                                                                                                                   |
| Missing norm stats error when training | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Dataset download fails                 | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                                                                                                                                                                                                                                                                                                                                        |
| CUDA/GPU errors                        | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via pip as part of openpi's dependencies. You may even want to try _uninstalling_ system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts.                                                                                                  |
| Import errors when running examples    | Make sure you've installed all dependencies with `pip install -e .`. Some examples may have additional requirements listed in their READMEs.                                                                                                                                                                                                                                                                                                                                                                        |
| Action dimensions mismatch             | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                                                                                                                                                                                                                                                                                                                                         |
| Diverging training loss                | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround.                                                                                                                                                                                                                   |

---

## 🤖 Xense Platform Training & Deployment

This section contains production-ready commands for training and deploying models on Xense Robotics platforms.

### Platform Overview

- **BiARX5**: Bi-manual ARX-5 robot setup with parallel grippers
- **BiFlexiv**: Dual-arm Flexiv Rizon4 real-time setup

### Environment Variables (optional, for multi-GPU / offline datasets)

```bash
# Offline mode for locally-cached HuggingFace datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# NCCL settings for multi-GPU training on hosts without NVLink/IB
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Training Commands (latest per platform)

#### BiARX5 — training-time RTC

```bash
python scripts/compute_norm_stats.py --config-name tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100 \
    --exp-name=tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100 --overwrite
```

#### BiFlexiv — assemble box with phone stand

```bash
python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100 \
    --exp-name=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100_0422 --overwrite

python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100 \
    --exp-name=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100_0430 --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100 \
    --exp-name=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100_0510 --overwrite
```

#### BiFlexiv - shoe_insole_retrieval_and_packing_0

```bash
python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100 \
    --exp-name=pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100_0516 --overwrite

python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100 \
    --exp-name=pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100_0519 --overwrite
```

#### BiFlexiv - shoe_insole_retrieval_and_packing_1

```bash
python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100 \
    --exp-name=pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100_0626 --overwrite
```

#### BiFlexiv - bag_inspection_0611

```bash
python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_bag_inspection_0611_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_bag_inspection_0611_h100 \
    --exp-name=pi05_base_bi_flexiv_bag_inspection_0611_h100_0625 --overwrite
```

### Deployment Commands (latest per platform)

#### BiFlexiv — assemble box inference

```bash
python scripts/serve_policy.py \
    --default-prompt="assemble the box with the phone stand" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100_0422/79999

python scripts/serve_policy.py \
    --default-prompt="assemble the box with the phone stand" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100_0430/79999

python scripts/serve_policy.py \
    --default-prompt="assemble the box with the phone stand" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0430_merged_fixed_h100_0510/66000
```

#### BiFlexiv - shoe_insole_retrieval_and_packing_0 inference

```bash
python scripts/serve_policy.py \
    --default-prompt="retrieve the shoe insole from the box and pack it into the shoe" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100/pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100_0519/39999
```

```bash
python scripts/serve_policy.py \
    --default-prompt="retrieve the shoe insole from the box and pack it into the shoe" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100/pi05_base_bi_flexiv_shoe_insole_retrieval_and_packing_0515_h100_0520/39999
```

#### BiFlexiv - newbalance_shoe_insole_retrieval_and_packing_0606 inference

```bash
python scripts/serve_policy.py \
    --default-prompt="Take the shoe out of the shoebox, open the shoe tongue, remove and reinsert the insole, then place the shoe into the shoebox." \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100/pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100_0701/77000
```

#### BiFlexiv - bag_inspection_0611 inference

```bash
python scripts/serve_policy.py \
    --default-prompt="Pick up the bag, open it, inspect its contents, close it, and place it on the opposite side." \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_bag_inspection_0611_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_bag_inspection_0611_h100/pi05_base_bi_flexiv_bag_inspection_0611_h100_0611/59999
```

### Running the robot client

A launch is described by two YAML files, so the command line stays short:

- **Recipe** — the bench: arm SNs, start/home poses, cameras, gripper block.
  `examples/<example>/recipes/`, selected with `--args.robot-recipe`.
- **Run** — everything else: policy server, RTC, recording, obs streaming,
  control-loop tuning. `examples/<example>/runs/`, selected with `--args.run`.

```bash
# One line: the run file names the bench and presets every flag.
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole

# Any flag still overrides the file, for a one-off.
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole --args.dry-run

# BiARX5 with tactile sensors.
python -m examples.bi_arx5_real.main --args.run tactile --args.host 192.168.2.215
```

Precedence is `dataclass defaults < run YAML < CLI flags`, and `--help` shows the
values actually in effect once a run file is applied. Every flag still works on
its own, exactly as before — a run file is optional:

```bash
# BiFlexiv RT with RTC enabled, all on the CLI. --args.robot-recipe names the
# physical bench; it replaced --args.bi-mount-type, which indexed a stations/
# table lerobot removed.
python -m examples.bi_flexiv_rizon4_rt.main \
    --args.robot-recipe forward-04 \
    --args.host 192.168.142.158 \
    --args.port 8000 \
    --args.inner-control-hz 1000 \
    --args.interpolate-cmds \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run
```

See [`examples/bi_flexiv_rizon4_rt/runs/README.md`](examples/bi_flexiv_rizon4_rt/runs/README.md)
for how to write one.

### BiFlexiv RT forward mount + dewu video switch — full 3-machine demo

The shoe-insole packing demo runs across three machines: the **inference server**
(RTX 5090 GPU, runs the VLA), the **screen PC** (video-playback laptop — detection

- seamless scene-video switching + browser display), and the **robot host**
  (Flexiv Rizon4 ×2 control loop, which forwards the head camera + state to the
  screen PC). Start them in this order; see
  [`examples/dewu_video_switch/README.md`](examples/dewu_video_switch/README.md)
  for the full architecture.

**① Inference server — run on the server side**

```bash
python scripts/serve_policy.py \
    --default-prompt="Take the shoe out of the shoebox, open the shoe tongue, remove and reinsert the insole, then place the shoe into the shoebox." \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100/pi05_base_bi_flexiv_newbalance_shoe_insole_retrieval_and_packing_0616_h100_0701/77000
```

**② Screen PC (video-playback laptop) — run on the screen pc side**

```bash
# Detection + seamless video player + switch ws
# (serves web/ on :8080, switch ws :9101, obs ws :9100)
python -m examples.dewu_video_switch.app \
    --detector shoe_sm \
    --detector-config examples/dewu_video_switch/shoe_sm.json \
    --save-blue-frames

# then open the display in a browser:  http://<screen-pc-ip>:8080
```

**③ Robot host — run on the robot side**

```bash
# --args.subscribe / --args.subscribe-url wire the robot's head camera + state to
# the screen PC (②) so the on-screen scene video switches with the real inspection.
# (Note: --args.subscribe is the detection-data stream — unrelated to the
# --args.robot-recipe bench selection below.)
# --args.subscribe BLOCKS at startup until ② is reachable (handshake, like the
# VLA client waits for the inference server), so start ② before this. Add
# --args.subscribe-handshake-timeout <s> to abort instead of waiting forever.
#
# All of that is preset in examples/bi_flexiv_rizon4_rt/runs/dewu-shoe-insole.yaml;
# edit the host and subscribe-url in that file to match your two machines.
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole

# First time on a new bench, look before you leap:
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole --args.dry-run
```

Keep a log of a run you intend to debug afterwards:

```bash
python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole \
    2>&1 | tee ~/rt_diag_$(date +%F_%H%M).log
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repository is based on the original [OpenPI](https://github.com/Physical-Intelligence/openpi) project by Physical Intelligence. We thank the Physical Intelligence team for open-sourcing their excellent work on vision-language-action models.
