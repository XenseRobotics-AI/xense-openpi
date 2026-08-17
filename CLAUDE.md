# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

OpenPI is Physical Intelligence's open-source robotics models repository containing:
- π₀ (pi0): Flow-based vision-language-action model (VLA)
- π₀-FAST: Autoregressive VLA based on FAST action tokenizer
- π₀.₅ (pi05): Upgraded version with better open-world generalization

The codebase supports both JAX and PyTorch implementations, with JAX being the primary framework and PyTorch being newly added.

## ⚠️ GPU Requirements

**This project MUST run on NVIDIA GPU platforms with CUDA support.**

- **Minimum VRAM:** 24GB
- **CUDA Version:** CUDA 12 (JAX requires CUDA 12, PyTorch uses CUDA 12.8)
- **Recommended GPUs:** RTX 4090 (24GB), A100 (80GB), H100
- **CPU-only execution is NOT supported**
- **Supported OS:** Ubuntu 22.04 or later (Linux only)

## Essential Commands

### Development Setup
```bash
# Initial setup (with submodules)
git submodule update --init --recursive

# Environment setup
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### PyTorch Setup (if using PyTorch models)
No extra patching is required. As of `transformers==5.3.0`, Pi0-specific
behaviour lives in `src/openpi/models_pytorch/transformers_compat/` as a set of
small subclasses that are imported directly from `gemma_pytorch.py`. The old
`transformers_replace/` + `cp` workflow has been removed.

### Linting and Formatting
```bash
# Run linter (with fixes)
ruff check . --fix

# Run formatter  
ruff format .
```

### Testing
```bash
# Run tests
pytest

# Run specific test file
pytest src/path/to/test_file.py
```

### Training Workflows

#### JAX Training
```bash
# Compute normalization statistics (required before training)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Train model
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name>

# Train with FSDP (for memory efficiency)
uv run scripts/train.py <config_name> --exp-name=<experiment_name> --fsdp-devices <num_gpus>
```

#### PyTorch Training
```bash
# Single GPU
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Resume training
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --resume
```

### Inference and Serving
```bash
# Serve policy (for inference)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>

# Convert JAX model to PyTorch
uv run examples/convert_jax_model_to_pytorch.py --checkpoint_dir <jax_checkpoint> --config_name <config> --output_path <pytorch_output>
```

## Architecture Overview

### Core Components
- **`src/openpi/models/`**: JAX model implementations (pi0, pi0_fast, gemma, tokenizers)
- **`src/openpi/models_pytorch/`**: PyTorch model implementations and transformers patches
- **`src/openpi/policies/`**: Robot platform-specific policy adapters (DROID, Xense platforms)
- **`src/openpi/training/`**: Training configurations, data processing, and optimization
- **`src/openpi/shared/`**: Shared utilities (download, normalization, transforms)
- **`src/openpi/serving/`**: Policy serving infrastructure (websocket server)

### Key Configuration System
The training system uses a centralized config system in `src/openpi/training/config.py`:
- **DataConfig**: Defines data processing pipelines for different datasets
- **TrainConfig**: Defines training hyperparameters, model selection, and optimization
- **AssetsConfig**: Manages normalization statistics and other training assets
- Configs are YAML files under `configs/`, resolved by filename through `get_config(name)` (e.g., `pi0_droid`, `pi05_droid`); see `configs/README.md`

### Policy Architecture
Each robot platform has a dedicated policy class that handles:
- **Input/Output mappings**: Transform between robot observations/actions and model format
- **Data preprocessing**: Handle camera feeds, proprioception, language prompts
- **Action post-processing**: Convert model outputs to robot-executable actions

### Model Checkpoints
- Base models: `gs://openpi-assets/checkpoints/{pi0_base, pi0_fast_base, pi05_base}`
- Fine-tuned models: `gs://openpi-assets/checkpoints/{pi0_droid, pi05_droid}`
- Auto-downloaded to `~/.cache/openpi` (configurable via `OPENPI_DATA_HOME`)

## Development Patterns

### Adding New Robot Support
1. Create new policy class in `src/openpi/policies/` inheriting from base policy
2. Define input/output mappings for your robot's observation/action space
3. Add data config in `src/openpi/training/config.py` for your dataset format
4. Create training config linking your policy, data config, and base model

### Fine-tuning Workflow
1. Convert your data to LeRobot format (see examples in `examples/droid` and `examples/bi_arx5_real`)
2. Create configs for your dataset following DROID or Xense examples in `config.py`
3. Compute normalization stats: `scripts/compute_norm_stats.py`
4. Train: `scripts/train.py` or `scripts/train_pytorch.py`
5. Serve: `scripts/serve_policy.py`

### JAX vs PyTorch Usage
- **JAX**: Full feature support, mixed precision training, FSDP, LoRA, EMA weights
- **PyTorch**: Limited feature set, no π₀-FAST support, no mixed precision, simpler deployment
- Both frameworks share the same high-level API for inference and policy serving

## Memory Management

### GPU Memory Optimization
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` for JAX training
- Use `--fsdp-devices <n>` for fully-sharded data parallelism
- Consider disabling EMA if running out of memory
- PyTorch uses either full bfloat16 or float32 (controlled by `pytorch_training_precision`)

### Training Requirements
**NVIDIA GPU with CUDA 12 and minimum 24GB VRAM is required.**

- Inference: 24GB+ GPU (RTX 4090)
- LoRA fine-tuning: 24GB+ GPU (RTX 4090)
- Full fine-tuning: 70GB+ GPU (A100 80GB / H100)

## Data Pipeline

### LeRobot Integration
- Uses LeRobot format for training data
- Custom data processors in `src/openpi/training/` handle robot-specific formats
- Normalization statistics are computed per-dataset and stored in checkpoint assets
- Can reload normalization stats from pre-training for related robot platforms