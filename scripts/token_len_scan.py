"""Measure the real tokenized prompt length of a training config's dataset.

For Pi0.5 the language prefix is ``Task: <prompt>, State: <discretized state>;\\nAction: `` and
everything up to ``max_token_len`` is padded (and masked) -- but attention/MLP still run over the
padding. On 2026-09-04 trimming ``max_token_len`` 200 -> 128 gave -6.2% step time with identical
loss (see docs/training-optimization.md). This script runs the *actual* transform chain
(repack -> data transforms -> model transforms) on real states with dummy images and reports the
token-length distribution so ``max_token_len`` can be set with a known margin.

Note: the tokenizer only warns and truncates when the prompt exceeds ``max_token_len``.

Example:
    JAX_PLATFORMS=cpu uv run scripts/token_len_scan.py --config-name pi05_droid --samples 4000
"""

import dataclasses
import glob
import json
import logging
import os
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tyro

import openpi.training.config as _config
import openpi.transforms as _transforms


@dataclasses.dataclass
class Args:
    config_name: str
    # Random states to sample; per-dimension min/max states are always added so the worst case is covered.
    samples: int = 4000
    # Override the LeRobot dataset root (default: $HF_LEROBOT_HOME/<repo_id>).
    dataset_root: str | None = None
    # Image shape handed to the transform chain (channels-first uint8, like LeRobot video frames).
    image_shape: tuple[int, int, int] = (3, 480, 640)
    seed: int = 0


def _lerobot_home() -> pathlib.Path:
    if "HF_LEROBOT_HOME" in os.environ:
        return pathlib.Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    return pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "lerobot"


def _load_tasks(root: pathlib.Path) -> list[str]:
    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.exists():
        return sorted({json.loads(line)["task"] for line in jsonl.read_text().splitlines() if line.strip()})
    parquet = root / "meta" / "tasks.parquet"
    if parquet.exists():
        table = pq.read_table(parquet).to_pandas().reset_index()
        col = "task" if "task" in table.columns else table.columns[0]
        return sorted(set(table[col].astype(str)))
    raise FileNotFoundError(f"No tasks metadata under {root / 'meta'}")


def main(args: Args) -> None:
    cfg = _config.get_config(args.config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_config.repo_id is None:
        raise ValueError("config has no repo_id")
    chain = _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            *data_config.model_transforms.inputs,
        ]
    )
    root = pathlib.Path(args.dataset_root) if args.dataset_root else _lerobot_home() / data_config.repo_id
    tasks = _load_tasks(root)
    logging.info("tasks: %s", tasks)

    # Discover the raw column/key names from the repack transform so this works for any robot config.
    repack = next(t for t in data_config.repack_transforms.inputs if isinstance(t, _transforms.RepackTransform))
    structure = repack.structure
    image_keys = list(structure["images"].values())
    state_key, action_key, prompt_key = structure["state"], structure["actions"], structure["prompt"]

    files = sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))
    states = np.concatenate(
        [np.stack(pq.read_table(f, columns=[state_key]).column(0).to_pylist()) for f in files]
    ).astype(np.float32)
    logging.info("states: %s", states.shape)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(states), size=min(args.samples, len(states)), replace=False)
    idx = np.unique(np.concatenate([idx, states.argmin(0), states.argmax(0)]))

    dummy_img = np.zeros(args.image_shape, np.uint8)
    action_dim = states.shape[1]
    lengths = []
    for i in idx:
        for task in tasks:
            sample = dict.fromkeys(image_keys, dummy_img)
            sample[state_key] = states[i]
            sample[action_key] = np.zeros((cfg.model.action_horizon, action_dim), np.float32)
            sample[prompt_key] = task
            out = chain(sample)
            lengths.append(int(np.asarray(out["tokenized_prompt_mask"]).sum()))
    lengths = np.asarray(lengths)
    cap = cfg.model.max_token_len
    print(
        f"n={len(lengths)} min={lengths.min()} p50={np.percentile(lengths, 50):.0f} "
        f"p99={np.percentile(lengths, 99):.0f} max={lengths.max()} "
        f"(max_token_len={cap}; samples_at_cap={(lengths >= cap).sum()})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
