"""Compute normalization statistics for a config.

**LeRobot Hub datasets**

1. **Sync from Hub (default):** ``huggingface_hub.snapshot_download`` mirrors the **full**
   dataset snapshot under ``$HF_LEROBOT_HOME/<org>/<name>/``, including ``videos/`` when
   the repo has them. Use this so training later has a complete local tree.

2. **Compute stats:** only reads ``data/**/*.parquet`` (state / action columns) with
   pyarrow. **No video decode**, no ``LeRobotDataset`` frame pipeline — so norm stats
   stay fast even after a full download.

**Flags:** ``--skip-hub-sync`` if local tree is already complete and you want no Hub
round-trip. ``--hub-data-meta-only`` to sync only ``data/**`` + ``meta/**`` (skips
videos). **Offline:** ``--dataset-root``, ``OPENPI_LEROBOT_DATASET_ROOT``, or
``HF_HUB_OFFLINE=1``.

``--use-torch-loader`` decodes images/video via LeRobot — avoid unless you need it.
"""

import logging
import os
import pathlib

from huggingface_hub import snapshot_download
import numpy as np
import pyarrow.dataset as pad
import tqdm
import tyro

try:
    from lerobot.utils.constants import HF_LEROBOT_HOME as _LEROBOT_HOME
except ImportError:
    _LEROBOT_HOME = None

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

logger = logging.getLogger(__name__)

_STATE_ACTION_TRANSFORM_TYPES = (
    transforms.DeltaActions,
    transforms.SubsampleActions,
)


def _lerobot_home() -> pathlib.Path:
    if _LEROBOT_HOME is not None:
        return pathlib.Path(_LEROBOT_HOME)
    hf_home = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface"))
    return hf_home / "lerobot"


def _legacy_hub_dataset_root(repo_id: str) -> pathlib.Path:
    return _lerobot_home().joinpath(*repo_id.split("/"))


def _has_parquet_under(data_dir: pathlib.Path) -> bool:
    return data_dir.is_dir() and any(data_dir.rglob("*.parquet"))


def _hub_offline_requested() -> bool:
    for key in ("HF_HUB_OFFLINE", "OPENPI_OFFLINE"):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def _is_network_unreachable(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 101:
        return True
    mod = getattr(type(exc), "__module__", "")
    if type(exc).__name__ == "ConnectionError" and "requests" in mod:
        return True
    low = str(exc).lower()
    if "network is unreachable" in low or "failed to establish a new connection" in low:
        return True
    if exc.__cause__ is not None:
        return _is_network_unreachable(exc.__cause__)
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        return _is_network_unreachable(exc.__context__)
    return False


def _validate_lerobot_tree(root: pathlib.Path) -> None:
    info_json = root / "meta" / "info.json"
    data_dir = root / "data"
    if not info_json.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root: missing {info_json}")
    if not _has_parquet_under(data_dir):
        raise FileNotFoundError(f"Not a LeRobot dataset root: no *.parquet under {data_dir}")


def _offline_dataset_help(repo_id: str, default_root: pathlib.Path) -> str:
    return (
        f"No usable local copy for {repo_id!r} and Hugging Face Hub is unreachable from Python.\n"
        f"  Default path (requires meta/info.json and data/**/*.parquet):\n    {default_root}\n"
        "  Check: ls meta/info.json data  (run inside that folder)\n"
        "  If your tree lives elsewhere:  --dataset-root /path/to/dataset  "
        "or OPENPI_LEROBOT_DATASET_ROOT\n"
        "  Override cache root with HF_LEROBOT_HOME (LeRobot) / HF_HOME (Hugging Face).\n"
        "  With a complete local copy only:  --skip-hub-sync  or  HF_HUB_OFFLINE=1\n"
        "  Browser can load huggingface.co but Python fails? Browsers often use a system VPN/proxy\n"
        "  that does not apply to the terminal. In the same shell run:  curl -I https://huggingface.co\n"
        "  If curl fails, set HTTPS_PROXY/HTTP_PROXY (or NO_PROXY) to match your browser/proxy app."
    )


def ensure_lerobot_meta_and_parquet(
    repo_id: str,
    dataset_root: pathlib.Path | None = None,
    *,
    skip_hub_sync: bool = False,
    hub_data_meta_only: bool = False,
) -> pathlib.Path:
    """Ensure local LeRobot tree exists; optionally full-hub snapshot before parquet stats."""
    default_root = _legacy_hub_dataset_root(repo_id)

    if dataset_root is not None:
        root = dataset_root.expanduser().resolve()
        _validate_lerobot_tree(root)
        logger.info("Using explicit dataset root %s (no Hub).", root)
        return root

    info_json = default_root / "meta" / "info.json"
    data_dir = default_root / "data"
    local_ready = info_json.is_file() and _has_parquet_under(data_dir)

    if local_ready and (skip_hub_sync or _hub_offline_requested()):
        logger.info("Using local LeRobot dataset under %s (skip hub sync or offline).", default_root)
        return default_root

    if not local_ready and _hub_offline_requested():
        raise FileNotFoundError(_offline_dataset_help(repo_id, default_root))

    default_root.mkdir(parents=True, exist_ok=True)
    try:
        if hub_data_meta_only:
            logger.info(
                "Hub sync (data+meta only, no videos): %r → %s. Stats still use parquet only.",
                repo_id,
                default_root,
            )
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=None,
                local_dir=str(default_root),
                allow_patterns=["data/**", "meta/**"],
                ignore_patterns=["videos/**"],
            )
        else:
            logger.info(
                "Hub sync (full snapshot, includes videos if any): %r → %s. "
                "Stats use parquet only — no video decode.",
                repo_id,
                default_root,
            )
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=None,
                local_dir=str(default_root),
            )
    except Exception as e:
        if local_ready and _is_network_unreachable(e):
            logger.warning(
                "Hub unreachable; continuing with existing local files under %s (sync skipped).",
                default_root,
            )
            return default_root
        if _is_network_unreachable(e):
            raise RuntimeError(_offline_dataset_help(repo_id, default_root)) from e
        raise

    if not info_json.is_file():
        raise FileNotFoundError(
            f"After snapshot_download, expected {info_json}. " "Check repo_id, HF_TOKEN (private), and network."
        )
    if not _has_parquet_under(data_dir):
        raise FileNotFoundError(f"No parquet files under {data_dir} after download.")
    return default_root


def _extract_parquet_key(repack_inputs: list, dest_key: str) -> str | None:
    for t in repack_inputs:
        if isinstance(t, transforms.RepackTransform):
            structure = t.structure
            if isinstance(structure, dict):
                val = structure.get(dest_key)
                if isinstance(val, str):
                    return val
    return None


def _state_action_transforms(data_transforms_inputs: list) -> list:
    return [t for t in data_transforms_inputs if isinstance(t, _STATE_ACTION_TRANSFORM_TYPES)]


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_parquet_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int = 512,
    max_frames: int | None = None,
    dataset_root: pathlib.Path | None = None,
    skip_hub_sync: bool = False,
    hub_data_meta_only: bool = False,
) -> tuple:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("data_config.repo_id must be set.")

    state_col = _extract_parquet_key(data_config.repack_transforms.inputs, "state")
    action_col = _extract_parquet_key(data_config.repack_transforms.inputs, "actions")
    if state_col is None or action_col is None:
        raise ValueError(
            f"Cannot infer parquet column names from repack_transforms "
            f"(state={state_col!r}, actions={action_col!r}). "
            "Use --use-torch-loader if your repack is non-standard."
        )
    logger.info("Parquet columns: state=%r, action=%r", state_col, action_col)

    root = ensure_lerobot_meta_and_parquet(
        repo_id,
        dataset_root=dataset_root,
        skip_hub_sync=skip_hub_sync,
        hub_data_meta_only=hub_data_meta_only,
    )
    data_dir = root / "data"
    logger.info("Reading parquet under %s …", data_dir)
    ds = pad.dataset(str(data_dir), format="parquet")

    cols = ["episode_index", "frame_index", state_col, action_col]
    table = ds.to_table(columns=cols)
    table = table.sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])

    episode_arr = np.array(table.column("episode_index").to_pylist())
    states_all = np.array(table.column(state_col).to_pylist(), dtype=np.float32)
    actions_all = np.array(table.column(action_col).to_pylist(), dtype=np.float32)

    n_total = len(episode_arr)
    logger.info("Loaded %d frames; iterating episodes …", n_total)

    ep_change = np.flatnonzero(np.diff(episode_arr, prepend=episode_arr[0] - 1))
    ep_starts = ep_change.tolist()
    ep_ends = [*ep_change[1:].tolist(), n_total]

    sa_transforms = _state_action_transforms(data_config.data_transforms.inputs)
    frames_limit = max_frames if max_frames is not None else n_total
    num_batches = min(frames_limit, n_total + batch_size - 1) // batch_size

    def _gen():
        buf_s: list[np.ndarray] = []
        buf_a: list[np.ndarray] = []
        frames_yielded = 0

        for ep_s, ep_e in zip(ep_starts, ep_ends):
            if frames_yielded >= frames_limit:
                break

            ep_len = ep_e - ep_s
            ep_states = states_all[ep_s:ep_e]
            ep_actions_flat = actions_all[ep_s:ep_e]

            row_idx = np.arange(ep_len)[:, None] + np.arange(action_horizon)[None, :]
            row_idx = np.clip(row_idx, 0, ep_len - 1)
            ep_action_seqs = ep_actions_flat[row_idx]

            buf_s.append(ep_states)
            buf_a.append(ep_action_seqs)

            while sum(len(x) for x in buf_s) >= batch_size:
                s = np.concatenate(buf_s, axis=0)
                a = np.concatenate(buf_a, axis=0)

                batch = {"state": s[:batch_size].copy(), "actions": a[:batch_size].copy()}
                for t in sa_transforms:
                    batch = t(batch)
                yield batch
                frames_yielded += batch_size

                rem_s, rem_a = s[batch_size:], a[batch_size:]
                buf_s = [rem_s] if len(rem_s) else []
                buf_a = [rem_a] if len(rem_a) else []

                if frames_yielded >= frames_limit:
                    return

        if buf_s and frames_yielded < frames_limit:
            s = np.concatenate(buf_s, axis=0)
            a = np.concatenate(buf_a, axis=0)
            if len(s):
                batch = {"state": s.copy(), "actions": a.copy()}
                for t in sa_transforms:
                    batch = t(batch)
                yield batch

    return _gen(), num_batches


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    dataset_root: pathlib.Path | None = None,
    skip_hub_sync: bool = False,
    hub_data_meta_only: bool = False,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    logger.warning("Torch loader: slower; may access Hub via LeRobotDataset.")
    ensure_lerobot_meta_and_parquet(
        data_config.repo_id,
        dataset_root=dataset_root,
        skip_hub_sync=skip_hub_sync,
        hub_data_meta_only=hub_data_meta_only,
    )
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(dataset, num_batches=num_batches)
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int = 512,
    use_torch_loader: bool = False,
    dataset_root: pathlib.Path | None = None,
    skip_hub_sync: bool = False,
    hub_data_meta_only: bool = False,
):
    """Compute norm stats. By default performs a **full** Hub snapshot (incl. videos);
    stats are still computed from parquet only (no video decode)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    root_override = dataset_root
    if root_override is None and os.environ.get("OPENPI_LEROBOT_DATASET_ROOT"):
        root_override = pathlib.Path(os.environ["OPENPI_LEROBOT_DATASET_ROOT"])

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    elif use_torch_loader:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
            dataset_root=root_override,
            skip_hub_sync=skip_hub_sync,
            hub_data_meta_only=hub_data_meta_only,
        )
    else:
        data_loader, num_batches = create_parquet_dataloader(
            data_config,
            config.model.action_horizon,
            batch_size=batch_size,
            max_frames=max_frames,
            dataset_root=root_override,
            skip_hub_sync=skip_hub_sync,
            hub_data_meta_only=hub_data_meta_only,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
