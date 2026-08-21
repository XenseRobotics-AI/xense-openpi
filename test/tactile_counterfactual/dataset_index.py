"""LeRobot v3 dataset access for the probe.

Wraps ``lerobot.datasets.lerobot_dataset.LeRobotDataset`` exactly the way
``openpi.training.data_loader.create_torch_dataset`` does (same
``delta_timestamps`` from ``DataConfig.action_sequence_keys``, same
``tolerance_s``), applies the *same* repack / data / normalize / model
transforms as training, and builds a one-time ``(episode, frame) -> row``
index so that ``get_sample(episode_index, frame_index)`` is O(1) per read.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from lerobot.datasets import lerobot_dataset
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.transforms import PromptFromLeRobotTask

logger = logging.getLogger("tactile_counterfactual")


def build_ep_frame_to_row(episodes, index) -> tuple[dict[tuple[int, int], int], dict[int, int], dict[int, int]]:
    """Map (episode_index, frame_index) -> dataset row.

    ``index`` is the global frame number; the in-episode frame number is
    ``index - episode_start``. Returns (ep_frame_to_row, ep_start, ep_len).
    """
    eps = np.asarray(episodes)
    idx = np.asarray(index)
    if eps.shape != idx.shape or eps.size == 0:
        raise RuntimeError(f"Unexpected index columns: episodes {eps.shape}, index {idx.shape}")

    ep_start: dict[int, int] = {}
    ep_len: dict[int, int] = {}
    for ep in np.unique(eps):
        indices = idx[eps == ep]
        ep_start[int(ep)] = int(indices.min())
        ep_len[int(ep)] = int(indices.max() - indices.min() + 1)

    ep_frame_to_row: dict[tuple[int, int], int] = {}
    for row, (ep, global_idx) in enumerate(zip(eps.tolist(), idx.tolist())):
        ep_frame_to_row[(int(ep), int(global_idx) - ep_start[int(ep)])] = row
    return ep_frame_to_row, ep_start, ep_len


class ProbeDataset:
    """Random access to transformed LeRobot samples by (episode, frame)."""

    def __init__(
        self,
        repo_id: str,
        data_config: _config.DataConfig,
        action_horizon: int,
        *,
        revision: str | None = None,
        root: str | None = None,
        transform: bool = True,
    ) -> None:
        if repo_id is None or repo_id == "fake":
            raise ValueError(f"ProbeDataset requires a real LeRobot repo_id, got {repo_id!r}")

        self.repo_id = repo_id
        self._data_config = data_config
        self._action_horizon = action_horizon

        # Same construction as openpi.training.data_loader.create_torch_dataset.
        metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
        self.fps = metadata.fps
        delta_timestamps = {
            key: [t / self.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        }
        logger.info(
            "Loading LeRobotDataset %s (fps=%s, delta_timestamps keys=%s)",
            repo_id,
            self.fps,
            list(delta_timestamps),
        )
        self._raw_dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=root,
            revision=revision,
            delta_timestamps=delta_timestamps,
            tolerance_s=1e-2,
        )
        self._dataset = self._raw_dataset

        if data_config.prompt_from_task:
            tasks = _convert_tasks_to_dict(metadata.tasks)
            self._dataset = _data_loader.TransformedDataset(self._dataset, [PromptFromLeRobotTask(tasks)])

        if transform:
            self._dataset = _data_loader.transform_dataset(self._dataset, data_config)

        self._build_index()

    # ------------------------------------------------------------------ #
    # Indexing                                                        #
    # ------------------------------------------------------------------ #

    def _build_index(self) -> None:
        """Build (episode_index, frame_index) -> row and per-episode lengths.

        Uses the HF dataset's ``episode_index`` / ``index`` columns only, so no
        video decoding happens here.
        """
        hf = self._raw_dataset.hf_dataset
        self._ep_frame_to_row, self._ep_start, self._ep_len = build_ep_frame_to_row(hf["episode_index"], hf["index"])
        self.num_episodes = len(self._ep_start)
        self.num_frames = len(self._ep_frame_to_row)
        logger.info("Probe dataset index built: %d episodes, %d frames", self.num_episodes, self.num_frames)

    # ------------------------------------------------------------------ #
    # Sampling                                                           #
    # ------------------------------------------------------------------ #

    def episode_length(self, episode_index: int) -> int:
        return self._ep_len[episode_index]

    def has_sample(self, episode_index: int, frame_index: int) -> bool:
        return (episode_index, frame_index) in self._ep_frame_to_row

    def get_sample(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        """Return the *transformed* sample (Observation dict + actions)."""
        try:
            row = self._ep_frame_to_row[(episode_index, frame_index)]
        except KeyError:
            raise KeyError(
                f"(episode={episode_index}, frame={frame_index}) not in dataset "
                f"(episodes available: {len(self._ep_len)}, "
                f"episode {episode_index} has {self._ep_len.get(episode_index, -1)} frames)"
            ) from None
        sample = self._dataset[row]
        # The action window extends action_horizon frames past the queried
        # frame; near the end of an episode LeRobot pads it.
        if frame_index + self._action_horizon > self._ep_len[episode_index]:
            logger.warning(
                "Sample (episode=%d, frame=%d) is within %d frames of the episode end (%d); "
                "the action window will be padded by LeRobot.",
                episode_index,
                frame_index,
                self._action_horizon,
                self._ep_len[episode_index],
            )
        return sample

    def observation_from_sample(self, sample: dict[str, Any]) -> _model.Observation:
        """Convert a transformed sample dict into a model Observation.

        Mirrors the production policy path (``policy.Policy.infer``), which
        converts every input to ``jnp.asarray`` before calling the sampler.
        """
        observation = _model.Observation.from_dict(sample)
        return jax.tree.map(lambda x: jnp.asarray(x), observation)

    # ------------------------------------------------------------------ #
    # Dataset metadata used by the transforms                            #
    # ------------------------------------------------------------------ #

    @property
    def data_config(self) -> _config.DataConfig:
        return self._data_config


def _convert_tasks_to_dict(tasks):
    """Mirror of openpi.training.data_loader._convert_tasks_to_dict (v3 DataFrame -> dict)."""
    import pandas as pd

    if isinstance(tasks, dict):
        return tasks
    if isinstance(tasks, pd.DataFrame):
        return {int(row["task_index"]): task_name for task_name, row in tasks.iterrows()}
    if hasattr(tasks, "iterrows"):
        return {int(row["task_index"]): task_name for task_name, row in tasks.iterrows()}
    return tasks
