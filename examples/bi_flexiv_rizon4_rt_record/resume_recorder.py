"""Recorder factory that can append new episodes to an existing dataset."""

from __future__ import annotations

import pathlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.utils.robot_utils import get_logger

from examples.bi_flexiv_rizon4_rt_record.recorder import (
    LeRobotRecorderSubscriber,
    make_bi_flexiv_dataset_features,
)

logger = get_logger("BiFlexivResumeRecorder")


def _default_dataset_root(repo_id: str) -> pathlib.Path:
    """Return the same default root used by LeRobotDataset."""

    return pathlib.Path(HF_LEROBOT_HOME) / repo_id


def make_recorder_subscriber(
    repo_id: str,
    task: str,
    fps: int = 30,
    root: str | pathlib.Path | None = None,
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
    image_writer_threads: int = 4,
    only_intervention: bool = False,
    resume: bool = False,
) -> LeRobotRecorderSubscriber:
    """Create a recorder, optionally appending episodes to an existing dataset.

    ``resume=True`` only loads a dataset when its local ``meta/info.json`` is
    present.  Otherwise a new dataset is created, which keeps first-time runs
    predictable and avoids silently downloading or replacing a remote dataset.
    """

    local_dir = pathlib.Path(root) if root else _default_dataset_root(repo_id)
    info_path = local_dir / "meta" / "info.json"

    if resume and info_path.is_file():
        logger.info(f"Resuming existing dataset: repo_id={repo_id} root={local_dir}")
        dataset = LeRobotDataset(repo_id=repo_id, root=local_dir, download_videos=False)
        expected_features = make_bi_flexiv_dataset_features(
            image_height=image_height,
            image_width=image_width,
            use_videos=use_videos,
        )
        if dataset.features != expected_features:
            raise ValueError(
                f"Existing dataset features do not match the recorder schema: {local_dir}"
            )
    else:
        if resume:
            logger.info(f"No existing dataset found at {local_dir}; creating a new dataset")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=make_bi_flexiv_dataset_features(image_height, image_width, use_videos),
            root=local_dir,
            robot_type="bi_flexiv_rizon4_rt",
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
        )

    logger.info(
        f"Recorder ready: episodes already stored={dataset.meta.total_episodes}, "
        f"total frames={dataset.meta.total_frames}, only_intervention={only_intervention}"
    )
    return LeRobotRecorderSubscriber(dataset=dataset, task=task, only_intervention=only_intervention)

