"""LeRobot recorder with local dataset append/resume support."""

from __future__ import annotations

import pathlib

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import get_logger

from examples.bi_flexiv_rizon4_rt_record.recorder import (
    LeRobotRecorderSubscriber,
    make_bi_flexiv_dataset_features,
)

logger = get_logger("BiFlexivResumableRecorder")


def _resolve_dataset_root(repo_id: str, root: str | pathlib.Path | None) -> pathlib.Path:
    return pathlib.Path(root) if root else pathlib.Path(HF_LEROBOT_HOME) / repo_id


def _validate_existing_dataset(
    dataset: LeRobotDataset,
    *,
    fps: int,
    image_height: int,
    image_width: int,
    use_videos: bool,
) -> None:
    expected = make_bi_flexiv_dataset_features(image_height, image_width, use_videos)
    mismatched = [key for key, feature in expected.items() if dataset.features.get(key) != feature]
    if mismatched:
        raise ValueError(
            "Existing dataset is incompatible with this recorder. "
            f"Mismatched features: {mismatched}"
        )
    if dataset.fps != fps:
        raise ValueError(f"Existing dataset fps={dataset.fps}, requested recorder fps={fps}")


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
    """Create a new recorder or append episodes to an existing local dataset."""

    local_dir = _resolve_dataset_root(repo_id, root)
    info_path = local_dir / "meta" / "info.json"

    if resume and info_path.is_file():
        logger.info(f"Resuming dataset: repo_id={repo_id}, root={local_dir}")
        dataset = LeRobotDataset(repo_id=repo_id, root=local_dir, download_videos=False)
        _validate_existing_dataset(
            dataset,
            fps=fps,
            image_height=image_height,
            image_width=image_width,
            use_videos=use_videos,
        )
    else:
        if resume:
            logger.info(f"Resume requested but no dataset exists at {local_dir}; creating it")
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
        f"Recorder ready: existing episodes={dataset.meta.total_episodes}, "
        f"existing frames={dataset.meta.total_frames}, "
        f"only_intervention={only_intervention}"
    )
    return LeRobotRecorderSubscriber(dataset=dataset, task=task, only_intervention=only_intervention)

