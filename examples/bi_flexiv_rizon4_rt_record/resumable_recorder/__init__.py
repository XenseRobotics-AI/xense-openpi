"""LeRobot recorder with normalized resume-schema validation."""

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

_LEROBOT_METADATA_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def _resolve_dataset_root(repo_id: str, root: str | pathlib.Path | None) -> pathlib.Path:
    return pathlib.Path(root) if root else pathlib.Path(HF_LEROBOT_HOME) / repo_id


def _feature_signature(feature: dict | None) -> tuple | None:
    """Normalize a LeRobot feature to fields required for compatibility.

    Features loaded from disk contain derived video metadata under ``info``
    (codec, pixel format, fps, channels). Fresh recorder feature declarations
    do not. Those values must not make an otherwise identical schema fail the
    resume check.
    """

    if feature is None:
        return None
    names = feature.get("names")
    return (
        feature.get("dtype"),
        tuple(feature.get("shape", ())),
        tuple(names) if names is not None else None,
    )


def _validate_existing_dataset(
    dataset: LeRobotDataset,
    *,
    fps: int,
    image_height: int,
    image_width: int,
    use_videos: bool,
    task: str,
    include_tactile_images: bool,
    tactile_image_height: int,
    tactile_image_width: int,
) -> None:
    expected = make_bi_flexiv_dataset_features(
        image_height,
        image_width,
        use_videos,
        include_tactile_images,
        tactile_image_height,
        tactile_image_width,
    )
    mismatched = [
        key
        for key, feature in expected.items()
        if _feature_signature(dataset.features.get(key)) != _feature_signature(feature)
    ]
    unexpected = sorted(
        set(dataset.features) - set(expected) - _LEROBOT_METADATA_FEATURES
    )
    if mismatched or unexpected:
        details = {
            key: {
                "existing": _feature_signature(dataset.features.get(key)),
                "expected": _feature_signature(expected[key]),
            }
            for key in mismatched
        }
        if unexpected:
            details["unexpected_features"] = unexpected
        raise ValueError(
            "Existing dataset is incompatible with this recorder. "
            f"Mismatched features: {details}"
        )
    if dataset.fps != fps:
        raise ValueError(f"Existing dataset fps={dataset.fps}, requested recorder fps={fps}")
    existing_tasks = set(dataset.meta.tasks.index.tolist())
    if existing_tasks and task not in existing_tasks:
        raise ValueError(
            "Existing dataset task is incompatible with this recorder. "
            f"Existing tasks={sorted(existing_tasks)!r}, requested task={task!r}"
        )


def _resume_vcodec(dataset: LeRobotDataset, requested: str) -> str:
    """Use the codec already present when appending to an older dataset."""

    codecs = {
        feature.get("info", {}).get("video.codec")
        for key, feature in dataset.features.items()
        if key.startswith("observation.images.") and feature.get("info", {}).get("video.codec")
    }
    if len(codecs) != 1:
        return requested
    stored = next(iter(codecs))
    # LeRobot's encoder argument for AV1 is libsvtav1, while metadata stores
    # the short codec name "av1". H.264 uses the same value in both places.
    compatible = {"av1": "libsvtav1", "h264": "h264"}.get(stored)
    if compatible is None or compatible == requested:
        return requested if compatible is None else compatible
    logger.info(f"Resuming with existing video codec {stored} ({compatible})")
    return compatible


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
    include_tactile_images: bool = True,
    tactile_image_height: int = 400,
    tactile_image_width: int = 700,
    vcodec: str = "h264",
) -> LeRobotRecorderSubscriber:
    """Create a new recorder or append episodes to an existing local dataset."""

    local_dir = _resolve_dataset_root(repo_id, root)
    info_path = local_dir / "meta" / "info.json"

    if resume and info_path.is_file():
        logger.info(f"Resuming dataset: repo_id={repo_id}, root={local_dir}")
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=local_dir,
            download_videos=False,
            vcodec=vcodec,
        )
        dataset.vcodec = _resume_vcodec(dataset, vcodec)
        try:
            _validate_existing_dataset(
                dataset,
                fps=fps,
                image_height=image_height,
                image_width=image_width,
                use_videos=use_videos,
                task=task,
                include_tactile_images=include_tactile_images,
                tactile_image_height=tactile_image_height,
                tactile_image_width=tactile_image_width,
            )
        except Exception:
            # Close parquet readers/writers before propagating validation
            # errors, preventing aborts from background/native resources.
            dataset.finalize()
            raise
    else:
        if resume:
            logger.info(f"Resume requested but no dataset exists at {local_dir}; creating it")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=make_bi_flexiv_dataset_features(
                image_height,
                image_width,
                use_videos,
                include_tactile_images,
                tactile_image_height,
                tactile_image_width,
            ),
            root=local_dir,
            robot_type="bi_flexiv_rizon4_rt",
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            vcodec=vcodec,
        )

    logger.info(
        f"Recorder ready: existing episodes={dataset.meta.total_episodes}, "
        f"existing frames={dataset.meta.total_frames}, "
        f"only_intervention={only_intervention}"
    )
    return LeRobotRecorderSubscriber(
        dataset=dataset,
        task=task,
        only_intervention=only_intervention,
        include_tactile_images=include_tactile_images,
    )
