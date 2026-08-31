"""LeRobot-format episode recorder for BiFlexiv Rizon4 RT inference.

Records observations and actions during policy execution in the same format
as the original training dataset (raw state, raw 640x480 HWC images, absolute actions).

Usage:
    from examples.bi_flexiv_rizon4_rt.recorder import make_recorder_subscriber

    subscriber = make_recorder_subscriber(
        repo_id="Xense/my_new_dataset",
        task="pack 6 cosmetic bottles into the carton",
        fps=30,
    )
    runtime = Runtime(..., subscribers=[subscriber])
"""

from contextlib import contextmanager
import os
import pathlib
from typing import override

from lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client.runtime import subscriber as _subscriber

logger = get_logger("BiFlexivRecorder")

# Feature names matching Xense/pack_6_cosmetic_bottles_into_carton and all bi_flexiv datasets.
_STATE_NAMES = [
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "left_gripper.pos",
    "right_gripper.pos",
]

_ACTION_NAMES = _STATE_NAMES  # action space identical to state space

# Policy camera names (order matches BiFlexivInputs.EXPECTED_CAMERAS)
_POLICY_CAMERAS = ("head", "left_wrist", "right_wrist")


def make_bi_flexiv_dataset_features(
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
) -> dict:
    """Build the LeRobot features dict for a bi_flexiv_rizon4_rt dataset."""
    dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(_STATE_NAMES),),
            "names": _STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(_ACTION_NAMES),),
            "names": _ACTION_NAMES,
        },
    }
    for cam in _POLICY_CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": dtype,
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class LeRobotRecorderSubscriber(_subscriber.Subscriber):
    """Records inference episodes to disk in LeRobot dataset format.

    Taps into the Runtime subscriber interface:
    - on_episode_start: resets per-episode state
    - on_step: appends (observation, action) frame to the dataset
    - on_episode_end: flushes episode to disk

    The observation dict is expected to contain:
        "state": np.ndarray (20,) — raw robot state, no normalization
        "images_raw": dict[str, np.ndarray (H, W, C)] — original resolution images

    The action dict is expected to contain:
        "actions": np.ndarray (20,) — absolute action after output transforms
    """

    def __init__(self, dataset: LeRobotDataset, task: str):
        self._dataset = dataset
        self._task = task
        self._step_count = 0
        self._finalized = False

    @override
    def on_episode_start(self) -> None:
        self._step_count = 0
        # Warm up streaming encoders before the first frame so encoder
        # initialization doesn't overrun it. Only available in newer
        # lerobot-xense; older versions auto-start the encoder on the
        # first add_frame instead.
        prepare = getattr(self._dataset, "prepare_episode_recording", None)
        if prepare is not None:
            prepare()
        logger.info("Episode recording started")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        images_raw = observation.get("images_raw", {})

        frame: dict = {
            "observation.state": np.asarray(observation["state"], dtype=np.float32),
            "action": np.asarray(action["actions"], dtype=np.float32),
            "task": self._task,
        }
        for cam in _POLICY_CAMERAS:
            if cam in images_raw:
                frame[f"observation.images.{cam}"] = np.asarray(images_raw[cam], dtype=np.uint8)
            else:
                logger.warn(f"Camera '{cam}' missing from images_raw, skipping frame image")

        self._dataset.add_frame(frame)
        self._step_count += 1

    @override
    def on_episode_end(self) -> None:
        if self._step_count == 0:
            logger.warn("Episode ended with 0 steps — not saving")
            return
        logger.info(f"Saving episode ({self._step_count} steps)...")
        self._dataset.save_episode()
        logger.info(
            f"Episode saved. Total episodes: {self._dataset.meta.total_episodes}, "
            f"total frames: {self._dataset.meta.total_frames}"
        )

    def finalize(self) -> None:
        """Flush any unfinished episode and close LeRobot writers.

        This is intended to run in ``main``'s ``finally`` block *before* the
        robot is disconnected.  It must be safe to call after a normal
        shutdown (where ``on_episode_end`` already saved and cleared the
        buffer) as well as after an unexpected exception in ``runtime.run()``.
        In the latter case the current in-memory episode buffer may still
        contain unsaved frames; save it first so the partial episode is not
        lost, then finalize the parquet writers so the dataset can be resumed
        or loaded from disk later.
        """
        if self._finalized:
            return

        buffer = self._dataset.episode_buffer
        if self._step_count > 0 and buffer is not None and int(buffer.get("size", 0)) > 0:
            logger.warn(
                f"Finalizing recorder with {int(buffer.get('size', 0))} unsaved frames "
                "— saving partial episode."
            )
            try:
                self._dataset.save_episode()
            except Exception as e:
                logger.error(f"Failed to save partial episode during finalize: {e}")

        try:
            self._dataset.finalize()
        except Exception as e:
            logger.error(f"Failed to finalize LeRobotDataset: {e}")

        try:
            self._dataset.stop_image_writer()
        except Exception as e:
            logger.error(f"Failed to stop image writer during finalize: {e}")

        self._finalized = True


def make_recorder_subscriber(
    repo_id: str,
    task: str,
    fps: int = 30,
    root: str | pathlib.Path | None = None,
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
    image_writer_threads: int = 4,
    resume: bool = False,
    vcodec: str = "auto",
    streaming_encoding: bool = True,
) -> LeRobotRecorderSubscriber:
    """Create a LeRobotRecorderSubscriber for a new dataset.

    Args:
        repo_id: HuggingFace dataset repo id (e.g. "Xense/my_dataset").
        task: Language description of the task being recorded.
        fps: Dataset frame rate. Should match the inference runtime_hz.
        root: Local root directory. Defaults to ~/.cache/huggingface/lerobot/<repo_id>.
        image_height: Raw image height in pixels (default 480).
        image_width: Raw image width in pixels (default 640).
        use_videos: Encode images as video (True) or individual frames (False).
        image_writer_threads: Async image writer thread count.
        resume: If True, open the existing dataset at repo_id/root and append
            new episodes to it instead of creating a fresh dataset. Episode
            numbering continues from the existing dataset, and fps/features/
            robot_type must match.
        vcodec: Video codec passed to LeRobotDataset. "auto" resolves to a
            hardware encoder (e.g. h264_nvenc) when available, else libsvtav1.
        streaming_encoding: Encode video frames in background threads during
            recording (near-instant save, smaller files) instead of buffering
            PNGs and encoding after each episode.

    Returns:
        A configured LeRobotRecorderSubscriber ready to attach to Runtime.
    """
    local_dir = pathlib.Path(root) if root else None

    if resume:
        dataset = _open_dataset_for_resume(
            repo_id=repo_id,
            root=local_dir,
            fps=fps,
            image_writer_threads=image_writer_threads,
            use_videos=use_videos,
            image_height=image_height,
            image_width=image_width,
            vcodec=vcodec,
            streaming_encoding=streaming_encoding,
        )
    else:
        logger.info(f"Creating dataset repo_id={repo_id}" + (f" root={local_dir}" if local_dir else ""))
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=make_bi_flexiv_dataset_features(image_height, image_width, use_videos),
            root=local_dir,
            robot_type="bi_flexiv_rizon4_rt",
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            vcodec=vcodec,
            streaming_encoding=streaming_encoding,
        )
    return LeRobotRecorderSubscriber(dataset=dataset, task=task)


def _open_dataset_for_resume(
    repo_id: str,
    root: pathlib.Path | None,
    fps: int,
    image_writer_threads: int,
    use_videos: bool,
    image_height: int,
    image_width: int,
    vcodec: str = "auto",
    streaming_encoding: bool = True,
) -> LeRobotDataset:
    """Open an existing dataset for appending new episodes.

    The lerobot-xense fork supports incremental recording by constructing
    ``LeRobotDataset`` directly (there is no ``resume=`` kwarg on ``create()``);
    its ``save_episode`` appends to the latest parquet/video chunk files and
    episode numbering continues from ``meta.total_episodes``.
    """
    if root is None:
        root = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume: no dataset found at {root} (missing {info_path}). "
            "Use --resume only when a dataset already exists at --record_root/--record_repo_id."
        )

    _validate_local_resume_dataset(root)

    # LeRobotDataset.__init__ falls back to downloading from the Hub whenever
    # its local parquet loader raises FileNotFoundError/NotADirectoryError.
    # For resume we already know the dataset should be local; prevent a
    # missing/incomplete local dataset from turning into a remote fetch (or a
    # confusing "repository not found / connection failed" error).
    with _offline_huggingface():
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            batch_encoding_size=1,
            vcodec=vcodec,
            streaming_encoding=streaming_encoding,
        )
    logger.info(
        f"Resuming dataset repo_id={repo_id} root={root}: "
        f"episodes={dataset.meta.total_episodes}, frames={dataset.meta.total_frames}"
    )

    # Hard compatibility checks: robot type / fps / schema must match.
    if dataset.meta.robot_type != "bi_flexiv_rizon4_rt":
        raise ValueError(
            f"Cannot resume: dataset robot_type is {dataset.meta.robot_type!r}, "
            "expected 'bi_flexiv_rizon4_rt'."
        )
    if dataset.fps != fps:
        raise ValueError(
            f"Cannot resume: dataset fps is {dataset.fps}, requested {fps}. "
            "Use --runtime_hz to match the existing dataset."
        )

    expected_features = {
        **make_bi_flexiv_dataset_features(image_height, image_width, use_videos),
        **DEFAULT_FEATURES,
    }
    _check_features_match(dataset.features, expected_features)

    if image_writer_threads:
        dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)

    return dataset


def _validate_local_resume_dataset(root: pathlib.Path) -> None:
    """Verify all required local LeRobot files are present before resuming.

    This is intentionally stricter than just checking ``meta/info.json``:
    ``LeRobotDatasetMetadata`` also loads tasks/episodes, and
    ``LeRobotDataset`` loads frame parquet files. If any of those are missing
    the normal constructor interprets it as "dataset not cached yet" and
    downloads from Hugging Face, which is exactly what we want to avoid for a
    resume-only workflow.
    """
    required_files = [
        root / INFO_PATH,
        root / DEFAULT_TASKS_PATH,
    ]
    required_parquet_dirs = [
        root / EPISODES_DIR,
        root / "data",
    ]

    missing_files = [str(path.relative_to(root)) for path in required_files if not path.is_file()]
    missing_dirs = []
    for directory in required_parquet_dirs:
        if not any(directory.glob("*/*.parquet")):
            missing_dirs.append(str(directory.relative_to(root)))

    if missing_files or missing_dirs:
        details = []
        if missing_files:
            details.append("missing files: " + ", ".join(missing_files))
        if missing_dirs:
            details.append("missing parquet files under: " + ", ".join(missing_dirs))
        raise FileNotFoundError(
            f"Cannot resume: local dataset at {root} is incomplete ({'; '.join(details)}). "
            "Run the original recording again without --resume, or record into a new dataset."
        )


@contextmanager
def _offline_huggingface():
    """Temporarily force huggingface_hub to stay offline."""
    key = "HF_HUB_OFFLINE"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _check_features_match(existing: dict, expected: dict) -> None:
    """Compare feature schemas, ignoring per-video codec info."""

    def normalize(features: dict) -> dict:
        return {
            name: {k: v for k, v in ft.items() if k != "info"}
            for name, ft in features.items()
        }

    existing_n = normalize(existing)
    expected_n = normalize(expected)
    if existing_n != expected_n:
        raise ValueError(
            "Cannot resume: dataset features do not match this recording setup.\n"
            f"Only in dataset: {sorted(set(existing_n) - set(expected_n))}\n"
            f"Only in this run: {sorted(set(expected_n) - set(existing_n))}\n"
            "Record into a new dataset (drop --resume or use a new --record_repo_id), "
            "or match the original recording options (cameras, image size, "
            "is_success/is_intervention flags)."
        )
