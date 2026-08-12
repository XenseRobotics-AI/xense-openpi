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

import pathlib
import shutil

from lerobot.datasets.utils import DEFAULT_FEATURES, DEFAULT_IMAGE_PATH
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client.runtime import subscriber as _subscriber

import examples.bi_flexiv_rizon4_rt.keyboard_control as _keyboard_control

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
    include_intervention: bool = False,
    include_success: bool = False,
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
    if include_intervention:
        features["observation.is_intervention"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_intervention"],
        }
    if include_success:
        features["observation.is_success"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_success"],
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

    def __init__(
        self,
        dataset: LeRobotDataset,
        task: str,
        controller: _keyboard_control.KeyboardEpisodeController | None = None,
        record_intervention: bool = False,
        confirm_success: bool = False,
    ):
        self._dataset = dataset
        self._task = task
        self._controller = controller
        self._record_intervention = record_intervention
        self._confirm_success = confirm_success
        self._step_count = 0

    @override
    def on_episode_start(self) -> None:
        self._step_count = 0
        logger.info("Episode recording started")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        images_raw = observation.get("images_raw", {})

        frame: dict = {
            "observation.state": np.asarray(observation["state"], dtype=np.float32),
            "action": np.asarray(action["actions"], dtype=np.float32),
            "task": self._task,
        }
        if self._record_intervention:
            frame["observation.is_intervention"] = np.asarray(
                [float(action.get("is_intervention", False))],
                dtype=np.float32,
            )
        if self._confirm_success:
            # Placeholder; backfilled with the keyboard-confirmed value in
            # on_episode_end (NaN means "unconfirmed").
            frame["observation.is_success"] = np.asarray([np.nan], dtype=np.float32)
        for cam in _POLICY_CAMERAS:
            if cam in images_raw:
                frame[f"observation.images.{cam}"] = np.asarray(images_raw[cam], dtype=np.uint8)
            else:
                logger.warning(f"Camera '{cam}' missing from images_raw, skipping frame image")

        self._dataset.add_frame(frame)
        self._step_count += 1

    @override
    def on_episode_end(self) -> None:
        if self._step_count == 0:
            logger.warning("Episode ended with 0 steps — not saving")
            return

        end_reason = None
        if self._controller is not None:
            end_reason = self._controller.consume_end_reason()

        if end_reason == _keyboard_control.EpisodeEndReason.DISCARD:
            logger.info(f"Discarding episode ({self._step_count} steps)...")
            self._discard_episode_images()
            self._dataset.clear_episode_buffer()
            return

        if self._confirm_success and "observation.is_success" in self._dataset.features:
            success = None
            if end_reason == _keyboard_control.EpisodeEndReason.SUCCESS:
                success = 1.0
            elif end_reason == _keyboard_control.EpisodeEndReason.FAILURE:
                success = 0.0
            self._backfill_success(success)

        logger.info(f"Saving episode ({self._step_count} steps)...")
        self._dataset.save_episode()
        logger.info(
            f"Episode saved. Total episodes: {self._dataset.meta.total_episodes}, "
            f"total frames: {self._dataset.meta.total_frames}"
        )

    def _discard_episode_images(self) -> None:
        """Remove the discarded episode's temp PNG frames.

        lerobot-xense's ``clear_episode_buffer(delete_images=...)`` only cleans
        features with dtype "image"; for video-typed datasets (``use_videos``)
        ``meta.image_keys`` is empty, so discarded frames would otherwise linger
        as orphaned PNGs (images on disk but no mp4). Remove them explicitly so
        the dataset directory stays consistent.
        """
        buffer = self._dataset.episode_buffer
        if buffer is None:
            return
        episode_index = buffer.get("episode_index")
        if isinstance(episode_index, np.ndarray):
            episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
        if not isinstance(episode_index, int):
            return
        for cam_key in self._dataset.meta.camera_keys:
            fpath = DEFAULT_IMAGE_PATH.format(
                image_key=cam_key, episode_index=episode_index, frame_index=0
            )
            img_dir = (self._dataset.root / fpath).parent
            if img_dir.is_dir():
                shutil.rmtree(img_dir)

    def _backfill_success(self, value: float | None) -> None:
        """Replace the per-frame is_success placeholders with the confirmed value.

        ``value=None`` keeps the NaN placeholders (episode ended via right
        arrow without a success confirmation). Works on the dataset's in-memory
        episode buffer before ``save_episode`` stacks it into parquet; no
        changes to lerobot itself are required.
        """
        buffer = self._dataset.episode_buffer
        if buffer is None or "observation.is_success" not in buffer:
            return
        if value is None:
            return
        fill = np.asarray([float(value)], dtype=np.float32)
        buffer["observation.is_success"] = [fill.copy() for _ in range(buffer["size"])]


def make_recorder_subscriber(
    repo_id: str,
    task: str,
    fps: int = 30,
    root: str | pathlib.Path | None = None,
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
    image_writer_threads: int = 4,
    controller: _keyboard_control.KeyboardEpisodeController | None = None,
    record_intervention: bool = False,
    confirm_success: bool = False,
    resume: bool = False,
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
            robot_type must match. Optional per-frame features
            (is_intervention / is_success) are auto-disabled when the existing
            dataset's schema does not contain them.

    Returns:
        A configured LeRobotRecorderSubscriber ready to attach to Runtime.
    """
    local_dir = pathlib.Path(root) if root else None

    if resume:
        dataset, record_intervention, confirm_success = _open_dataset_for_resume(
            repo_id=repo_id,
            root=local_dir,
            fps=fps,
            image_writer_threads=image_writer_threads,
            use_videos=use_videos,
            image_height=image_height,
            image_width=image_width,
            record_intervention=record_intervention,
            confirm_success=confirm_success,
        )
    else:
        logger.info(f"Creating dataset repo_id={repo_id}" + (f" root={local_dir}" if local_dir else ""))
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=make_bi_flexiv_dataset_features(
                image_height,
                image_width,
                use_videos,
                include_intervention=record_intervention,
                include_success=confirm_success,
            ),
            root=local_dir,
            robot_type="bi_flexiv_rizon4_rt",
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
        )
    return LeRobotRecorderSubscriber(
        dataset=dataset,
        task=task,
        controller=controller,
        record_intervention=record_intervention,
        confirm_success=confirm_success,
    )


def _open_dataset_for_resume(
    repo_id: str,
    root: pathlib.Path | None,
    fps: int,
    image_writer_threads: int,
    use_videos: bool,
    image_height: int,
    image_width: int,
    record_intervention: bool,
    confirm_success: bool,
) -> tuple[LeRobotDataset, bool, bool]:
    """Open an existing dataset for appending new episodes.

    The lerobot-xense fork supports incremental recording by constructing
    ``LeRobotDataset`` directly (there is no ``resume=`` kwarg on ``create()``);
    its ``save_episode`` appends to the latest parquet/video chunk files and
    episode numbering continues from ``meta.total_episodes``.

    Returns the opened dataset plus the adjusted feature flags (derived from
    the dataset's existing schema so recorded frames never violate it).
    """
    if root is None:
        root = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume: no dataset found at {root} (missing {info_path}). "
            "Use --resume only when a dataset already exists at --record_root/--record_repo_id."
        )

    dataset = LeRobotDataset(repo_id=repo_id, root=root, batch_encoding_size=1)
    logger.info(
        f"Resuming dataset repo_id={repo_id} root={root}: "
        f"episodes={dataset.meta.total_episodes}, frames={dataset.meta.total_frames}"
    )

    # Adjust optional per-frame features to the dataset's existing schema.
    if record_intervention and "observation.is_intervention" not in dataset.features:
        logger.warning(
            "Existing dataset has no 'observation.is_intervention' feature; "
            "recording without the intervention flag."
        )
        record_intervention = False
    if confirm_success and "observation.is_success" not in dataset.features:
        logger.warning(
            "Existing dataset has no 'observation.is_success' feature; "
            "--confirm_success will only end episodes (no success column)."
        )
        confirm_success = False

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
        **make_bi_flexiv_dataset_features(
            image_height,
            image_width,
            use_videos,
            include_intervention=record_intervention,
            include_success=confirm_success,
        ),
        **DEFAULT_FEATURES,
    }
    _check_features_match(dataset.features, expected_features)

    if image_writer_threads:
        dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)

    return dataset, record_intervention, confirm_success


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
