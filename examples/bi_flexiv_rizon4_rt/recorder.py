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
from typing import override

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


def make_recorder_subscriber(
    repo_id: str,
    task: str,
    fps: int = 30,
    root: str | pathlib.Path | None = None,
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
    image_writer_threads: int = 4,
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

    Returns:
        A configured LeRobotRecorderSubscriber ready to attach to Runtime.
    """
    local_dir = pathlib.Path(root) if root else None

    logger.info(f"Creating dataset repo_id={repo_id}" + (f" root={local_dir}" if local_dir else ""))
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=make_bi_flexiv_dataset_features(image_height, image_width, use_videos),
        root=local_dir,
        robot_type="bi_flexiv_rizon4_rt",
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
    )
    return LeRobotRecorderSubscriber(dataset=dataset, task=task)
