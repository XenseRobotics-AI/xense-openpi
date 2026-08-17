"""Non-blocking LeRobot recorder for inference and intervention segments.

The runtime subscriber thread only copies frames into a FIFO queue. A dedicated
worker owns all LeRobotDataset mutation, image writes, video encoding, and
episode saving, so releasing a Pico4 takeover never blocks robot command
delivery.
"""

from __future__ import annotations

import atexit
import pathlib
import queue
import threading
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client.runtime import subscriber as _subscriber

logger = get_logger("BiFlexivRecorder")

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
_ACTION_NAMES = _STATE_NAMES
_POLICY_CAMERAS = ("head", "left_wrist", "right_wrist")
_TACTILE_CAMERAS = (
    "left_tactile_0",
    "left_tactile_1",
    "right_tactile_0",
    "right_tactile_1",
)

_FRAME = "frame"
_END_SEGMENT = "end_segment"
_STOP = "stop"


def _install_spdlog_warning_compatibility() -> None:
    """Give the robot reset module a ``warning`` alias when using spdlog."""

    try:
        import examples.bi_flexiv_rizon4_rt.real_env as robot_real_env

        robot_logger = robot_real_env.logger
        if hasattr(robot_logger, "warning") or not hasattr(robot_logger, "warn"):
            return

        class LoggerAdapter:
            def __init__(self, inner) -> None:
                self._inner = inner

            def warning(self, message: str) -> Any:
                return self._inner.warn(message)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        robot_real_env.logger = LoggerAdapter(robot_logger)
    except Exception:
        # Logging compatibility must never prevent the recorder from loading.
        pass


_install_spdlog_warning_compatibility()


def make_bi_flexiv_dataset_features(
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
    include_tactile_images: bool = True,
    tactile_image_height: int = 400,
    tactile_image_width: int = 700,
) -> dict:
    """Build the LeRobot features for BiFlexiv Rizon4 RT recordings."""

    dtype = "video" if use_videos else "image"
    features = {
        "action": {
            "dtype": "float32",
            "shape": (len(_ACTION_NAMES),),
            "names": _ACTION_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(_STATE_NAMES),),
            "names": _STATE_NAMES,
        },
    }
    for camera in _POLICY_CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": dtype,
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        }
    if include_tactile_images:
        for camera in _TACTILE_CAMERAS:
            features[f"observation.images.{camera}"] = {
                "dtype": dtype,
                "shape": (tactile_image_height, tactile_image_width, 3),
                "names": ["height", "width", "channels"],
            }
    return features


class _DatasetFinalizeProxy:
    """Delegate dataset access while routing finalize through the worker."""

    def __init__(self, recorder: LeRobotRecorderSubscriber, dataset: LeRobotDataset) -> None:
        self._recorder = recorder
        self._dataset = dataset

    def finalize(self) -> None:
        self._recorder.finalize()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


class LeRobotRecorderSubscriber(_subscriber.Subscriber):
    """Queue recording work so save/encoding never blocks robot control.

    In ``only_intervention`` mode, every contiguous run of actions carrying
    ``is_intervention=True`` becomes an independent LeRobot episode. Policy
    actions between those runs are ignored, while the runtime episode itself
    continues normally.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        task: str,
        only_intervention: bool = False,
        include_tactile_images: bool = True,
    ):
        self._writer_dataset = dataset
        self._dataset = _DatasetFinalizeProxy(self, dataset)
        self._task = task
        self._only_intervention = only_intervention
        self._record_cameras = _POLICY_CAMERAS + (_TACTILE_CAMERAS if include_tactile_images else ())
        self._segment_active = False
        self._episode_has_frames = False
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._finalize_lock = threading.Lock()
        self._finalized = False
        self._worker_error: Exception | None = None
        self._worker = threading.Thread(target=self._worker_loop, name="lerobot-recorder", daemon=True)
        self._worker.start()
        atexit.register(self.finalize)

    @override
    def on_episode_start(self) -> None:
        self._segment_active = False
        self._episode_has_frames = False
        if self._only_intervention:
            logger.info("Intervention recording armed; policy frames will be skipped")
        else:
            logger.info("Episode recording started")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        if self._worker_error is not None:
            return

        is_intervention = bool(action.get("is_intervention", False))
        if self._only_intervention and not is_intervention:
            if self._segment_active:
                self._events.put((_END_SEGMENT, "intervention released"))
                self._segment_active = False
                logger.info("Intervention segment queued for background save; inference continues")
            return

        if self._only_intervention and not self._segment_active:
            self._segment_active = True
            logger.info("Intervention segment recording started")

        frame = self._copy_frame(observation, action)
        if frame is None:
            return
        self._events.put((_FRAME, frame))
        self._episode_has_frames = True

    @override
    def on_episode_end(self) -> None:
        if self._only_intervention:
            if self._segment_active:
                self._events.put((_END_SEGMENT, "runtime episode ended during intervention"))
                self._segment_active = False
            else:
                logger.info("Runtime episode ended; no active intervention segment to close")
        elif self._episode_has_frames:
            self._events.put((_END_SEGMENT, "runtime episode ended"))
        else:
            logger.warn("Runtime episode ended with 0 queued recording frames")
        self._episode_has_frames = False

    def _copy_frame(self, observation: dict, action: dict) -> dict | None:
        images_raw = observation.get("images_raw", {})
        missing = [camera for camera in self._record_cameras if camera not in images_raw]
        if missing:
            logger.warn(f"Skipping recording frame; missing cameras: {missing}")
            return None

        frame: dict = {
            "observation.state": np.array(observation["state"], dtype=np.float32, copy=True),
            "action": np.array(action["actions"], dtype=np.float32, copy=True),
            "task": self._task,
        }
        for camera in self._record_cameras:
            frame[f"observation.images.{camera}"] = np.array(
                images_raw[camera], dtype=np.uint8, copy=True
            )
        return frame

    def _worker_loop(self) -> None:
        frames_in_segment = 0
        try:
            while True:
                event, payload = self._events.get()
                if event == _FRAME:
                    self._writer_dataset.add_frame(payload)
                    frames_in_segment += 1
                elif event == _END_SEGMENT:
                    frames_in_segment = self._save_segment(frames_in_segment, str(payload))
                elif event == _STOP:
                    if frames_in_segment > 0:
                        self._save_segment(frames_in_segment, "recorder finalization")
                    return
        except Exception as error:
            self._worker_error = error
            logger.error(f"Recorder background worker failed: {error}")

    def _save_segment(self, frame_count: int, reason: str) -> int:
        if frame_count == 0:
            return 0
        logger.info(f"Background-saving intervention segment ({frame_count} frames, {reason})...")
        # Encoding cameras sequentially avoids a burst of CPU-heavy encoder
        # processes competing with the robot's real-time command thread.
        self._writer_dataset.save_episode(parallel_encoding=False)
        logger.info(
            f"Segment saved. Total episodes: {self._writer_dataset.meta.total_episodes}, "
            f"total frames: {self._writer_dataset.meta.total_frames}"
        )
        return 0

    def finalize(self) -> None:
        """Drain queued segments, close LeRobot writers, and return safely."""

        with self._finalize_lock:
            if self._finalized:
                return
            self._finalized = True
            self._events.put((_STOP, None))
            self._worker.join()
            self._writer_dataset.finalize()
            if self._worker_error is not None:
                logger.error(f"Recorder finalized after worker error: {self._worker_error}")


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
    include_tactile_images: bool = True,
    tactile_image_height: int = 400,
    tactile_image_width: int = 700,
    vcodec: str = "h264",
) -> LeRobotRecorderSubscriber:
    """Create a non-blocking recorder for a new dataset."""

    local_dir = pathlib.Path(root) if root else None
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
    return LeRobotRecorderSubscriber(
        dataset=dataset,
        task=task,
        only_intervention=only_intervention,
        include_tactile_images=include_tactile_images,
    )
