"""Recording entry point with resume, intervention-only, and keyboard modes."""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from examples.bi_flexiv_rizon4_rt_record import keyboard_runtime as _keyboard_runtime
from examples.bi_flexiv_rizon4_rt_record import main as _main
from examples.bi_flexiv_rizon4_rt_record import resumable_recorder as _resumable_recorder


@dataclass
class Args(_main.Args):
    """Original recording options plus append/resume and keyboard control."""

    # Append newly saved episodes to the local dataset selected by
    # record_repo_id/record_root instead of creating it from scratch.
    record_resume: bool = False
    # Right Arrow: finish episode/reset; Right Arrow again: next inference.
    keyboard_episode_control: bool = False


def main(args: Args) -> None:
    if args.record_resume and not (args.record or args.record_intervention):
        raise SystemExit("--record-resume requires --record or --record-intervention")
    if args.keyboard_episode_control and args.action_hz > 0:
        raise SystemExit(
            "--keyboard-episode-control requires synchronous runtime; set --action-hz 0."
        )

    original_recorder_factory = _main._recorder.make_recorder_subscriber
    original_runtime = _main._runtime.Runtime
    created_recorders = []

    def make_recorder(**kwargs):
        recorder = _resumable_recorder.make_recorder_subscriber(
            **kwargs,
            resume=args.record_resume,
        )
        created_recorders.append(recorder)
        return recorder

    _main._recorder.make_recorder_subscriber = make_recorder
    try:
        if args.keyboard_episode_control:
            listener = _keyboard_runtime.RightArrowListener()

            def make_runtime(**kwargs):
                return _keyboard_runtime.KeyboardEpisodeRuntime(keyboard=listener, **kwargs)

            _main._runtime.Runtime = make_runtime
            with listener:
                _main.main(args)
        else:
            _main.main(args)
    finally:
        # LeRobot v3 buffers parquet metadata. Explicit finalization makes the
        # dataset immediately valid for the next --record-resume process.
        for recorder in created_recorders:
            recorder._dataset.finalize()
        _main._recorder.make_recorder_subscriber = original_recorder_factory
        _main._runtime.Runtime = original_runtime


if __name__ == "__main__":
    main(tyro.cli(Args))

