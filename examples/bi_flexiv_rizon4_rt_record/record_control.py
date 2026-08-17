"""Unified recording entry point with resume and intervention-only modes.

This entry point preserves all arguments from the original recording example
and adds:

* ``--record-resume``: append new episodes to an existing local dataset.
* ``--keyboard-episode-control``: Right Arrow reset/next-inference gating.

Use the existing ``--record-intervention`` flag to record only Pico4 takeover
frames.  Policy inference frames are skipped in that mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from examples.bi_flexiv_rizon4_rt_record import keyboard_runtime as _keyboard_runtime
from examples.bi_flexiv_rizon4_rt_record import main as _main
from examples.bi_flexiv_rizon4_rt_record import resumable_recorder as _resumable_recorder


@dataclass
class Args(_main.Args):
    """Original arguments plus resumable and keyboard-controlled recording."""

    record_resume: bool = False
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

    def make_recorder(**kwargs):
        return _resumable_recorder.make_recorder_subscriber(
            **kwargs,
            resume=args.record_resume,
        )

    listener: _keyboard_runtime.RightArrowListener | None = None
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
        _main._recorder.make_recorder_subscriber = original_recorder_factory
        _main._runtime.Runtime = original_runtime


if __name__ == "__main__":
    main(tyro.cli(Args))

