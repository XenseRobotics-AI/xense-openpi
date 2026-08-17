"""Optional keyboard-gated entry point for ``bi_flexiv_rizon4_rt_record``.

This keeps the original ``main.py`` entry point unchanged while adding the
``--keyboard_episode_control`` option.  With that option enabled, Right Arrow
ends the active episode, waits in reset mode after moving the robot home, and a
second Right Arrow starts the next inference episode.

Example:

    python -m examples.bi_flexiv_rizon4_rt_record.keyboard_main \
        --host 10.142.1.1 --port 8000 \
        --record --num_episodes 10 --keyboard_episode_control
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from examples.bi_flexiv_rizon4_rt_record import keyboard_runtime as _keyboard_runtime
from examples.bi_flexiv_rizon4_rt_record import main as _main


@dataclass
class Args(_main.Args):
    """Original recording-example arguments plus keyboard episode control."""

    keyboard_episode_control: bool = False


def main(args: Args) -> None:
    if not args.keyboard_episode_control:
        _main.main(args)
        return
    if args.action_hz > 0:
        raise SystemExit(
            "--keyboard_episode_control requires synchronous runtime; "
            "set --action_hz 0."
        )

    listener = _keyboard_runtime.RightArrowListener()
    original_runtime = _main._runtime.Runtime

    def make_runtime(**kwargs):
        return _keyboard_runtime.KeyboardEpisodeRuntime(keyboard=listener, **kwargs)

    _main._runtime.Runtime = make_runtime
    try:
        with listener:
            _main.main(args)
    finally:
        _main._runtime.Runtime = original_runtime


if __name__ == "__main__":
    tyro.cli(main)

