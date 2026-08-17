"""Command-line alias for Right Arrow controlled recording episodes.

Run with options such as ``--host``, ``--record``, and
``--keyboard-episode-control``.  The implementation is shared with
``keyboard_main``.
"""

from __future__ import annotations

import tyro

from examples.bi_flexiv_rizon4_rt_record.keyboard_main import Args, main


if __name__ == "__main__":
    main(tyro.cli(Args))

