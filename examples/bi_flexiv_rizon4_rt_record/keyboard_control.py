"""Flat-CLI alias for the optional keyboard-controlled recording entry point.

Use this module when you want the options without the ``--args.`` prefix:

    python -m examples.bi_flexiv_rizon4_rt_record.keyboard_control \
        --args.host 10.142.1.1 --args.record \
        --args.num-episodes 10 --args.keyboard-episode-control

The ``--args.`` spelling is retained here because it matches the existing
recording example's Tyro CLI.  The module exists as a discoverable alias for
the keyboard mode; ``keyboard_main`` is the implementation entry point.
"""

from __future__ import annotations

import tyro

from examples.bi_flexiv_rizon4_rt_record.keyboard_main import Args, main


if __name__ == "__main__":
    main(tyro.cli(Args))

