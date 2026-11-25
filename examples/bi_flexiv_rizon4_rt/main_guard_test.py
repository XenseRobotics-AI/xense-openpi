"""The startup guard against running a checkpoint with the wrong 20D layout."""

import pytest


def _main():
    # main.py pulls in the xense-client brokers, which need dm-tree (present on the
    # robot host, not in the openpi-only env these tests also run in).
    pytest.importorskip("tree")
    from examples.bi_flexiv_rizon4_rt import main

    return main


def test_xtac_umi_policy_is_refused():
    main = _main()
    with pytest.raises(SystemExit, match="action_space='xtac_umi'"):
        main._refuse_foreign_action_space({"action_space": "xtac_umi"}, main.Args())


def test_this_examples_own_policy_passes():
    main = _main()
    main._refuse_foreign_action_space({}, main.Args())
    main._refuse_foreign_action_space({"action_space": "bi_flexiv"}, main.Args())


def test_the_error_names_the_right_entry_point():
    main = _main()
    with pytest.raises(SystemExit, match=r"xtac_umi_bi_flexiv_rizon4_rt\.main"):
        main._refuse_foreign_action_space({"action_space": "xtac_umi"}, main.Args())
