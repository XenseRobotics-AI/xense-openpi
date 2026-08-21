#!/usr/bin/env python3
"""Offline tactile counterfactual inference diagnostic tool.

Usage:

    python scripts/tactile_counterfactual_probe.py \
        --config configs/probes/water_weight_counterfactual.yaml

    # quick validation (no experiment loop)
    python scripts/tactile_counterfactual_probe.py \
        --config configs/probes/water_weight_counterfactual.yaml \
        --max-pairs 2 --dry-run

    # resume an interrupted run
    python scripts/tactile_counterfactual_probe.py \
        --config configs/probes/water_weight_counterfactual.yaml \
        --resume outputs/tactile_counterfactual/20260821_120000

Run from the repo root with the repo's training environment, e.g.:

    PYTHONPATH=src:packages/xense-client/src \
        python scripts/tactile_counterfactual_probe.py --config ...

See test/tactile_counterfactual/ for the implementation.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "packages" / "xense-client" / "src"))

from test.tactile_counterfactual.io_utils import create_run_dir
from test.tactile_counterfactual.probe_config import load_probe_config
from test.tactile_counterfactual.runner import ProbeRunner
from test.tactile_counterfactual.runner import build_probe_dataset
from test.tactile_counterfactual.runner import build_train_config
from test.tactile_counterfactual.runner import load_model
from test.tactile_counterfactual.runner import resolve_data_config
from test.tactile_counterfactual.trace_sampler import TraceSampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Tactile counterfactual inference probe")
    parser.add_argument("--config", required=True, help="Path to the probe YAML spec")
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-end", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-strict-validation",
        action="store_true",
        help="Downgrade sanity-check failures to warnings (default: fail fast).",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default=None,
        help="Resume into an existing timestamped output directory.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("openpi").setLevel(logging.INFO)

    probe = load_probe_config(args.config)
    if probe.experiment.batch_size != 1:
        raise NotImplementedError(
            f"batch_size={probe.experiment.batch_size} is not supported yet; "
            "the first version requires batch_size=1 so each counterfactual "
            "condition keeps its paired noise exactly."
        )

    train_config = build_train_config(probe)
    checkpoint_dir = pathlib.Path(probe.model.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"model.checkpoint_dir does not exist: {checkpoint_dir}")

    data_config = resolve_data_config(train_config, probe, checkpoint_dir)
    dataset = build_probe_dataset(probe, train_config, data_config)
    model, _ = load_model(train_config, checkpoint_dir)

    # The model config owns the tactile key order.
    tactile_keys = tuple(train_config.model.tactile_image_keys)
    sampler = TraceSampler(model, tactile_keys, inference_mode=probe.experiment.inference_mode)

    if args.resume is not None:
        run_dir = pathlib.Path(args.resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume directory does not exist: {run_dir}")
        logging.info("Resuming into existing run directory %s", run_dir)
    else:
        run_dir = create_run_dir(probe.output.dir)
        logging.info("Created run directory %s", run_dir)
    (run_dir / "probe_config.yaml").write_text(pathlib.Path(args.config).read_text(encoding="utf-8"))

    runner = ProbeRunner(
        probe,
        train_config,
        model,
        dataset,
        sampler,
        run_dir=run_dir,
    )
    summary = runner.run(
        pair_start=args.pair_start,
        pair_end=args.pair_end,
        max_pairs=args.max_pairs,
        dry_run=args.dry_run,
        resume=args.resume is not None,
        skip_strict_validation=args.skip_strict_validation,
    )
    logging.info("Summary written to %s", run_dir / "summary.yaml")
    if args.dry_run:
        print(f"DRY RUN OK — run directory: {run_dir}")
    else:
        print(f"Done — run directory: {run_dir}")


if __name__ == "__main__":
    main()
