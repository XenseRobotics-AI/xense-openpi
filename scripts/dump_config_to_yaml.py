"""Dump a training config to YAML.

Training configs live in `configs/` as YAML (see `configs/README.md`); this script
exists for the configs that are still built in Python — the RoboArena baselines from
`config._generated_configs` — and for turning a config you assembled in a REPL into a
file you can check in.

Usage:
    # One config, printed to stdout.
    python scripts/dump_config_to_yaml.py paligemma_fast_droid

    # One config, written to configs/<name>.yaml (your gitignored working dir).
    python scripts/dump_config_to_yaml.py debug_pi05 --output-dir configs

    # Everything that can be serialized. Configs carrying lambdas or bare classes
    # are reported as skipped — those cannot round-trip and stay in Python.
    python scripts/dump_config_to_yaml.py --all --output-dir configs/_examples --overwrite
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import openpi.training.config as _config
import openpi.training.yaml_loader as _yaml_loader


def dump_one(name: str, config, output_dir: pathlib.Path | None, overwrite: bool) -> tuple[str | None, str | None]:
    """Serialize one config. Returns (written_path_or_None, error_or_None)."""
    try:
        yaml_text = _yaml_loader.dump(config)
    except ValueError as exc:
        return None, str(exc).splitlines()[0]

    # Re-parse and compare: a YAML that doesn't reload to the same config is worse
    # than no YAML at all, since it would silently train something else.
    try:
        reloaded = _yaml_loader.loads(yaml_text, name=name)
    except Exception as exc:
        return None, f"reload failed: {exc!r}"
    if reloaded != config:
        return None, "round-trip dataclass equality failed"

    if output_dir is None:
        print(yaml_text)
        return "<stdout>", None

    target = output_dir / f"{name}.yaml"
    if target.exists() and not overwrite:
        return None, f"exists, use --overwrite to replace: {target}"
    output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml_text)
    return str(target), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="Config name (see scripts/train.py --help for the list)")
    parser.add_argument("--all", action="store_true", help="Dump every known config instead of one")
    parser.add_argument("--output-dir", type=pathlib.Path, help="Write files here instead of printing to stdout")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in --output-dir")
    args = parser.parse_args()

    if bool(args.name) == bool(args.all):
        parser.error("Pass either a config name or --all.")

    configs = _config.all_configs() if args.all else {args.name: _config.get_config(args.name)}

    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name, config in sorted(configs.items()):
        path, error = dump_one(name, config, args.output_dir, args.overwrite)
        if error is None:
            written.append(path or name)
        else:
            skipped.append((name, error))

    if args.output_dir is not None:
        print(f"\nwrote {len(written)} file(s):")
        for path in written:
            print(f"  + {path}")
    if skipped:
        print(f"\nskipped {len(skipped)} config(s) — these stay in Python:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
