"""Load a run's CLI arguments from a YAML file.

A *recipe* (``examples/flexiv_recipe.py``) describes a bench: which arms, which
cameras, which gripper. A *run* describes everything else about one launch —
which policy server, which task, whether RTC is on, whether to record — the
knobs that used to be eight lines of flags on every command line, copied between
laptops and slowly drifting apart.

Run YAML keys are exactly the CLI field names, so a run file is the command you
would have typed::

    # examples/bi_flexiv_rizon4_rt/runs/dewu-shoe-insole.yaml
    robot_recipe: forward-05
    host: 192.168.2.100
    rtc_enabled: true
    subscribe: true
    subscribe_url: ws://192.168.2.50:9100

    python -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole

Precedence is ``dataclass defaults < run YAML < CLI flags``: the YAML supplies
the defaults tyro then parses over, so any flag still wins for a one-off, and
``--help`` shows the values that are actually in effect. With no ``--args.run``
the CLI behaves exactly as it did before this file existed.

A key that is not a field of the example's ``Args`` fails the load, the same way
a typo in a recipe fails draccus: a silently ignored line in a run file is how
you end up believing RTC was on when it wasn't.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
import sys
import types
import typing
from typing import Any

import tyro
import yaml

_SUFFIXES = (".yaml", ".yml")


def available(directory: Path) -> list[str]:
    """Names of the YAML files in ``directory``, sorted."""
    return sorted({p.stem for suffix in _SUFFIXES for p in directory.glob(f"*{suffix}")})


def resolve_path(directory: Path, value: str | Path, *, kind: str = "file") -> Path:
    """Turn a CLI value into a path.

    A bare name (``forward-05``) resolves against ``directory``. Anything with a
    separator or a YAML suffix is taken as a path, so a file living outside the
    repo can be passed directly.
    """
    text = str(value)
    candidate = Path(text).expanduser()
    if "/" in text or candidate.suffix in _SUFFIXES:
        if not candidate.is_file():
            raise FileNotFoundError(f"{kind.capitalize()} not found: {candidate}")
        return candidate

    for suffix in _SUFFIXES:
        path = directory / f"{text}{suffix}"
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Unknown {kind} {text!r}. Available: {', '.join(available(directory)) or '(none)'}. "
        f"Pass a path instead to use a {kind} from outside {directory}."
    )


def peek_run(argv: list[str] | None = None, *, flag: str = "--args.run") -> str | None:
    """The value of ``--args.run`` in argv, before tyro parses anything."""
    args = sys.argv[1:] if argv is None else argv
    for index, item in enumerate(args):
        if item == flag:
            if index + 1 >= len(args):
                raise SystemExit(f"{flag} needs a value (a run name or a path to a YAML file).")
            return args[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def load_defaults[T](args_cls: type[T], runs_dir: Path, argv: list[str] | None = None) -> T:
    """Build an ``args_cls`` instance from the run YAML named on the command line.

    Returns a plain default instance when no ``--args.run`` was passed.
    """
    run = peek_run(argv)
    if run is None:
        return args_cls()

    path = resolve_path(runs_dir, run, kind="run")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a YAML mapping — see runs/README.md.")

    fields = {f.name: f for f in dataclasses.fields(args_cls)}
    if "run" in raw:
        raise ValueError(f"{path} sets `run:`, which is the file itself. Remove that line.")
    if unknown := sorted(set(raw) - set(fields)):
        raise ValueError(
            f"{path} has keys that are not run settings: {unknown}. "
            f"Valid keys are the CLI field names: {sorted(fields)}."
        )

    values = {name: _coerce(fields[name].type, value) for name, value in raw.items()}
    return args_cls(run=str(run), **values)


def cli[T](main_fn: Any, args_cls: type[T], runs_dir: Path, argv: list[str] | None = None) -> T:
    """Parse the CLI with a run YAML underneath it, and return the result.

    ``main_fn`` is only used for its docstring, which tyro prints as the command's
    help text — the parsed arguments come back to the caller instead of being
    passed straight on, so the example can log what the run file did.
    """
    defaults = load_defaults(args_cls, runs_dir, argv)

    def collect(args):
        return args

    collect.__annotations__ = {"args": args_cls, "return": args_cls}
    collect.__doc__ = main_fn.__doc__
    return tyro.cli(functools.partial(collect, args=defaults), args=argv)


def describe(args: Any, args_cls: type, runs_dir: Path, argv: list[str] | None = None) -> str:
    """A one-line summary of where the settings came from, for the startup log."""
    run = getattr(args, "run", None)
    if not run:
        return "No run file (--args.run); using CLI flags and defaults."

    defaults = load_defaults(args_cls, runs_dir, argv)
    overridden = sorted(
        f.name
        for f in dataclasses.fields(args_cls)
        if f.name != "run" and getattr(args, f.name) != getattr(defaults, f.name)
    )
    path = resolve_path(runs_dir, run, kind="run")
    suffix = f"; CLI overrides: {', '.join(overridden)}" if overridden else ""
    return f"Run config: {path}{suffix}"


def _coerce(annotation: Any, value: Any) -> Any:
    """YAML gives lists where some fields want tuples; everything else passes through."""
    if isinstance(value, list) and _wants_tuple(annotation):
        return tuple(value)
    return value


def _wants_tuple(annotation: Any) -> bool:
    if isinstance(annotation, str):
        # Postponed annotation (`from __future__ import annotations` in the example).
        return annotation.startswith(("tuple", "Tuple", "typing.Tuple"))
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return any(_wants_tuple(arg) for arg in typing.get_args(annotation))
    return origin is tuple
