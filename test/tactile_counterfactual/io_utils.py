"""Run output: per-run YAML (human readable) + .npz attachments for bulky arrays.

Layout:

    <output.dir>/<timestamp>/
        probe_config.yaml          # copy of the input probe spec
        summary.yaml               # per-pair metrics + overall progress
        progress.json              # resume bookkeeping (completed pair ids)
        runs/
            <pair_id:04d>_<CONDITION>.yaml
            <pair_id:04d>_<CONDITION>_trace.npz   # optional bulky arrays
"""

from __future__ import annotations

from collections.abc import Sequence
import datetime as _datetime
import json
import pathlib
from typing import Any

import numpy as np
import yaml

from test.tactile_counterfactual.trace_sampler import ActionTrace

_FLOAT_TYPES = (np.floating,)
_INT_TYPES = (np.integer,)


def _flow_list(dumper: yaml.Dumper, data: list) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class _FlowListDumper(yaml.SafeDumper):
    """SafeDumper that renders lists in flow style (compact, readable)."""


_FlowListDumper.add_representer(list, _flow_list)


def _to_python(value: Any) -> Any:
    """Recursively convert numpy/jax values to plain Python (float32 -> float)."""
    if isinstance(value, np.ndarray):
        return [_to_python(v) for v in value.tolist()]
    if isinstance(value, _FLOAT_TYPES):
        return float(value)
    if isinstance(value, _INT_TYPES):
        return int(value)
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def create_run_dir(base_dir: str | pathlib.Path) -> pathlib.Path:
    """Create <base_dir>/<timestamp> and return it."""
    base = pathlib.Path(base_dir)
    ts = _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "runs").mkdir(parents=True, exist_ok=True)
    return run_dir


class RunWriter:
    """Writes one YAML + optional npz per traced inference."""

    def __init__(self, run_dir: pathlib.Path, output_spec, tactile_keys: Sequence[str]) -> None:
        self._run_dir = pathlib.Path(run_dir)
        (self._run_dir / "runs").mkdir(parents=True, exist_ok=True)
        self._output = output_spec
        self._tactile_keys = tuple(tactile_keys)
        self._hidden_dtype = np.dtype(
            {
                "float16": np.float16,
                "bfloat16": np.float32,  # numpy has no bfloat16; store as float32
                "float32": np.float32,
            }[output_spec.hidden_storage_dtype]
        )

    # ------------------------------------------------------------------ #

    def save_run(
        self,
        run_id: str,
        metadata: dict[str, Any],
        trace: ActionTrace,
    ) -> dict[str, Any]:
        """Save one traced run; returns the metadata dict actually written."""
        out = dict(metadata)
        out["run_id"] = run_id
        out["condition"] = metadata.get("condition")

        o = self._output
        npz_paths: dict[str, str] = {}
        npz_arrays: dict[str, np.ndarray] = {}
        steps: list[dict[str, Any]] = []

        # -- per-step traces -------------------------------------------------
        num_steps = len(trace.steps)
        out["num_steps"] = num_steps
        out["shapes"] = {}
        out["shapes"]["action_horizon"] = int(trace.final_action.shape[1])
        out["shapes"]["action_dim"] = int(trace.final_action.shape[2])

        # Bulky per-step arrays -> npz (stacked over steps).
        if o.save_action_hidden:
            hidden = np.stack([np.asarray(s.action_hidden)[0] for s in trace.steps]).astype(
                self._hidden_dtype, copy=False
            )
            npz_arrays["action_hidden"] = hidden
            npz_paths["action_hidden"] = "action_hidden"
            out["shapes"]["action_hidden"] = list(hidden.shape)
        if o.save_suffix_tokens:
            suffix_tokens = np.stack([np.asarray(s.suffix_tokens)[0] for s in trace.steps]).astype(
                self._hidden_dtype, copy=False
            )
            npz_arrays["suffix_tokens"] = suffix_tokens
            npz_paths["suffix_tokens"] = "suffix_tokens"
            out["shapes"]["suffix_tokens"] = list(suffix_tokens.shape)
        if o.save_adarms_cond:
            conds = []
            for s in trace.steps:
                if s.adarms_cond is None:
                    conds.append(None)
                else:
                    conds.append(np.asarray(s.adarms_cond)[0].astype(self._hidden_dtype, copy=False))
            # Per-token 3D cond (b, s, emb) is bulky -> npz; the 2D (b, emb)
            # case (scalar timestep, standard mode) is small enough for YAML.
            # After the batch squeeze the 3D case has ndim 2, the 2D case ndim 1.
            if conds and all(c is not None and c.ndim >= 2 for c in conds):
                stacked = np.stack(conds)
                npz_arrays["adarms_cond"] = stacked
                npz_paths["adarms_cond"] = "adarms_cond"
                out["shapes"]["adarms_cond"] = list(stacked.shape)
            else:
                out["adarms_cond"] = [None if c is None else _to_python(c) for c in conds]
                if conds and conds[0] is not None:
                    out["shapes"]["adarms_cond"] = list(conds[0].shape)

        # Small per-step quantities -> YAML.
        for s in trace.steps:
            entry: dict[str, Any] = {
                "step": s.step,
                "timestep": float(np.asarray(s.timestep)),
                "suffix_len": int(np.asarray(s.suffix_input_mask).shape[1]),
                "suffix_input_mask": _to_python(np.asarray(s.suffix_input_mask)[0]),
                "suffix_ar_mask": _to_python(np.asarray(s.suffix_ar_mask)),
                "action_hidden_rms": _rms(np.asarray(s.action_hidden)[0]),
                "v_t_rms": _rms(np.asarray(s.v_t)[0]),
                "x_t_before_rms": _rms(np.asarray(s.x_t_before)[0]),
                "x_t_after_rms": _rms(np.asarray(s.x_t_after)[0]),
            }
            if s.time_masked is not None:
                tm = np.asarray(s.time_masked)[0]
                entry["time_masked_min"] = float(tm.min())
                entry["time_masked_max"] = float(tm.max())
            steps.append(entry)

            if o.save_v_t:
                entry["v_t"] = _to_python(np.asarray(s.v_t)[0].astype(np.float32, copy=False))
            if o.save_x_t:
                entry["x_t_before"] = _to_python(np.asarray(s.x_t_before)[0].astype(np.float32, copy=False))
                entry["x_t_after"] = _to_python(np.asarray(s.x_t_after)[0].astype(np.float32, copy=False))
        out["steps"] = steps

        # -- tactile representation -------------------------------------------
        fastvit = np.asarray(trace.fastvit_features)[0]
        tokens = np.asarray(trace.tactile_tokens)[0]
        tokens_from_suffix = np.asarray(trace.tactile_tokens_from_suffix)[0]
        out["tactile_keys"] = list(self._tactile_keys)
        out["tactile_masks"] = _to_python(np.asarray(trace.tactile_mask)[0])
        if o.save_fastvit_features:
            out["fastvit_features"] = _to_python(fastvit.astype(np.float32, copy=False))
        if o.save_tactile_tokens:
            out["tactile_tokens"] = _to_python(tokens.astype(np.float32, copy=False))
        out["shapes"]["fastvit_features"] = list(fastvit.shape)
        out["shapes"]["tactile_tokens"] = list(tokens.shape)
        # Sanity cross-check numbers (manual FastViT branch vs embed_suffix).
        out["tactile_token_check"] = {
            "max_abs_diff_vs_suffix": float(
                np.abs(tokens.astype(np.float64) - tokens_from_suffix.astype(np.float64)).max()
            ),
        }

        # -- final action -----------------------------------------------------
        final = np.asarray(trace.final_action)[0].astype(np.float32, copy=False)
        out["final_action"] = _to_python(final)
        out["shapes"]["final_action"] = list(final.shape)
        out["final_action_rms"] = _rms(final)

        # -- write files ------------------------------------------------------
        written: dict[str, str] = {}
        if npz_arrays:
            npz_path = self._run_dir / "runs" / f"{run_id}_trace.npz"
            np.savez(npz_path, **npz_arrays)
            written["npz"] = str(npz_path.relative_to(self._run_dir))
            out["trace_npz"] = str(npz_path.name)
            out["trace_npz_keys"] = {k: list(v.shape) for k, v in npz_arrays.items()}

        # -- write files ------------------------------------------------------
        yaml_path = self._run_dir / "runs" / f"{run_id}.yaml"
        text = yaml.dump(_to_python(out), Dumper=_FlowListDumper, sort_keys=False, width=200)
        yaml_path.write_text(text, encoding="utf-8")
        written["yaml"] = str(yaml_path.relative_to(self._run_dir))

        return out

    # ------------------------------------------------------------------ #

    def write_probe_config(self, text: str) -> None:
        (self._run_dir / "probe_config.yaml").write_text(text, encoding="utf-8")

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self._run_dir / "summary.yaml").write_text(
            yaml.dump(_to_python(summary), Dumper=_FlowListDumper, sort_keys=False, width=200), encoding="utf-8"
        )

    def load_summary(self) -> dict[str, Any] | None:
        path = self._run_dir / "summary.yaml"
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Progress / resume                                                   #
    # ------------------------------------------------------------------ #

    @property
    def progress_path(self) -> pathlib.Path:
        return self._run_dir / "progress.json"

    def load_progress(self) -> set[int]:
        if not self.progress_path.is_file():
            return set()
        data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        return {int(p) for p in data.get("completed_pairs", [])}

    def save_progress(self, completed: set[int], runs_written: int) -> None:
        self.progress_path.write_text(
            json.dumps({"completed_pairs": sorted(completed), "runs_written": runs_written}, indent=2),
            encoding="utf-8",
        )
