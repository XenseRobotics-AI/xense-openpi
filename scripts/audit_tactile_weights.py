"""Experiment A: did gradients ever reach the tactile branch?

Two independent checks on a trained checkpoint:

1. Adam second moment (``nu``) for every ``tactile_*`` parameter. Adam accumulates
   ``nu = E[g^2]``, so ``nu == 0`` after N steps means the gradient was *exactly*
   zero for all N steps -- the branch was never trained. Compared against the
   backbone's own ``nu`` to give the number a scale.

2. The checkpoint's ``tactile_encoder`` weights versus the ImageNet FastViT file
   they were initialised from. Bit-identical means no update ever landed.

Only the tactile subtree (plus a few reference tensors) is restored, so this runs
in seconds on a laptop instead of pulling the full 12 GB params tree.

Usage:
    python scripts/audit_tactile_weights.py \
        --checkpoint-dir checkpoints/<config>/<step> \
        --pretrained ~/.cache/fastvit_t12_apple_dist_in1k_flax/params.safetensors
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import orbax.checkpoint as ocp

# Backbone tensors restored alongside the tactile ones purely to give `nu` a scale:
# a branch that trained normally should sit in the same ballpark as these.
_REFERENCE_KEYS = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")


def _build(node, path: str = "", *, keep=None, as_args: bool = False):
    """Mirror the checkpoint tree, keeping only the leaves `keep` accepts.

    Skipped leaves become ``ocp.PLACEHOLDER`` so orbax never touches them on disk.
    That matters here: the full train_state is 19 GB, the tactile slice is ~60 MB.
    Lists have to be walked too — optax's opt_state is a list of lists, not a dict.
    """
    if isinstance(node, dict):
        return {k: _build(v, f"{path}/{k}" if path else str(k), keep=keep, as_args=as_args) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_build(v, f"{path}/{i}" if path else str(i), keep=keep, as_args=as_args) for i, v in enumerate(node)]
    if as_args:
        # Every leaf position needs restore args, placeholders included, or orbax
        # falls through to a sharding-based path and fails on restore_type=None.
        return ocp.ArrayRestoreArgs(restore_type=np.ndarray)
    return node if keep(path) else ocp.PLACEHOLDER


def _flatten(tree, prefix: str = "") -> dict[str, np.ndarray]:
    out = {}
    if isinstance(tree, dict):
        items = tree.items()
    elif isinstance(tree, (list, tuple)):
        items = enumerate(tree)
    else:
        return {prefix: tree} if isinstance(tree, np.ndarray) else {}
    for key, value in items:
        out.update(_flatten(value, f"{prefix}/{key}" if prefix else str(key)))
    return out


def _restore(path: pathlib.Path, keep) -> dict[str, np.ndarray]:
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = dict(ckptr.metadata(path))
        restored = ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=_build(metadata, keep=keep),
                restore_args=_build(metadata, as_args=True),
            ),
        )
    return _flatten(restored)


def _stats(name: str, values: dict[str, np.ndarray]) -> tuple[float, float, int]:
    """Return (rms, max, n_leaves) over a group of tensors."""
    total_sq, total_n, peak = 0.0, 0, 0.0
    for arr in values.values():
        arr = np.asarray(arr, dtype=np.float64)
        total_sq += float((arr**2).sum())
        total_n += arr.size
        peak = max(peak, float(np.abs(arr).max()) if arr.size else 0.0)
    rms = float(np.sqrt(total_sq / total_n)) if total_n else 0.0
    print(f"  {name:28s} leaves={len(values):4d}  rms={rms:.6e}  max={peak:.6e}")
    return rms, peak, len(values)


def check_optimizer_moments(checkpoint_dir: pathlib.Path) -> None:
    print("\n[A1] Adam second moment (nu) -- zero means the gradient was zero for every step")
    train_state = checkpoint_dir / "train_state"
    if not train_state.is_dir():
        print(f"  train_state not found at {train_state}; skipping (params-only checkpoint)")
        return

    def keep(path: str) -> bool:
        if not path.startswith("opt_state/"):
            return False
        if "/nu/" not in path and "/mu/" not in path:
            return False
        return "tactile" in path or any(ref in path for ref in _REFERENCE_KEYS)

    restored = _restore(train_state, keep)
    for moment in ("mu", "nu"):
        sel = {k: v for k, v in restored.items() if f"/{moment}/" in k}
        if not sel:
            continue
        print(f"  -- {moment} --")
        tactile = {k: v for k, v in sel.items() if "tactile" in k}
        encoder = {k: v for k, v in tactile.items() if "tactile_encoder" in k}
        proj = {k: v for k, v in tactile.items() if "tactile_proj" in k}
        backbone = {k: v for k, v in sel.items() if "tactile" not in k}
        _stats("tactile_encoder", encoder)
        _stats("tactile_proj", proj)
        _stats("backbone (reference)", backbone)
        if moment == "nu":
            zero = [k for k, v in tactile.items() if not np.any(np.asarray(v))]
            print(f"  all-zero nu leaves: {len(zero)} / {len(tactile)}")
            if zero:
                print(f"    e.g. {zero[:3]}")


def check_weights_moved(checkpoint_dir: pathlib.Path, pretrained: pathlib.Path) -> None:
    print("\n[A2] checkpoint tactile_encoder vs the ImageNet weights it was initialised from")
    if not pretrained.is_file():
        print(f"  pretrained file not found: {pretrained}; skipping")
        return
    from safetensors.numpy import load_file

    ref = load_file(str(pretrained))
    restored = _restore(checkpoint_dir / "params", lambda p: "tactile" in p)
    # Checkpoint paths look like params/tactile_encoder/module/<...>/value; the
    # safetensors file is keyed by the bare <...> path.
    trimmed = {}
    for path, arr in restored.items():
        if "tactile_encoder" not in path:
            continue
        body = path.split("tactile_encoder/", 1)[1].removeprefix("module/").removesuffix("/value")
        trimmed[body] = arr

    matched = sorted(set(trimmed) & set(ref))
    print(f"  checkpoint tactile_encoder leaves={len(trimmed)}  pretrained={len(ref)}  matched={len(matched)}")
    if not matched:
        print(f"  no key overlap. checkpoint e.g. {sorted(trimmed)[:2]}, pretrained e.g. {sorted(ref)[:2]}")
        return

    identical, moved = [], []
    for key in matched:
        a, b = np.asarray(trimmed[key], np.float64), np.asarray(ref[key], np.float64)
        if a.shape != b.shape:
            continue
        delta = float(np.abs(a - b).max())
        scale = float(np.abs(b).max()) or 1.0
        (identical if delta == 0.0 else moved).append((key, delta, delta / scale))

    print(f"  bit-identical to pretrained: {len(identical)} / {len(matched)}")
    print(f"  changed:                     {len(moved)} / {len(matched)}")
    if moved:
        moved.sort(key=lambda kv: -kv[2])
        print("  largest relative movement:")
        for key, delta, rel in moved[:5]:
            print(f"    {key:60s} max|d|={delta:.4e}  rel={rel:.4e}")
        rels = np.array([r for _, _, r in moved])
        print(f"  relative movement: median={np.median(rels):.4e}  mean={rels.mean():.4e}")
    if identical:
        print(f"  still identical, e.g.: {[k for k, _, _ in identical[:3]]}")

    proj = {k: v for k, v in restored.items() if "tactile_proj" in k}
    if proj:
        print("  tactile_proj (fresh nnx.Linear at init, no pretrained reference):")
        for key, arr in sorted(proj.items()):
            arr = np.asarray(arr, np.float64)
            print(
                f"    {key:56s} shape={str(arr.shape):14s} std={arr.std():.6e} "
                f"max|.|={np.abs(arr).max():.6e}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--pretrained",
        type=pathlib.Path,
        default=pathlib.Path("~/.cache/fastvit_t12_apple_dist_in1k_flax/params.safetensors"),
    )
    args = parser.parse_args()
    ckpt = args.checkpoint_dir.expanduser().resolve()
    print(f"checkpoint: {ckpt}")
    check_optimizer_moments(ckpt)
    check_weights_moved(ckpt, args.pretrained.expanduser())


if __name__ == "__main__":
    main()
