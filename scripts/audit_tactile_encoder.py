"""Experiment B: does the trained tactile encoder tell contact from no-contact?

Loads only ``tactile_encoder`` + ``tactile_proj`` out of a trained checkpoint
(~30 MB, not the 12 GB backbone), runs a set of tactile images through them, and
reports pairwise distances between the resulting action-expert tokens.

The question this answers is whether the tactile *representation* carries a
usable signal at all. If contact and no-contact land at the same distance as two
frames of the same state, the encoder has collapsed and nothing downstream --
including the action expert -- could use it. That is a different failure from
"the action expert learned to ignore an informative token", and it has to be
ruled out first.

Controls matter more than the headline number here: a pure-noise image and a
black image are fed through the same path to show what a genuinely large
embedding distance looks like on this encoder.

Usage:
    python scripts/audit_tactile_encoder.py \
        --checkpoint-dir checkpoints/<config>/<step> \
        --contact 有水_..._tactile_0.jpg 有水_..._tactile_1.jpg \
        --no-contact 无水_..._tactile_0.jpg 无水_..._tactile_1.jpg \
        --no-contact-b .../右臂未抓取_..._tactile_0.jpg .../右臂未抓取_..._tactile_1.jpg
"""

from __future__ import annotations

import argparse
import itertools
import pathlib

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import PIL.Image

from openpi.models.tactile_encoders import build_tactile_encoder
from openpi.shared import image_tools

# audit_tactile_weights lives next to this file and owns the partial-restore logic.
from scripts.audit_tactile_weights import _build  # noqa: PLC2701
import orbax.checkpoint as ocp


def _restore_tactile(checkpoint_dir: pathlib.Path) -> dict:
    """Restore just params/tactile_* as a nested dict, nnx replace_by_pure_dict-shaped."""
    path = (checkpoint_dir / "params").resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = dict(ckptr.metadata(path))
        restored = ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=_build(metadata, keep=lambda p: "tactile" in p),
                restore_args=_build(metadata, as_args=True),
            ),
        )
    # save_state writes nnx.State, so every path ends in "value". Strip it to get
    # what nnx calls a pure dict (same fixup restore_params does).
    import flax.traverse_util as tu

    flat = tu.flatten_dict(restored["params"])
    if all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    return tu.unflatten_dict(flat)


def _load_image(path: pathlib.Path) -> np.ndarray:
    """JPEG -> the exact (224, 224, 3) float32 [-1, 1] array the model receives.

    Mirrors the serving path: ResizeImages(224, 224) does resize_with_pad on the
    uint8 HWC frame, then Observation.from_dict maps uint8 -> [-1, 1].
    """
    raw = np.asarray(PIL.Image.open(path).convert("RGB"), dtype=np.uint8)
    resized = image_tools.resize_with_pad(raw[None], 224, 224)[0]
    return np.asarray(resized, dtype=np.float32) / 255.0 * 2.0 - 1.0


def _tokens(encoder, proj, images: np.ndarray) -> np.ndarray:
    """(N, 224, 224, 3) in [-1, 1] -> (N, action_expert_width) suffix tokens."""
    feats = encoder(jnp.asarray(images))
    return np.asarray(proj(feats), dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--contact", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--no-contact", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--no-contact-b", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument("--width", type=int, default=1024, help="action expert width")
    parser.add_argument("--compute-dtype", default="bfloat16")
    args = parser.parse_args()

    ckpt = args.checkpoint_dir.expanduser().resolve()
    print(f"checkpoint: {ckpt}")

    rngs = nnx.Rngs(0)
    encoder = build_tactile_encoder(
        "fastvit_t12", rngs=rngs, pretrained_path=None, compute_dtype=jnp.dtype(args.compute_dtype)
    )
    proj = nnx.Linear(encoder.feature_dim, args.width, rngs=rngs)

    params = _restore_tactile(ckpt)
    for module, pure in ((encoder, params["tactile_encoder"]), (proj, params["tactile_proj"])):
        # nnx.state() hands back a copy; the module only sees the trained weights
        # after nnx.update writes the modified State back.
        state = nnx.state(module)
        state.replace_by_pure_dict(pure)
        nnx.update(module, state)
    encoder.eval()
    print(f"loaded tactile_encoder (feature_dim={encoder.feature_dim}) + tactile_proj -> {args.width}")

    groups: dict[str, list[pathlib.Path]] = {
        "contact": [p.expanduser() for p in args.contact],
        "no_contact_a": [p.expanduser() for p in args.no_contact],
    }
    if args.no_contact_b:
        groups["no_contact_b"] = [p.expanduser() for p in args.no_contact_b]

    n_sensors = len(groups["contact"])
    for name, paths in groups.items():
        if len(paths) != n_sensors:
            raise SystemExit(f"group {name} has {len(paths)} images, expected {n_sensors}")

    images = {name: np.stack([_load_image(p) for p in paths]) for name, paths in groups.items()}

    # Controls: what a genuinely large input change looks like on this encoder.
    rng = np.random.default_rng(0)
    images["noise"] = rng.uniform(-1.0, 1.0, size=images["contact"].shape).astype(np.float32)
    images["black"] = np.full_like(images["contact"], -1.0)

    print("\n[B0] input images actually differ (per sensor, in [-1, 1] space)")
    base = images["no_contact_a"]
    for name, arr in images.items():
        if name == "no_contact_a":
            continue
        per_sensor = [float(np.abs(arr[i] - base[i]).mean()) for i in range(n_sensors)]
        print(f"  mean|{name:12s} - no_contact_a| per sensor = {[f'{v:.4f}' for v in per_sensor]}")

    tokens = {name: _tokens(encoder, proj, arr) for name, arr in images.items()}

    print("\n[B1] token statistics (action-expert width vectors)")
    for name, tok in tokens.items():
        norms = np.linalg.norm(tok, axis=-1)
        print(
            f"  {name:14s} ||t||={np.array2string(norms, precision=3)}  "
            f"std_over_dims={tok.std(axis=-1).mean():.5f}"
        )

    print("\n[B2] pairwise distance between conditions, per sensor")
    print(f"  {'pair':34s} {'sensor':>6s} {'cosine':>10s} {'L2':>10s} {'rel_L2':>10s}")
    for a, b in itertools.combinations(tokens, 2):
        for i in range(n_sensors):
            ta, tb = tokens[a][i], tokens[b][i]
            l2 = float(np.linalg.norm(ta - tb))
            rel = l2 / float(np.linalg.norm(ta)) if np.linalg.norm(ta) else float("nan")
            print(f"  {a + ' vs ' + b:34s} {i:6d} {_cos(ta, tb):10.6f} {l2:10.4f} {rel:10.4f}")

    print("\n[B3] verdict input")
    signal, floor = [], []
    for i in range(n_sensors):
        signal.append(1.0 - _cos(tokens["contact"][i], tokens["no_contact_a"][i]))
        if "no_contact_b" in tokens:
            floor.append(1.0 - _cos(tokens["no_contact_a"][i], tokens["no_contact_b"][i]))
    ctrl = [1.0 - _cos(tokens["contact"][i], tokens["noise"][i]) for i in range(n_sensors)]
    print(f"  signal  (contact vs no_contact_a)   1-cos = {[f'{v:.6f}' for v in signal]}")
    if floor:
        print(f"  floor   (no_contact_a vs _b)        1-cos = {[f'{v:.6f}' for v in floor]}")
    print(f"  control (contact vs random noise)   1-cos = {[f'{v:.6f}' for v in ctrl]}")
    if floor:
        ratio = [s / f if f else float("inf") for s, f in zip(signal, floor, strict=True)]
        print(f"  signal / floor = {[f'{v:.2f}' for v in ratio]}   (~1 => encoder cannot separate contact)")


if __name__ == "__main__":
    main()
