"""Build the future-tactile label store for the Latent Tactile Predictor.

docs/action-conditioned-tactile-pretraining.md section 4.1. For one LeRobot dataset the
script decodes the four tactile streams once, in episode/frame order, and writes to
``--out``:

    meta.json               what was built and from what (cameras, fps, horizons, PCA)
    episode_offsets.npy     int64 [E + 1]; episode e owns global frames offsets[e]:offsets[e+1]
    pixel_field.npy         uint8 [N, S, 16, 16, 3]  centre-cropped, down-sampled frames
    feat_tac.npy            float16 [N, S, 1024]     frozen FastViT-T12 features (E_tac)
    pca_mean.npy / pca_components.npy / pca_eigvals.npy   fitted on --fit-episodes
    z_tac.npy               float16 [N, S, D]        PCA-whitened features, the main target

``transforms.InjectTactileFutureLabels`` looks these up by ``(episode_index,
frame_index + k)``; ``meta.json`` also carries the per-horizon RMS of the pixel change
field so the ``pixel_delta`` control target has a unit-variance zero-prediction baseline.

E_tac is the *frozen* copy of the same initial FastViT weights the policy starts from,
so pass the same ``--encoder-weights`` the training config uses for
``tactile_pretrained_path`` -- the stock ImageNet file, BN statistics included: the plan
deliberately re-estimates nothing, so that the two sides are bit-identical by
construction (docs/action-conditioned-tactile-pretraining.md section 2.1). E_tac never
trains, which is what rules out target collapse.

Usage:
    python scripts/compute_tactile_future_labels.py \\
        --repo-id Xense/bottle-sorting-0810 \\
        --out assets/tactile_future_labels/bottle-sorting-0810

    # quick check on two episodes
    python scripts/compute_tactile_future_labels.py --repo-id ... --out /tmp/labels --episodes 0 1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

# Same two recorder conventions as compute_tactile_refs.py; pad 0 = left jaw, 1 = right jaw.
_CAMERA_CONVENTIONS = (
    ("left_tactile_0", "left_tactile_1", "right_tactile_0", "right_tactile_1"),
    ("left_tactile_left", "left_tactile_right", "right_tactile_left", "right_tactile_right"),
)
# Must match Pi0TactileFastVitConfig.tactile_image_keys order (tactile_0..3_rgb).
_POLICY_NAMES = ("left_tactile_top", "left_tactile_bottom", "right_tactile_top", "right_tactile_bottom")
_ENCODER_INPUT = 224


def _detect_cameras(features: dict) -> tuple[str, ...]:
    for cameras in _CAMERA_CONVENTIONS:
        if all(f"observation.images.{cam}" in features for cam in cameras):
            return cameras
    present = sorted(k for k in features if "tactile" in k)
    raise SystemExit(f"no known tactile naming matches this dataset; tactile columns present: {present}")


def _parse_episodes(spec: list[str] | None, num_episodes: int) -> list[int]:
    if not spec:
        return list(range(num_episodes))
    out: list[int] = []
    for token in spec:
        if "-" in token:
            lo, hi = token.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    bad = [e for e in out if not 0 <= e < num_episodes]
    if bad:
        raise SystemExit(f"episodes {bad} out of range [0, {num_episodes})")
    return sorted(set(out))


def _center_crop_square(frames: np.ndarray) -> np.ndarray:
    """[T, H, W, C] -> [T, side, side, C], same geometry as transforms.fit_square("center_crop")."""
    h, w = frames.shape[1:3]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return frames[:, top : top + side, left : left + side]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="Xense/bottle-sorting-0810")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="output directory (one per dataset)")
    parser.add_argument(
        "--encoder-weights",
        default="~/.cache/fastvit_t12_apple_dist_in1k_flax/params.safetensors",
        help="Flax FastViT-T12 weights; the same file the training config's tactile_pretrained_path points at",
    )
    parser.add_argument("--cameras", nargs="+", default=None, help="tactile columns; auto-detected if unset")
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--pixel-size", type=int, default=16)
    parser.add_argument("--episodes", nargs="*", default=None, help="subset to process, e.g. `0 1 5-9` (debug)")
    parser.add_argument(
        "--fit-episodes",
        nargs="*",
        default=None,
        help="episodes the PCA and pixel RMS are fitted on (default: every processed episode). "
        "Give the training split here when holding episodes out.",
    )
    parser.add_argument("--batch", type=int, default=256, help="frames per decode / encoder call")
    parser.add_argument("--encoder-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    # Heavy imports after argparse so --help stays fast.
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames

    from openpi.models.tactile_encoders import build_tactile_encoder
    from openpi.shared import nnx_utils

    dataset = LeRobotDataset(args.repo_id)
    meta = dataset.meta
    fps = float(meta.fps)
    cameras = tuple(args.cameras) if args.cameras else _detect_cameras(meta.features)
    if len(cameras) != len(_POLICY_NAMES):
        raise SystemExit(f"expected {len(_POLICY_NAMES)} tactile cameras, got {list(cameras)}")
    print(
        f"{args.repo_id}: {meta.total_episodes} episodes, {meta.total_frames} frames @ {fps:g} fps; cameras {cameras}"
    )

    # Episode layout. The store is indexed by global frame, so the offsets have to agree
    # with LeRobot's own `index` column; the training transform re-checks this per item.
    lengths = np.array([int(meta.episodes[e]["length"]) for e in range(meta.total_episodes)], dtype=np.int64)
    starts = np.array([int(meta.episodes[e]["dataset_from_index"]) for e in range(meta.total_episodes)])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    if not np.array_equal(starts, offsets[:-1]) or int(offsets[-1]) != int(meta.total_frames):
        raise SystemExit("episodes are not contiguous in frame order; the positional store would be wrong")
    num_frames = int(offsets[-1])
    num_pads = len(cameras)

    episodes = _parse_episodes(args.episodes, meta.total_episodes)
    fit_episodes = _parse_episodes(args.fit_episodes, meta.total_episodes) if args.fit_episodes else episodes
    if not set(fit_episodes) <= set(episodes):
        raise SystemExit("--fit-episodes must be a subset of --episodes")

    encoder = build_tactile_encoder(
        "fastvit_t12",
        rngs=nnx.Rngs(0),
        pretrained_path=pathlib.Path(args.encoder_weights).expanduser(),
        compute_dtype=jnp.dtype(args.encoder_dtype),
    )
    encode = nnx_utils.module_jit(encoder.__call__)
    feature_dim = encoder.feature_dim

    @jax.jit
    def preprocess(square_uint8):  # [T, side, side, 3] uint8 -> encoder input [-1, 1] + pixel field uint8
        x = square_uint8.astype(jnp.float32) / 255.0
        enc = jax.image.resize(x, (x.shape[0], _ENCODER_INPUT, _ENCODER_INPUT, 3), method="linear", antialias=True)
        pix = jax.image.resize(x, (x.shape[0], args.pixel_size, args.pixel_size, 3), method="linear", antialias=True)
        return enc * 2.0 - 1.0, jnp.round(jnp.clip(pix, 0.0, 1.0) * 255.0).astype(jnp.uint8)

    args.out.mkdir(parents=True, exist_ok=True)
    feat = np.lib.format.open_memmap(
        args.out / "feat_tac.npy", mode="w+", dtype=np.float16, shape=(num_frames, num_pads, feature_dim)
    )
    pixel = np.lib.format.open_memmap(
        args.out / "pixel_field.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(num_frames, num_pads, args.pixel_size, args.pixel_size, 3),
    )

    t_start = time.monotonic()
    done_frames = 0
    for n, ep in enumerate(episodes):
        ep_meta = meta.episodes[ep]
        start, length = int(offsets[ep]), int(lengths[ep])
        for pad, cam in enumerate(cameras):
            key = f"observation.images.{cam}"
            video_path = dataset.root / meta.get_video_file_path(ep, key)
            from_ts = float(ep_meta[f"videos/{key}/from_timestamp"])
            for block in range(0, length, args.batch):
                frames_idx = range(block, min(block + args.batch, length))
                timestamps = [from_ts + i / fps for i in frames_idx]
                # backend=None lets lerobot pick get_safe_default_codec(), the same choice
                # LeRobotDataset makes internally; 0.5.1 has no per-dataset backend to honour.
                frames = decode_video_frames(video_path, timestamps, dataset.tolerance_s)
                if frames.shape[0] != len(frames_idx):
                    raise SystemExit(
                        f"episode {ep} {cam}: asked for {len(frames_idx)} frames from {block}, decoded {frames.shape[0]}"
                    )
                # decode_video_frames returns float [T, C, H, W] in [0, 1].
                hwc = (np.asarray(frames.permute(0, 2, 3, 1)) * 255.0).round().clip(0, 255).astype(np.uint8)
                enc_in, pix = preprocess(jnp.asarray(_center_crop_square(hwc)))
                f = np.asarray(encode(enc_in), dtype=np.float32)
                rows = slice(start + block, start + block + len(frames_idx))
                feat[rows, pad] = f.astype(np.float16)
                pixel[rows, pad] = np.asarray(pix)
        done_frames += length
        elapsed = time.monotonic() - t_start
        print(
            f"  [{n + 1}/{len(episodes)}] episode {ep}: {length} frames; "
            f"{done_frames / max(elapsed, 1e-9):.0f} frames/s ({elapsed:.0f}s)"
        )
    feat.flush()
    pixel.flush()

    # ---- PCA whitening of E_tac on the fit split (all pads pooled) ----
    print("fitting PCA ...")
    total = np.zeros(feature_dim, dtype=np.float64)
    outer = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    count = 0
    for ep in fit_episodes:
        rows = feat[offsets[ep] : offsets[ep + 1]].reshape(-1, feature_dim).astype(np.float64)
        total += rows.sum(axis=0)
        outer += rows.T @ rows
        count += rows.shape[0]
    mean = total / count
    cov = outer / count - np.outer(mean, mean)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pca_dim = min(args.pca_dim, feature_dim)
    components = eigvecs[:, :pca_dim].T  # [D, feature_dim]
    top = np.clip(eigvals[:pca_dim], 1e-12, None)
    explained = float(top.sum() / np.clip(eigvals.clip(0).sum(), 1e-12, None))
    print(f"  PCA {pca_dim} dims explain {100 * explained:.1f}% of the variance ({count} fit vectors)")
    np.save(args.out / "pca_mean.npy", mean.astype(np.float32))
    np.save(args.out / "pca_components.npy", components.astype(np.float32))
    np.save(args.out / "pca_eigvals.npy", eigvals[:pca_dim].astype(np.float32))

    z = np.lib.format.open_memmap(
        args.out / "z_tac.npy", mode="w+", dtype=np.float16, shape=(num_frames, num_pads, pca_dim)
    )
    scale = 1.0 / np.sqrt(top)
    for ep in episodes:
        rows = feat[offsets[ep] : offsets[ep + 1]].astype(np.float32)
        z[offsets[ep] : offsets[ep + 1]] = (((rows - mean) @ components.T) * scale).astype(np.float16)
    z.flush()

    # ---- per-horizon RMS of the pixel change field, for the pixel_delta control ----
    print("computing pixel-field RMS ...")
    sq_sum = {k: np.zeros(num_pads, dtype=np.float64) for k in args.horizons}
    n_pix = dict.fromkeys(args.horizons, 0)
    for ep in fit_episodes:
        field = pixel[offsets[ep] : offsets[ep + 1]].astype(np.float32) / 255.0  # [L, S, h, w, 3]
        for k in args.horizons:
            if field.shape[0] <= k:
                continue
            delta = field[k:] - field[:-k]
            sq_sum[k] += np.square(delta).sum(axis=(0, 2, 3, 4))
            n_pix[k] += delta.shape[0] * int(np.prod(delta.shape[2:]))
    y_delta_rms = {str(k): np.sqrt(sq_sum[k] / max(n_pix[k], 1)).clip(1e-6).tolist() for k in args.horizons}

    written = {
        "repo_id": args.repo_id,
        "cameras": list(cameras),
        "policy_names": list(_POLICY_NAMES),
        "fps": fps,
        "num_episodes": int(meta.total_episodes),
        "num_frames": num_frames,
        "num_pads": num_pads,
        "processed_episodes": episodes if args.episodes else "all",
        "fit_episodes": fit_episodes if args.fit_episodes else "all processed",
        "horizons": list(args.horizons),
        "resize_mode": "center_crop",
        "encoder": "fastvit_t12",
        "encoder_weights": str(pathlib.Path(args.encoder_weights).expanduser()),
        "encoder_input": _ENCODER_INPUT,
        "feature_dim": int(feature_dim),
        "pca_dim": int(pca_dim),
        "pca_explained_variance": explained,
        "pixel_size": args.pixel_size,
        "y_delta_rms": y_delta_rms,
        "layout": {
            "episode_offsets.npy": "int64 [E+1], episode e -> global frames offsets[e]:offsets[e+1]",
            "feat_tac.npy": "float16 [N, S, feature_dim], frozen encoder output",
            "z_tac.npy": "float16 [N, S, pca_dim], (feat - pca_mean) @ pca_components.T / sqrt(pca_eigvals)",
            "pixel_field.npy": "uint8 [N, S, pixel_size, pixel_size, 3]",
        },
    }
    np.save(args.out / "episode_offsets.npy", offsets)
    (args.out / "meta.json").write_text(json.dumps(written, indent=2))
    print(f"wrote {args.out} ({sum(p.stat().st_size for p in args.out.iterdir()) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
