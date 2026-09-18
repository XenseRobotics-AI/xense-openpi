"""Does the trained LTP head use the tactile tokens?  L_tac on the same frames under
real / tac-shuffle / null / tac-zero, paired noise & tau (same rng), head enabled."""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "test")
from tactile_counterfactual import layer_probe as lp
from tactile_counterfactual import runner as _runner

from openpi.models import pi0_tactile_fastvit as ptf
from openpi.shared import nnx_utils
from openpi.training import config as _config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="pi05_base_bi_flexiv_bottle_sorting_0917_fastvit_ltp_h100")
    p.add_argument(
        "--checkpoint-dir", default="checkpoints/pi05_base_bi_flexiv_bottle_sorting_0917_fastvit_ltp_h100/10000"
    )
    p.add_argument("--labels-dir", default="assets/tactile_future_labels/bottle-sorting-0810")
    p.add_argument("--frames", default="outputs/tactile_layer_probes/sensitivity/ltp10k_512/frames.json")
    p.add_argument("--out", default="outputs/tactile_layer_probes/head_shuffle/ltp10k")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    CFG = args.config_name
    CKPT = pathlib.Path(args.checkpoint_dir).resolve()
    LABELS = args.labels_dir
    FRAMES = args.frames
    OUT = pathlib.Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    CONDS = ["real", "tac-shuffle", "null", "tac-zero"]
    SEEDS = args.seeds

    tc = _config.get_config(CFG)
    mc = dataclasses.replace(tc.model, use_cudnn_attention=False)
    assert mc.tactile_future_layer == 10
    tc = dataclasses.replace(tc, model=mc)
    data_config = _runner.resolve_data_config(tc, None, CKPT)
    dataset = lp.ProbeDataset(data_config.repo_id, data_config, action_horizon=mc.action_horizon)
    model, _ = _runner.load_model(tc, CKPT)
    assert model.has_tactile_future_head
    store = lp.FutureTactileStore(LABELS, mc.tactile_future_horizons, target="latent")
    fr = json.load(open(FRAMES))
    refs = lp.FrameRefs(
        episode=np.array(fr["episode"]),
        frame=np.array(fr["frame"]),
        batch_size=int(fr["batch_size"]),
        contact=np.array(fr["contact"]),
    )
    tactile_keys = (
        [k for k in dataset[0]["image"].keys() if "tactile" in k] if hasattr(dataset, "__getitem__") else None
    )
    loss_fn = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))
    n = len(refs)
    res = {
        c: {
            "tac": np.zeros((len(SEEDS), n), np.float32),
            "flow": np.zeros((len(SEEDS), n), np.float32),
            "tac_by_time": np.zeros((len(SEEDS), refs.num_batches, 4), np.float32),
        }
        for c in CONDS
    }
    time_all = np.zeros((len(SEEDS), n), np.float32)
    t0 = time.time()
    for b, batch in enumerate(lp.iter_batches(dataset, refs, num_workers=4)):
        rows = slice(int(batch.index[0]), int(batch.index[-1]) + 1)
        obs = batch.observation
        if tactile_keys is None:
            tactile_keys = [k for k in obs.images if "tactile" in k]
        z = np.stack([store.targets(int(e), int(f))[0] for e, f in zip(batch.episode, batch.frame)])
        m = np.stack([store.targets(int(e), int(f))[1] for e, f in zip(batch.episode, batch.frame)])
        aux = {ptf.FUTURE_TACTILE_Z: jnp.asarray(z, jnp.float32), ptf.FUTURE_TACTILE_MASK: jnp.asarray(m)}
        for si, seed in enumerate(SEEDS):
            rng = jax.random.key(1000 + seed)
            for c in CONDS:
                v = dataclasses.replace(lp.make_variant(obs, c, tactile_keys), aux_targets=aux)
                out = loss_fn(rng, v, jnp.asarray(batch.actions, jnp.float32), train=False)
                tac = np.asarray(out["tac"])
                mask = np.asarray(out["tac_mask"])
                res[c]["tac"][si, rows] = (tac * mask).reshape(tac.shape[0], -1).sum(1) / mask.reshape(
                    tac.shape[0], -1
                ).sum(1)
                res[c]["flow"][si, rows] = np.asarray(out["flow"]).mean(1)
                res[c]["tac_by_time"][si, b] = np.asarray(out["tac_by_time"])
        if b % 4 == 0:
            print(f"[{rows.stop}/{n}] {time.time() - t0:.0f}s", flush=True)

    contact = refs.contact
    summary = {}
    for c in CONDS:
        t = res[c]["tac"]
        f = res[c]["flow"]
        summary[c] = {
            "tac": float(t.mean()),
            "tac_contact": float(t[:, contact].mean()),
            "tac_noncontact": float(t[:, ~contact].mean()),
            "tac_by_time": np.nanmean(res[c]["tac_by_time"], axis=(0, 1)).tolist(),
            "flow": float(f.mean()),
            "tac_per_seed": t.mean(1).tolist(),
        }
    json.dump(summary, open(OUT / "results.json", "w"), indent=1)
    lines = [
        f"# LTP head under tactile counterfactuals ({CKPT.parent.name}/{CKPT.name}, {n} frames x {len(SEEDS)} rng seeds, paired tau/noise)",
        "",
        "| condition | L_tac | contact | non-contact | by tau bin (0-.25,.25-.5,.5-.75,.75-1) | L_flow |",
        "|---|---|---|---|---|---|",
    ]
    for c in CONDS:
        s = summary[c]
        lines.append(
            f"| {c} | {s['tac']:.4f} | {s['tac_contact']:.4f} | {s['tac_noncontact']:.4f} | "
            + ", ".join(f"{x:.4f}" for x in s["tac_by_time"])
            + f" | {s['flow']:.3e} |"
        )
    lines += ["", "zero-prediction baseline = 1.0; persistence-copy baseline on 2048-frame val rows = 0.487"]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
