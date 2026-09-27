"""RLT phase two: online chunked-TD RL on the real robot.

Freezes the VLA (reference chunks and prefix features) and the phase-one token
encoder, and trains the MLP actor and twin-Q critic from labeled critical
phases collected round by round (see ``openpi.rlt.collector``). This process
listens on ``rl.listen``; the robot host dials in with the RLT mode of
``examples/bi_flexiv_rizon4_rt``.

    uv run scripts/rlt/train_rl.py <rlt_config> --exp-name <run> [--resume | --overwrite]

Checkpoints: ``<checkpoint_base_dir>/<config>/<exp-name>/rl/<round>/learner.pkl``.
A lost robot connection discards the current round and waits for the host to reconnect.
"""

import argparse
import collections
import dataclasses
import logging
import pathlib
import shutil

import numpy as np
import wandb

from openpi.rlt import collector as _collector
from openpi.rlt import config as _rlt_config
from openpi.rlt import env_protocol
from openpi.rlt import features as _features
from openpi.rlt import learner as _learner
from openpi.rlt import vla as _vla


def _latest(directory: pathlib.Path) -> pathlib.Path | None:
    steps = [p for p in directory.iterdir() if p.name.isdigit()] if directory.is_dir() else []
    return max(steps, key=lambda p: int(p.name), default=None)


def _mean(infos: list[dict[str, float]]) -> dict[str, float]:
    keys = {key for info in infos for key in info}
    return {f"train/{key}": float(np.mean([info[key] for info in infos if key in info])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="RLT config name (configs/rlt/<name>.yaml)")
    parser.add_argument("--exp-name", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    config = dataclasses.replace(_rlt_config.get_config(args.config), exp_name=args.exp_name)
    rl = config.rl
    rl_dir = config.rl_checkpoint_dir
    if rl_dir.exists() and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"{rl_dir} exists; pass --resume or --overwrite.")
        shutil.rmtree(rl_dir)

    frozen = _vla.resolve(config.token_training)
    token_checkpoint = _features.resolve_token_checkpoint(rl.token_checkpoint or config.token_checkpoint_dir)
    extractor = _features.FeatureExtractor.from_vla(
        frozen, token_checkpoint, rl, num_steps=rl.num_steps, seed=config.seed
    )
    learner = _learner.Learner(rl, extractor.space, z_dim=extractor.z_dim, seed=config.seed)
    binding = {"vla": frozen.vla_identity(), "token_checkpoint": str(token_checkpoint)}
    if args.resume and (latest := _latest(rl_dir)) is not None:
        saved = learner.restore(latest)
        if saved["binding"]["vla"] != binding["vla"]:
            raise ValueError(f"{latest} was trained against a different VLA checkpoint.")
        if saved["binding"]["token_checkpoint"] != binding["token_checkpoint"]:
            logging.warning(
                "Resuming with token checkpoint %s; %s used %s.",
                token_checkpoint,
                latest,
                saved["binding"]["token_checkpoint"],
            )
        logging.info("Resumed %s: round %d, replay %d", latest, learner.counters.rounds, len(learner.replay))

    wandb.init(
        project=config.project_name,
        name=f"{config.name}/{config.exp_name}/rl",
        config=dataclasses.asdict(config),
        resume="allow" if args.resume else None,
        mode=None if config.wandb_enabled else "disabled",
    )
    host, port = rl.listen.rsplit(":", 1)
    env = env_protocol.RemoteEnv(
        host, int(port), state_dim=extractor.space.state_dim, action_dim=extractor.space.action_dim
    )
    collector = _collector.Collector(env, extractor, learner)
    totals = collections.Counter()

    def save() -> None:
        learner.save(rl_dir / str(learner.counters.rounds), {"binding": binding})

    try:
        while learner.counters.rounds < rl.total_rounds:
            try:
                result = collector.run_round()
            except env_protocol.EnvConnectionLostError as exc:
                logging.warning("Robot connection lost (%s); the round is discarded.", exc)
                continue
            learner.commit(result["rows"])
            env.status(
                f"Round {learner.counters.rounds + 1}: +{len(result['rows'])} transitions, replay "
                f"{len(learner.replay)}/{rl.buffer_size} (warm_up {rl.warm_up}); training {learner.pending_updates()} updates..."
            )
            infos = learner.train()
            learner.counters.rounds += 1
            totals.update({key: value for key, value in result["metrics"].items() if key != "actor_chunk_ratio"})
            counters = dataclasses.asdict(learner.counters)
            wandb.log(
                {
                    **{f"round/{key}": value for key, value in result["metrics"].items()},
                    **{f"total/{key}": value for key, value in totals.items()},
                    **counters,
                    "replay_size": len(learner.replay),
                    **_mean(infos),
                },
                step=learner.counters.rounds,
            )
            env.status(
                f"Round {learner.counters.rounds}/{rl.total_rounds} done: {len(infos)} updates "
                f"(critic {counters['critic_updates']}, actor {counters['actor_updates']}), actor "
                f"{'ON' if learner.warmed_up else 'OFF (replay below warm_up)'} for the next windows."
            )
            if learner.counters.rounds % rl.save_interval == 0:
                save()
    finally:
        save()
        env.close()
        wandb.finish()


if __name__ == "__main__":
    main()
