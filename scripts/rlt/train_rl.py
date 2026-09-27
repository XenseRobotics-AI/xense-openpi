"""RLT phase two: online chunked-TD RL on the real robot.

Freezes the VLA (reference chunks and prefix features) and the phase-one token
encoder, and trains the MLP actor and twin-Q critic from labeled critical
phases collected round by round (see ``openpi.rlt.collector``). This process
listens on ``rl.listen``; the robot host dials in with the RLT mode of
``examples/bi_flexiv_rizon4_rt``.

    uv run scripts/rlt/train_rl.py <rlt_config> --exp-name <run> [--resume | --overwrite]

Outputs under ``<checkpoint_base_dir>/<config>/<exp-name>/``:

- ``rl/<round>/learner.pkl``: checkpoints (weights, optimizers, replay), pruned by ``rl.keep_period``;
- ``transitions/``: with ``rl.dump_transitions``, each round's committed transitions
  (``round_<n>.npz``), first-chunk diagnostics, and ``events.jsonl`` mirroring the W&B metrics.

W&B logs three axes: ``round/*``, ``update/*`` (every gradient update) and ``chunk/*`` (every
executed chunk, with the actor's output - or, while the VLA drives, its shadow proposal -
against the VLA reference). A resumed run continues the same W&B run. A lost robot connection
discards the current round and waits for the host to reconnect.
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
from openpi.rlt import diagnostics as _diagnostics
from openpi.rlt import env_protocol
from openpi.rlt import features as _features
from openpi.rlt import learner as _learner
from openpi.rlt import vla as _vla


def _latest(directory: pathlib.Path) -> pathlib.Path | None:
    steps = [p for p in directory.iterdir() if p.name.isdigit()] if directory.is_dir() else []
    return max(steps, key=lambda p: int(p.name), default=None)


def _summary(config: _rlt_config.RLConfig, learner: _learner.Learner, result: dict, infos: list[dict]) -> str:
    """The operator's end-of-round block, one fact per line."""
    m, c = result["metrics"], learner.counters
    n = {key: int(m.get(key, 0)) for key in m.keys() | {"success", "failure", "discards", "steps", "human_steps"}}
    warm_up = (
        "reached"
        if learner.warmed_up
        else _diagnostics.warmup_estimate(config.warm_up, c.transitions, c.phases, c.phase_steps)
    )
    lines = [
        f"==== RLT round {c.rounds}/{config.total_rounds} ====",
        (
            f"outcome : success={n['success']} failure={n['failure']} discards={n['discards']} "
            f"steps={n['steps']} human_steps={n['human_steps']}"
        ),
        f"actor   : {n.get('actor_chunks', 0)}/{n.get('chunks', 0)} chunks routed to the actor",
        (
            f"actions : out_of_range={n.get('out_of_range', 0)} gripper_clips={n.get('gripper_clips', 0)} "
            f"rot6d_fallbacks={n.get('rot6d_fallbacks', 0)}"
        ),
        f"data    : +{len(result['rows'])} transitions, replay {len(learner.replay)}/{config.buffer_size}, total {c.transitions}",
        f"warm-up : {config.warm_up} transitions, {warm_up}",
        (
            f"updates : this round {len(infos)} | total critic {c.critic_updates} actor {c.actor_updates} | "
            f"pending {learner.pending_updates()}"
        ),
    ]
    if infos:
        lines.append("losses (mean over this round's updates):")
        for key in ("critic_loss", "actor_loss", "bc_loss", "q_pi", "q_data", "q_target", "bc_q_grad_cosine"):
            values = [info[key] for info in infos if key in info]
            if values:
                lines.append(f"  {key:<18}: {np.mean(values):.5g}")
    return "\n".join(lines)


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
    dump_dir = config.checkpoint_dir / "transitions" if rl.dump_transitions else None
    if rl_dir.exists() and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"{rl_dir} exists; pass --resume or --overwrite.")
        shutil.rmtree(rl_dir)
        if dump_dir is not None:
            shutil.rmtree(dump_dir, ignore_errors=True)
    rl_dir.mkdir(parents=True, exist_ok=True)
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    frozen = _vla.resolve(config.token_training)
    prompt = frozen.check_default_prompt()
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

    space = extractor.space
    logging.info(
        "Online RLT: VLA %s (%s), token %s, prompt %r | sampler %s, %d denoising steps | state %d, action %d, "
        "grippers %s, rot6d %s | C=%d, reference %d, stride %d, warm_up %d, UTD %d, actor every %d critic updates",
        frozen.train_config.name,
        frozen.checkpoint_dir,
        token_checkpoint,
        prompt,
        "training-time RTC (no frozen prefix)" if frozen.train_config.model.enable_training_time_rtc else "standard",
        rl.num_steps,
        space.state_dim,
        space.action_dim,
        space.gripper_dims,
        space.rot6d_blocks,
        rl.num_action_chunks,
        rl.ref_num_action_chunks,
        rl.replay_stride,
        rl.warm_up,
        rl.utd,
        rl.critic_actor_ratio,
    )
    logging.info("Discount horizon: %s", _diagnostics.discount_horizon(rl.gamma, rl.num_action_chunks))

    # Reattach to the same W&B run on resume, so the curves continue.
    wandb_id_path = rl_dir / "wandb_id.txt"
    run = wandb.init(
        project=config.project_name,
        name=f"{config.name}/{config.exp_name}/rl",
        id=wandb_id_path.read_text().strip() if args.resume and wandb_id_path.exists() else None,
        config=dataclasses.asdict(config),
        resume="allow" if args.resume else None,
        mode=None if config.wandb_enabled else "disabled",
    )
    if config.wandb_enabled:
        wandb_id_path.write_text(run.id)
    logger = _diagnostics.RunLogger(run if config.wandb_enabled else None, dump_dir)

    host, port = rl.listen.rsplit(":", 1)
    env = env_protocol.RemoteEnv(host, int(port), state_dim=space.state_dim, action_dim=space.action_dim)
    collector = _collector.Collector(env, extractor, learner, logger=logger, dump_dir=dump_dir)
    totals = collections.Counter()

    def save() -> None:
        _learner.save_round(learner, rl_dir, {"binding": binding})

    try:
        while learner.counters.rounds < rl.total_rounds:
            try:
                result = collector.run_round()
            except env_protocol.EnvConnectionLostError as exc:
                logging.warning("Robot connection lost (%s); the round is discarded.", exc)
                continue
            learner.commit(result["rows"])
            learner.counters.phases += len(result["phase_steps"])
            learner.counters.phase_steps += sum(result["phase_steps"])
            env.status(
                f"Round {learner.counters.rounds + 1}: +{len(result['rows'])} transitions; "
                f"training {learner.pending_updates()} updates (the robot idles)..."
            )
            first_update = learner.counters.critic_updates
            infos = learner.train()
            for index, info in enumerate(infos, start=first_update + 1):
                logger.log("update", index, info)
            learner.counters.rounds += 1
            if dump_dir is not None:
                _diagnostics.dump_transitions(
                    dump_dir / f"round_{learner.counters.rounds:05d}.npz",
                    result["rows"],
                    critic_updates_after=learner.counters.critic_updates,
                    actor_updates_after=learner.counters.actor_updates,
                )
            totals.update({key: value for key, value in result["metrics"].items() if key != "actor_chunk_ratio"})
            logger.log(
                "round",
                learner.counters.rounds,
                {
                    **result["metrics"],
                    **{f"total_{key}": value for key, value in totals.items()},
                    **dataclasses.asdict(learner.counters),
                    "replay_size": len(learner.replay),
                    "warmed_up": learner.warmed_up,
                    "pending_updates": learner.pending_updates(),
                    **{
                        f"train_{key}": np.mean([i[key] for i in infos if key in i])
                        for key in {k for i in infos for k in i}
                    },
                },
            )
            summary = _summary(rl, learner, result, infos)
            logging.info("\n%s", summary)
            env.status(summary)
            if learner.counters.rounds % rl.save_interval == 0:
                save()
    finally:
        save()
        env.close()
        wandb.finish()


if __name__ == "__main__":
    main()
