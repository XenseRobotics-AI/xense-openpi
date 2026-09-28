"""One operator round of online RLT collection (synchronous chunks, windowed labels).

A round runs from reset to the operator's round end. Chunk by chunk, the
server extracts features at the current observation, executes either the VLA
reference or - inside an open recording window once replay has warmed up - the
actor, and appends what actually ran (policy or human steps) to the window's
``CriticalTrace``. A success/failure label closes the window: its sliding
windows become replay rows. Unlabeled windows are dropped; a discard drops the
whole round's rows. Rows reach replay only when the round ends, so a lost
connection or a discard never leaves a partial round in replay.
"""

from __future__ import annotations

import collections
import logging
import pathlib
import time

import numpy as np

from openpi.rlt import critical_trace as _critical_trace
from openpi.rlt import diagnostics as _diagnostics
from openpi.rlt import env_protocol
from openpi.rlt import features as _features
from openpi.rlt import learner as _learner
from openpi.rlt import replay as _replay


class Collector:
    def __init__(
        self,
        env: env_protocol.RemoteEnv,
        extractor: _features.FeatureExtractor,
        learner: _learner.Learner,
        *,
        logger: _diagnostics.RunLogger | None = None,
        dump_dir: pathlib.Path | None = None,
    ):
        self.env = env
        self.extractor = extractor
        self.learner = learner
        self.config = learner.config
        self.logger = logger or _diagnostics.RunLogger(None)
        self.dump_dir = dump_dir
        self._episode_id = 0  # one id per critical phase (recording window)

    def run_round(self) -> dict:
        """Collect one round.

        Returns ``{"rows": uncommitted replay rows, "phase_steps": [steps per kept labeled phase],
        "metrics": {...}}``.
        """
        config = self.config
        space = self.learner.space
        horizon = config.num_action_chunks
        reply = self.env.request(
            {
                "op": "reset",
                "takeover_position_m": config.takeover_position_m,
                "takeover_rotation_deg": config.takeover_rotation_deg,
                "capture_stride": config.replay_stride,
            }
        )
        features = self.extractor.extract(reply["obs"])
        recording = bool(reply["recording"])
        trace: _critical_trace.CriticalTrace | None = None
        rows: list[dict] = []
        phase_steps: list[int] = []
        tally = collections.Counter()
        round_end = False
        while not round_end:
            # The actor only ever runs inside an open window, and only once replay has warmed up.
            use_actor = recording and self.learner.warmed_up
            reference = features["ref_chunk"][:horizon]
            if use_actor:
                decision, actions = self.learner.act(features)
                chunk_log = _prefixed(
                    "actor", _diagnostics.output_metrics(space, decision, reference, features["state"])
                )
                chunk_log["actor_rot6d_fallbacks"] = space.diagnose(decision, features["state"], normalized=True)[
                    "rot6d_fallbacks"
                ]
            else:
                actions = features["ref_exec"][:horizon]
                # What the actor would have done, to watch it while the VLA drives.
                shadow = self.learner.mean(features)
                chunk_log = _prefixed(
                    "shadow", _diagnostics.output_metrics(space, shadow, reference, features["state"])
                )
            if tally["chunks"] == 0:
                self._initial_action_diagnostic(features, actions, use_actor=use_actor)
            started = time.monotonic()
            reply = self.env.request({"op": "chunk", "actions": actions, "source": "actor" if use_actor else "vla"})
            execution_s = time.monotonic() - started
            tally["chunks"] += 1
            tally["actor_chunks"] += use_actor
            captures = {int(c["step"]): c["obs"] for c in reply["captures"]}

            chunk = collections.Counter()
            for segment in reply["segments"]:
                if segment["recording"] and trace is None:
                    self._episode_id += 1
                    trace = _critical_trace.CriticalTrace(horizon, config.replay_stride)
                    trace.add_features(0, features)
                elif not segment["recording"] and trace is not None:
                    logging.info("Window closed without a label; dropping %d steps.", len(trace))
                    trace = None
                human = np.asarray(segment["human"], bool)
                executed = np.asarray(segment["executed"], np.float32).reshape(len(human), space.action_dim)
                chunk.update(
                    steps=len(human), human_steps=int(human.sum()), recording_steps=len(human) * segment["recording"]
                )
                if len(human):
                    chunk.update(space.diagnose(executed, features["state"]))
                extract_started = time.monotonic()
                features = self.extractor.extract(segment["obs"])
                chunk["feature_ms"] += 1000 * (time.monotonic() - extract_started)
                if trace is not None:
                    source = np.where(
                        human, _replay.SOURCE_HUMAN, _replay.SOURCE_ACTOR if use_actor else _replay.SOURCE_VLA
                    )
                    start = len(trace)
                    trace.extend(executed, np.zeros(len(human)), source, actor_enabled=use_actor)
                    trace.add_features(len(trace), features)
                    for step in range(start + 1, len(trace)):
                        if step in captures:
                            trace.add_observation(step, captures[step])
                if segment["discard"]:
                    self.env.status(f"Discard: dropped this round's {len(rows)} labeled transitions.")
                    rows.clear()
                    phase_steps.clear()
                    trace = None
                    tally["discards"] += 1
                elif segment["label"] is not None and trace is not None:
                    phase_rows, dropped = self._close_phase(trace, segment["label"])
                    if phase_rows:
                        rows += phase_rows
                        phase_steps.append(len(trace))
                    # Confirm on the operator's terminal what the label produced, before the next chunk.
                    self.env.status(
                        f"{segment['label'].capitalize()} labeled: {len(trace)}-step phase -> "
                        + (
                            f"dropped ({dropped})"
                            if dropped
                            else f"+{len(phase_rows)} transitions ({len(rows)} this round)"
                        )
                    )
                    tally[segment["label"]] += 1
                    trace = None
                round_end = round_end or bool(segment["round_end"])
            recording = bool(reply["recording_next"])

            tally.update({k: v for k, v in chunk.items() if k != "feature_ms"})
            self.learner.counters.chunks += 1
            self.logger.log(
                "chunk",
                self.learner.counters.chunks,
                {
                    **chunk,
                    **chunk_log,
                    "round": self.learner.counters.rounds + 1,
                    "use_actor": use_actor,
                    "segments": len(reply["segments"]),
                    "execution_s": execution_s,
                },
            )

        metrics = {**{key: float(value) for key, value in tally.items()}, "rows": float(len(rows))}
        metrics["actor_chunk_ratio"] = tally["actor_chunks"] / max(tally["chunks"], 1)
        return {"rows": rows, "phase_steps": phase_steps, "metrics": metrics}

    def _initial_action_diagnostic(self, features: dict, actions: np.ndarray, *, use_actor: bool) -> None:
        """Log (and dump) the round's first chunk against the robot state; catches frame or unit mix-ups."""
        state = features["state"]
        distances = [
            1000 * np.linalg.norm(actions[:, a : a + 3] - state[a : a + 3], axis=-1).max()
            for a, _ in _diagnostics.arm_blocks(self.learner.space)
        ]
        logging.info(
            "First chunk (%s): max xyz distance from the current TCP %s mm; first step %s",
            "actor" if use_actor else "VLA",
            ", ".join(f"{d:.2f}" for d in distances),
            np.array2string(actions[0], precision=4, max_line_width=250),
        )
        if self.dump_dir is not None:
            np.savez(
                self.dump_dir / f"initial_actions_round{self.learner.counters.rounds + 1}.npz",
                state=state,
                proposed=actions,
                vla_reference=features["ref_exec"],
                use_actor=use_actor,
            )

    def _close_phase(self, trace: _critical_trace.CriticalTrace, label: str) -> tuple[list[dict], str | None]:
        """Turn a labeled phase into replay rows (terminal reward 1 for success, 0 for failure).

        Returns the rows and, when the phase yields none, why.
        """
        trace.set_terminal_reward(float(label == "success"))
        missing = trace.missing_feature_indices()
        if any(not trace.has_observation(i) for i in missing):
            logging.warning("Labeled phase is missing stride captures; dropped.")
            return [], "stride captures missing"
        batch = self.config.replay_feature_batch_size
        for start in range(0, len(missing), batch):
            indices = missing[start : start + batch]
            observations = [trace.observation(i) for i in indices]
            for index, features in zip(indices, self.extractor.extract_batch(observations, pad_to=batch), strict=False):
                trace.add_features(index, features)
        windows = trace.windows()
        if not windows:
            return [], f"shorter than one {trace.horizon}-step window"
        space = self.learner.space
        rows = [
            {
                "curr_obs": window.features,
                "next_obs": window.next_features,
                "actions": np.asarray(space.encode(window.executed, window.features["state"])),
                "chunk_rewards": window.rewards,
                "intervention_mask": window.human,
                "action_source": window.source,
                "terminated": window.terminal,
                "success": label == "success",
                "actor_enabled": window.actor_enabled,
                "episode_id": self._episode_id,
                "round_id": self.learner.counters.rounds,
                # For the transition dump only (replay ignores them): what the robot ran, and the
                # raw VLA reference before any normalization or clipping.
                "executed_actions": window.executed,
                "ref_exec": window.features["ref_exec"],
            }
            for window in windows
        ]
        return rows, None


def _prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}/{key}": value for key, value in values.items()}
