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

import numpy as np

from openpi.rlt import critical_trace as _critical_trace
from openpi.rlt import env_protocol
from openpi.rlt import features as _features
from openpi.rlt import learner as _learner
from openpi.rlt import replay as _replay


class Collector:
    def __init__(self, env: env_protocol.RemoteEnv, extractor: _features.FeatureExtractor, learner: _learner.Learner):
        self.env = env
        self.extractor = extractor
        self.learner = learner
        self.config = learner.config
        self._episode_id = 0  # one id per critical phase (recording window)

    def run_round(self) -> dict:
        """Collect one round; returns ``{"rows": uncommitted replay rows, "metrics": {...}}``."""
        config = self.config
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
        tally = collections.Counter()
        round_end = False
        while not round_end:
            # The actor only ever runs inside an open window, and only once replay has warmed up.
            use_actor = recording and self.learner.warmed_up
            actions = self.learner.act(features) if use_actor else features["ref_exec"][:horizon]
            reply = self.env.request({"op": "chunk", "actions": actions})
            tally["chunks"] += 1
            tally["actor_chunks"] += use_actor
            captures = {int(c["step"]): c["obs"] for c in reply["captures"]}

            for segment in reply["segments"]:
                if segment["recording"] and trace is None:
                    self._episode_id += 1
                    trace = _critical_trace.CriticalTrace(horizon, config.replay_stride)
                    trace.add_features(0, features)
                elif not segment["recording"] and trace is not None:
                    logging.info("Window closed without a label; dropping %d steps.", len(trace))
                    trace = None
                human = np.asarray(segment["human"], bool)
                executed = np.asarray(segment["executed"], np.float32).reshape(len(human), -1)
                tally["steps"] += len(human)
                tally["human_steps"] += int(human.sum())
                features = self.extractor.extract(segment["obs"])
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
                    logging.info("Operator discarded the round's data (%d rows).", len(rows))
                    rows.clear()
                    trace = None
                    tally["discards"] += 1
                elif segment["label"] is not None and trace is not None:
                    rows += self._close_phase(trace, segment["label"])
                    tally[segment["label"]] += 1
                    trace = None
                round_end = round_end or bool(segment["round_end"])
            recording = bool(reply["segments"][-1]["recording_next"])

        metrics = {**{key: float(value) for key, value in tally.items()}, "rows": float(len(rows))}
        metrics["actor_chunk_ratio"] = tally["actor_chunks"] / max(tally["chunks"], 1)
        return {"rows": rows, "metrics": metrics}

    def _close_phase(self, trace: _critical_trace.CriticalTrace, label: str) -> list[dict]:
        """Turn a labeled phase into replay rows (terminal reward 1 for success, 0 for failure)."""
        trace.set_terminal_reward(float(label == "success"))
        missing = trace.missing_feature_indices()
        if any(not trace.has_observation(i) for i in missing):
            logging.warning("Labeled phase is missing stride captures; dropped.")
            return []
        batch = self.config.replay_feature_batch_size
        for start in range(0, len(missing), batch):
            indices = missing[start : start + batch]
            observations = [trace.observation(i) for i in indices]
            for index, features in zip(indices, self.extractor.extract_batch(observations, pad_to=batch), strict=False):
                trace.add_features(index, features)
        windows = trace.windows()
        if not windows:
            logging.warning(
                "Labeled phase of %d steps is shorter than one %d-step window; dropped.", len(trace), trace.horizon
            )
        space = self.learner.space
        return [
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
            }
            for window in windows
        ]
