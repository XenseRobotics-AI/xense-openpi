import math
import threading
import time
from collections import deque
from typing import override

import numpy as np  # noqa: F401
from lerobot.utils.robot_utils import get_logger

from xense_client import base_policy as _base_policy
from xense_client.action_queue import ActionQueue
from xense_client.latency_tracker import LatencyTracker

logger = get_logger("RTCActionChunkBroker")


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return actions using an RTC-style ActionQueue.

    This broker runs a background thread to fetch action chunks from the policy
    and maintains a thread-safe queue of actions. It handles:
    - Asynchronous action fetching
    - Latency tracking (basic)
    - Action queue management (merging/replacing based on delay)

    Delay estimation strategy (important for smoothness):
        The model freezes the FIRST ``inference_delay`` actions of the newly
        generated chunk as a copy of ``prev_chunk_left_over`` (the frozen
        prefix). At merge time the queue truncates at ``real_delay`` (the
        steps actually consumed during inference). For smooth transitions we
        need ``real_delay <= estimated_delay`` so the truncation point falls
        inside the frozen prefix (where new_actions == old_actions byte for
        byte). To achieve this, we take the MAX of the recent real delays
        and add ``delay_margin`` on top, rather than using just the last
        observed delay.

    Args:
        policy: The underlying policy (e.g., WebsocketClientPolicy).
        frequency_hz: The control frequency in Hz.
        action_queue_size_to_get_new_actions: Threshold to request new actions.
        rtc_enabled: Whether to enable RTC mode (replace queue) or append mode.
        execution_horizon: Informational execution horizon forwarded to the
            policy. Not used for queue trigger logic.
        blend_steps: Number of steps to linearly blend old/new actions at the
            merge point. 0 disables blending.
        default_delay: Fallback ``inference_delay`` used during warmup and
            before any real delay has been measured.
        delay_margin: Safety margin added on top of ``max(recent_real_delays)``
            when computing the delay sent to the model. Larger values are
            safer (guarantee real_delay <= estimated_delay) but shrink the
            usable denoised postfix. 2 is a reasonable default for latencies
            that jitter by a couple of control steps.
        delay_history_size: Number of past real delays tracked for the max.
        dry_run: If True, each infer() includes ``rtc_metrics`` (delay, round-trip ms, etc.).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        frequency_hz: float = 50.0,
        action_queue_size_to_get_new_actions: int = 20,
        rtc_enabled: bool = True,
        execution_horizon: int = 20,
        blend_steps: int = 0,
        default_delay: int = 4,  # Default inference_delay for warmup and fallback
        delay_margin: int = 2,  # Safety margin added to the estimated delay
        delay_history_size: int = 10,  # Number of recent delays tracked
        dry_run: bool = False,
        delta_state_dim: int = 0,
    ):
        self._policy = policy
        self._frequency_hz = frequency_hz
        self._time_per_chunk = 1.0 / frequency_hz
        self._action_queue_size_to_get_new_actions = action_queue_size_to_get_new_actions
        self._execution_horizon = execution_horizon
        self._default_delay = default_delay
        self._delay_margin = max(0, int(delay_margin))
        # Retained for compatibility with existing call sites. RTC prefixes are
        # now transformed on the policy server using the training input
        # pipeline, so the client no longer attempts to re-base delta actions.
        self._delta_state_dim = int(delta_state_dim)

        self._action_queue = ActionQueue(rtc_enabled=rtc_enabled, blend_steps=blend_steps)
        self._latency_tracker = LatencyTracker()
        self._dry_run = dry_run
        self._metrics_lock = threading.Lock()
        self._last_inference_metrics: dict | None = None
        self._inference_seq = 0
        self._latest_obs: dict | None = None
        self._latest_obs_lock = threading.Lock()

        # Track last real_delay (kept for metrics/logging).
        self._last_real_delay: int | None = None
        # Rolling window of recent real delays; we take the max of this as
        # the base estimate so that transient latency spikes dominate.
        self._recent_real_delays: deque[int] = deque(maxlen=max(1, int(delay_history_size)))

        # Track warmup state: first inference is for JIT, second is for real execution
        self._warmup_done = False
        self._warmup_prev_chunk: np.ndarray | None = None  # Store first inference result for second inference

        self._stop_event = threading.Event()
        self._first_inference_done = threading.Event()  # Signal when actions are ready for execution (after warmup)
        self._thread = threading.Thread(target=self._get_actions_loop, daemon=True)
        self._thread_started = False

    def _start_thread_if_needed(self):
        if not self._thread_started:
            self._thread.start()
            self._thread_started = True
            logger.info("RTCActionChunkBroker background thread started")

    def _get_actions_loop(self):
        while not self._stop_event.is_set():
            try:
                # Check if we need more actions
                if self._action_queue.qsize() <= self._action_queue_size_to_get_new_actions:
                    # Get latest observation
                    with self._latest_obs_lock:
                        obs = self._latest_obs

                    if obs is None:
                        # No observation yet, wait a bit
                        time.sleep(0.001)
                        continue

                    # Prepare for inference
                    current_time = time.perf_counter()
                    action_index_before_inference = self._action_queue.get_action_index()

                    # Snapshot of the prefix region we send to the model, used
                    # after inference to verify the frozen-prefix invariant.
                    # Populated only in normal operation (not during warmup).
                    prefix_sent: np.ndarray | None = None

                    # Determine prev_chunk_left_over and estimated_delay based on warmup state
                    if not self._warmup_done:
                        if self._warmup_prev_chunk is None:
                            # ===== WARMUP PHASE 1: First inference (JIT compilation) =====
                            # Send None, use default_delay, result will NOT be executed
                            prev_chunk_left_over = None
                            estimated_delay_steps = self._default_delay
                            logger.info(
                                f"🔥 WARMUP Phase 1: First inference (JIT). "
                                f"Sending prev_chunk_left_over=None, inference_delay={estimated_delay_steps}"
                            )
                        else:
                            # ===== WARMUP PHASE 2: Second inference (with prev_chunk from phase 1) =====
                            # Use last N actions from warmup result as prev_chunk_left_over
                            # This ensures consistent shape for JAX JIT
                            prev_chunk_left_over = self._warmup_prev_chunk[
                                -self._action_queue_size_to_get_new_actions :
                            ]
                            estimated_delay_steps = self._default_delay
                            logger.info(
                                f"🔥 WARMUP Phase 2: Second inference with prev_chunk. "
                                f"Sending prev_chunk_left_over shape: {prev_chunk_left_over.shape}, "
                                f"inference_delay={estimated_delay_steps}"
                            )
                    else:
                        # ===== NORMAL OPERATION =====
                        # Conservative estimated_delay: max(recent real delays) + margin.
                        #
                        # Why conservative? The model freezes new_actions[0:estimated_delay]
                        # as an exact copy of prev_chunk_left_over. At merge time the queue
                        # truncates at real_delay. If real_delay <= estimated_delay the
                        # truncation point lies inside the frozen prefix, so the first
                        # action executed from the new chunk is identical to what the
                        # old queue would have returned -> zero discontinuity. Using the
                        # raw last delay (as before) makes real_delay > estimated_delay
                        # about half the time, which causes jitter.
                        if self._recent_real_delays:
                            base_delay = max(self._recent_real_delays)
                            estimated_delay_steps = base_delay + self._delay_margin
                        else:
                            estimated_delay_steps = self._default_delay

                        # Clamp to the amount of prefix actually available in the queue.
                        # The model will pad with zeros if the prefix is too short, so
                        # estimated_delay must not exceed qsize or we'd freeze zero
                        # actions as prefix.
                        current_qsize = self._action_queue.qsize()
                        if current_qsize > 0:
                            estimated_delay_steps = min(estimated_delay_steps, current_qsize)
                        estimated_delay_steps = max(0, estimated_delay_steps)

                        # Get leftover actions for RTC guidance
                        # Use fixed_length to prevent JAX recompilation on shape changes
                        prev_chunk_left_over = self._action_queue.get_left_over(
                            fixed_length=self._action_queue_size_to_get_new_actions
                        )
                        logger.info(
                            f"RTC: Starting inference. Queue size: {current_qsize}, "
                            f"estimated_delay={estimated_delay_steps} "
                            f"(recent max={max(self._recent_real_delays) if self._recent_real_delays else None}, "
                            f"margin={self._delay_margin}), "
                            f"prev_chunk shape: {prev_chunk_left_over.shape if prev_chunk_left_over is not None else None}"
                        )

                        # Snapshot the prefix region we are about to send so we
                        # can verify after inference whether the model actually
                        # froze it byte-for-byte. This is the only reliable
                        # frozen-prefix check: comparing against original_queue
                        # in action_queue.merge() is meaningless when
                        # delta_state_dim > 0 because the two sides live in
                        # different reference frames (delta w.r.t. prev_state
                        # vs delta w.r.t. cur_state).
                        if prev_chunk_left_over is not None and estimated_delay_steps > 0:
                            n_pref = min(estimated_delay_steps, len(prev_chunk_left_over))
                            if n_pref > 0:
                                prefix_sent = prev_chunk_left_over[:n_pref].copy()

                    results = self._policy.infer(
                        obs,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=estimated_delay_steps,
                        execution_horizon=self._execution_horizon,
                    )

                    # Calculate actual latency for next time
                    latency = time.perf_counter() - current_time
                    self._latency_tracker.add(latency)
                    inference_delay_steps = math.ceil(latency / self._time_per_chunk)

                    # Get actions
                    processed_actions = results.get("actions")

                    if processed_actions is None:
                        logger.error("Policy returned no 'actions' key")
                        continue

                    # RTC guidance now uses the same action space that is
                    # actually executed on the robot. The policy server
                    # converts that absolute prefix back into the model's
                    # training space using its normal input transforms before
                    # sampling the next chunk. That avoids client-side
                    # assumptions about delta masks, normalization, or state
                    # ordering.
                    queued_actions = processed_actions

                    # End-to-end frozen-prefix check in execution space. If the
                    # policy server correctly transforms prev_chunk_left_over
                    # into model space and the model freezes it, then after
                    # output transforms the first `inference_delay` actions
                    # should match the prefix we sent byte-for-byte.
                    if prefix_sent is not None:
                        n_pref = len(prefix_sent)
                        n_pref = min(n_pref, len(processed_actions))
                        if n_pref > 0:
                            diff_pref = np.abs(processed_actions[:n_pref] - prefix_sent[:n_pref])
                            max_diff = float(np.max(diff_pref))
                            mean_diff = float(np.mean(diff_pref))
                            # per-dim max to identify which dimensions drift
                            dim_max = np.max(diff_pref, axis=0)
                            worst_dim = int(np.argmax(dim_max))
                            logger.info(
                                f"RTC Prefix Freeze Check: inference_delay={estimated_delay_steps}, "
                                f"new[0:{n_pref}] vs sent_prev[0:{n_pref}] "
                                f"max={max_diff:.4f} (dim {worst_dim}), mean={mean_diff:.4f}, "
                                "(should be ~0 if model froze prefix exactly)"
                            )

                    # Handle warmup phases
                    if not self._warmup_done:
                        if self._warmup_prev_chunk is None:
                            # ===== WARMUP PHASE 1 COMPLETE =====
                            # Save queued_actions for next inference, do NOT execute
                            self._warmup_prev_chunk = queued_actions
                            logger.info(
                                f"✅ WARMUP Phase 1 complete. Latency: {latency * 1000:.0f}ms. "
                                f"Saved {queued_actions.shape} for Phase 2. Actions NOT executed."
                            )
                            # Don't set _first_inference_done, continue to Phase 2
                            continue
                        else:
                            # ===== WARMUP PHASE 2 COMPLETE =====
                            # This inference has correct prev_chunk shape, execute these actions
                            self._warmup_done = True
                            # Use estimated_delay (default_delay) instead of actual latency for warmup
                            # because JIT compilation causes artificially high latency
                            inference_delay_steps = estimated_delay_steps
                            logger.info(
                                f"✅ WARMUP Phase 2 complete. Latency: {latency * 1000:.0f}ms (JIT). "
                                f"Using default_delay={estimated_delay_steps} for merge. Warmup done."
                            )
                            # Fall through to normal merge logic below

                    # Normal operation: merge actions into queue
                    # Skip CRITICAL check during warmup (Phase 2) since JIT causes high latency
                    if not self._warmup_done or inference_delay_steps < len(processed_actions):
                        pass  # OK
                    else:
                        logger.error(
                            f"RTC: CRITICAL - Inference delay ({inference_delay_steps} steps) "
                            f"exceeds action length ({len(processed_actions)} steps). "
                            "All actions will be discarded!"
                        )

                    merge_start = time.perf_counter()
                    real_delay_before_merge = self._action_queue.get_action_index() - action_index_before_inference
                    self._action_queue.merge(
                        new_original_actions=queued_actions,
                        new_processed_actions=processed_actions,
                        estimated_delay=estimated_delay_steps,
                        action_index_before_inference=action_index_before_inference,
                    )
                    merge_ms = (time.perf_counter() - merge_start) * 1000
                    logger.info(f"RTC: Merge total time: {merge_ms:.2f}ms")

                    # Update last_real_delay and the rolling window used for
                    # the next estimated_delay.
                    if real_delay_before_merge > 0:
                        self._last_real_delay = int(real_delay_before_merge)
                    else:
                        # Only use time-based if it's reasonable (< 10 steps)
                        if inference_delay_steps <= 10:
                            self._last_real_delay = int(inference_delay_steps)
                        else:
                            # JIT compilation case: use default
                            self._last_real_delay = 4
                            logger.info(
                                f"RTC: First inference took {inference_delay_steps} steps "
                                f"(likely JIT), using default delay=4 for next inference"
                            )

                    # Append to rolling max window (skip the warmup step whose
                    # latency is JIT-inflated: inference_delay_steps was forced
                    # to estimated_delay above, so real_delay_before_merge may
                    # still be noisy. We keep the append to seed the window and
                    # rely on delay_margin as the safety cushion.)
                    if self._last_real_delay is not None:
                        self._recent_real_delays.append(self._last_real_delay)

                    if self._dry_run:
                        self._inference_seq += 1
                        p95 = self._latency_tracker.p95()
                        recent_max = max(self._recent_real_delays) if self._recent_real_delays else None
                        next_est = (recent_max + self._delay_margin) if recent_max is not None else None
                        with self._metrics_lock:
                            self._last_inference_metrics = {
                                "inference_seq": self._inference_seq,
                                # Full client→server→inference→response time for this chunk request
                                "infer_round_trip_ms": latency * 1000.0,
                                "inference_delay_steps": inference_delay_steps,
                                "estimated_delay_steps": estimated_delay_steps,
                                "real_delay_steps": int(real_delay_before_merge),
                                "merge_ms": merge_ms,
                                "queue_size_after_merge": self._action_queue.qsize(),
                                "latency_p95_ms": (p95 or 0.0) * 1000.0,
                                "recent_max_real_delay": recent_max,
                                "delay_margin": self._delay_margin,
                                "delay_for_next_infer_steps": next_est,
                            }

                    # Signal that first inference is done (queue now has actions)
                    if not self._first_inference_done.is_set():
                        logger.info("First inference completed, action queue initialized")
                        self._first_inference_done.set()
                else:
                    # Sleep to prevent busy waiting
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in RTC background thread: {e}")
                time.sleep(0.1)

    @override
    def infer(self, obs: dict) -> dict:
        self._start_thread_if_needed()

        # Update latest observation for the background thread
        with self._latest_obs_lock:
            self._latest_obs = obs

        # Wait for first inference to complete (handles JIT compilation delay)
        if not self._first_inference_done.is_set():
            logger.info("Waiting for first inference to complete (JIT compilation)...")
            # Wait up to 120 seconds for first inference (JIT can be slow)
            if not self._first_inference_done.wait(timeout=120.0):
                raise RuntimeError(
                    "RTCActionChunkBroker: Timeout waiting for first inference. "
                    "Check if the policy server is running and responsive."
                )
            logger.info("First inference done, proceeding with action queue")

        # Get action from queue
        action = self._action_queue.get()

        if action is None:
            # Queue empty after first inference - this shouldn't happen often
            # Wait briefly and retry
            logger.warning("Action queue empty! Waiting...")
            start_wait = time.time()
            while action is None and (time.time() - start_wait) < 5.0:
                time.sleep(0.005)
                action = self._action_queue.get()

            if action is None:
                logger.error("Action queue empty after waiting!")
                raise RuntimeError("RTCActionChunkBroker: Action queue is empty.")

        # Return in the format expected by the agent (dict)
        out: dict = {"actions": action}
        if self._dry_run:
            with self._metrics_lock:
                snap = dict(self._last_inference_metrics) if self._last_inference_metrics else None
            if snap is not None:
                out["rtc_metrics"] = snap
        return out

    @override
    def reset(self) -> None:
        self._policy.reset()
        # Clear the action queue for next episode
        self._action_queue.clear()
        self._first_inference_done.clear()  # Reset for next episode
        self._last_real_delay = None  # Reset delay tracking
        self._recent_real_delays.clear()
        with self._metrics_lock:
            self._last_inference_metrics = None
        self._inference_seq = 0
        # Reset warmup state for next episode
        self._warmup_done = False
        self._warmup_prev_chunk = None
        with self._latest_obs_lock:
            self._latest_obs = None

    @override
    def warmup(self, obs: dict) -> None:
        """Pre-warm JIT compilation before the episode control loop.

        Must be called AFTER reset() (which clears warmup state) and BEFORE
        the first infer() call. Blocks until both warmup phases complete so
        the first real control step has zero JIT compilation delay.

        On the first episode JAX compiles the computation graph (~400 ms per
        phase); on subsequent episodes the cache is hot and both phases finish
        in ~50 ms each.

        Args:
            obs: A real observation from the environment.
        """
        logger.info("Pre-episode warmup: starting JIT compilation...")
        self._start_thread_if_needed()

        # Feed the observation so the background thread can begin inference.
        with self._latest_obs_lock:
            self._latest_obs = obs

        # Block until Phase 1 (JIT) + Phase 2 (prev_chunk shape) both finish.
        if not self._first_inference_done.wait(timeout=120.0):
            raise RuntimeError(
                "RTCActionChunkBroker.warmup(): Timed out waiting for JIT " "compilation. Is the policy server running?"
            )
        logger.info("Pre-episode warmup complete. First control step will have no JIT delay.")

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
