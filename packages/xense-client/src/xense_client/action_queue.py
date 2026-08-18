from threading import Lock

import numpy as np
from lerobot.utils.robot_utils import get_logger

logger = get_logger("ActionQueue")


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control (NumPy version).

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity
    """

    def __init__(self, rtc_enabled: bool = True, blend_steps: int = 0):
        """Initialize the action queue.

        Args:
            rtc_enabled: Whether Real-Time Chunking is enabled.
            blend_steps: Number of steps to blend between old and new actions at merge.
                        0 = no blending (hard switch), >0 = linear blend over N steps.
        """
        self.queue: np.ndarray | None = None  # Processed actions for robot rollout
        self.original_queue: np.ndarray | None = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.rtc_enabled = rtc_enabled
        self.blend_steps = blend_steps

    def get(self) -> np.ndarray | None:
        """Get the next action from the queue.

        Returns:
            np.ndarray | None: The next action (action_dim,) or None if queue is empty.
                              Returns a copy to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                logger.warning(
                    "Action queue exhausted! No actions available. "
                    "This may cause robot to stall. Consider increasing execution_horizon "
                    "or reducing action_queue_size_to_get_new_actions."
                )
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        return self.last_index

    def clear(self) -> None:
        """Clear the queue, removing all actions.

        This resets the queue to its initial empty state while preserving
        the rtc_enabled setting.
        """
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def get_left_over(self, fixed_length: int = 0) -> np.ndarray | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC as the action prefix for the next chunk.

        The model uses the FIRST ``inference_delay`` elements of this array as
        the frozen prefix (see training-time-rtc paper, Algorithm 1:
        ``prefix_mask = arange(ah) < delay``). Those first elements must be
        the imminent actions — the ones the robot will consume during the
        upcoming inference delay window.

        Args:
            fixed_length: If > 0, pad or truncate to this exact length so the
                         returned shape is always (fixed_length, action_dim).
                         This prevents JAX JIT recompilation when the leftover
                         count fluctuates by 1-2 steps between inferences.
                         Truncation keeps the HEAD (imminent) actions so that
                         the prefix alignment is preserved. Padding appends
                         copies of the last action at the tail, which never
                         enter the prefix region (since fixed_length is always
                         >= inference_delay in normal operation).

        Returns:
            np.ndarray | None: Remaining original actions, or None if no
                              original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            left_over = self.original_queue[self.last_index :]

            if fixed_length <= 0 or len(left_over) == 0:
                return left_over

            if len(left_over) >= fixed_length:
                # Keep the HEAD: the first fixed_length actions are the
                # imminent ones. The model uses left_over[0:inference_delay]
                # as the frozen prefix, so truncating from the tail would
                # break prefix alignment.
                return left_over[:fixed_length]
            else:
                # Pad at the tail: repeat the last action to fill. The
                # padded region is beyond the prefix window and is ignored
                # by the model's prefix mask.
                pad_count = fixed_length - len(left_over)
                padding = np.repeat(left_over[-1:], pad_count, axis=0)
                return np.concatenate([left_over, padding], axis=0)

    def merge(
        self,
        new_original_actions: np.ndarray,
        new_processed_actions: np.ndarray,
        estimated_delay: int,
        action_index_before_inference: int | None = 0,
    ):
        """Merge new actions into the queue.

        Args:
            new_original_actions: Unprocessed actions from policy (time_steps, action_dim).
            new_processed_actions: Post-processed actions for robot (time_steps, action_dim).
            estimated_delay: estimated delay steps passed to model.
            action_index_before_inference: Index before inference started.

        Note:
            We use real_delay for truncation to avoid skipping actions:
            - real_delay = actual steps consumed during inference
            - Truncating at real_delay ensures no action is skipped
            - RTC guidance ensures new_actions[real_delay] is smooth with old trajectory
        """
        real_delay = None
        truncate_delay = None
        with self.lock:
            if self.rtc_enabled:
                real_delay = self.get_action_index() - action_index_before_inference
                # Use real_delay for truncation to avoid skipping actions
                # RTC guidance ensures actions near real_delay are smooth
                truncate_delay = real_delay
                self._replace_actions_queue(
                    new_original_actions,
                    new_processed_actions,
                    truncate_delay,
                    action_index_before_inference,
                )
            else:
                self._append_actions_queue(new_original_actions, new_processed_actions)

        # Log outside lock to avoid blocking get() calls
        if real_delay is not None:
            logger.info(f"RTC: Truncate at {truncate_delay}, " f"estimated={estimated_delay}, real={real_delay}")

    def _replace_actions_queue(
        self,
        new_original_actions: np.ndarray,
        new_processed_actions: np.ndarray,
        truncate_delay: int,
        action_index_before_inference: int = 0,
    ):
        """Replace the queue with new actions (RTC mode).

        Args:
            truncate_delay: The delay used to truncate actions (should be estimated_delay
                           that was passed to the model, ensuring consistency).
            action_index_before_inference: Queue index when inference started, used
                           for correct prefix alignment comparison.
        """
        truncate_idx = max(0, min(truncate_delay, len(new_original_actions)))

        # Debug: Verify RTC alignment in the prefix region.
        # NOTE: When delta_state_dim > 0 in the broker (delta-action policies
        # like BiFlexiv), this check is MISLEADING. self.original_queue holds
        # deltas w.r.t. the PREVIOUS inference's obs.state, while
        # new_original_actions holds deltas w.r.t. the CURRENT obs.state. The
        # broker re-bases prev_chunk_left_over before sending it to the model,
        # so new[0:d] == (rebased prev_chunk)[0:d] != original_queue[aib:aib+d]
        # when the robot has moved between inferences. The true frozen-prefix
        # invariant is logged by the broker as "RTC Prefix Freeze Check"
        # (compares new[0:d] against what was actually sent). This log below
        # is kept for absolute-action policies where the two frames coincide.
        if self.original_queue is not None and truncate_idx > 0:
            aib = action_index_before_inference
            align_len = min(truncate_idx, len(self.original_queue) - aib, len(new_original_actions))
            if align_len > 0:
                old_aligned = self.original_queue[aib : aib + align_len]
                new_aligned = new_original_actions[:align_len]
                diff_rtc = np.abs(new_aligned - old_aligned)
                max_diff_rtc = np.max(diff_rtc)
                mean_diff_rtc = np.mean(diff_rtc)
                logger.info(
                    f"RTC Alignment Check: new[0:{align_len}] vs old[{aib}:{aib + align_len}] "
                    f"max={max_diff_rtc:.4f}, mean={mean_diff_rtc:.4f} "
                    f"(should be ~0 for frozen prefix)"
                )

        # Debug: Check actual jump at merge point
        if self.queue is not None and truncate_idx < len(new_processed_actions):
            next_action = new_processed_actions[truncate_idx]
            # Compare with last executed action (the actual jump)
            last_executed_idx = self.last_index - 1
            if last_executed_idx >= 0 and last_executed_idx < len(self.queue):
                last_action = self.queue[last_executed_idx]
                diff_jump = np.abs(next_action - last_action)
                max_diff_jump = np.max(diff_jump)
                mean_diff_jump = np.mean(diff_jump)
                max_diff_dim = np.argmax(diff_jump)
                logger.info(
                    f"RTC Actual Jump: new[{truncate_idx}] vs old[{last_executed_idx}] "
                    f"max={max_diff_jump:.4f} (dim {max_diff_dim}), mean={mean_diff_jump:.4f}"
                )

        # Get the new actions starting from truncate_idx
        new_original = new_original_actions[truncate_idx:].copy()
        new_processed = new_processed_actions[truncate_idx:].copy()

        # Apply blending if enabled and we have old actions
        if self.blend_steps > 0 and self.queue is not None and self.last_index < len(self.queue):
            # Get remaining old actions
            old_remaining = self.queue[self.last_index :]
            logger.debug(f"old_remaining steps: {len(old_remaining)}")
            blend_len = min(self.blend_steps, len(old_remaining), len(new_processed))

            if blend_len > 0:
                # Linear blend: alpha goes from 0 to 1 over blend_steps
                for i in range(blend_len):
                    alpha = (i + 1) / (blend_len + 1)  # 0 < alpha < 1
                    new_processed[i] = (1 - alpha) * old_remaining[i] + alpha * new_processed[i]
                    new_original[i] = (1 - alpha) * self.original_queue[self.last_index + i] + alpha * new_original[i]

        self.original_queue = new_original
        self.queue = new_processed
        self.last_index = 0

    def _append_actions_queue(self, new_original_actions: np.ndarray, new_processed_actions: np.ndarray):
        """Append new actions to the queue (non-RTC mode)."""
        if self.queue is None:
            self.original_queue = new_original_actions.copy()
            self.queue = new_processed_actions.copy()
            return

        # Remove consumed actions
        self.original_queue = self.original_queue[self.last_index :]
        self.queue = self.queue[self.last_index :]

        # Append new actions
        self.original_queue = np.concatenate([self.original_queue, new_original_actions])
        self.queue = np.concatenate([self.queue, new_processed_actions])

        self.last_index = 0
