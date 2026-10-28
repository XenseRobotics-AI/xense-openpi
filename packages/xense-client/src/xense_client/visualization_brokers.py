"""Opt-in action observation without changing the shared control implementations.

Only visualization-enabled entry points select these subclasses. The original
brokers still own inference, warmup, scheduling, clipping and blending; these
extensions mirror the resulting actions and attach submission references.
"""

import threading

from xense_client.action_chunk_broker import ActionChunkBroker
from xense_client.action_queue import ActionQueue
from xense_client.action_stream import ACTION_REF_KEY
from xense_client.base_policy import BasePolicy
from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker


class _ChunkObserver(BasePolicy):
    def __init__(self, policy, on_chunk):
        self.policy, self.on_chunk = policy, on_chunk

    def infer(self, obs):
        result = self.policy.infer(obs)
        self.on_chunk(result)
        return result

    def reset(self):
        self.policy.reset()


class ObservedActionChunkBroker(ActionChunkBroker):
    """Run the original chunk broker and report only its executable prefix."""

    def __init__(self, policy, action_horizon, *, action_stream):
        self._action_stream = action_stream
        self._visualization_refs = None
        super().__init__(_ChunkObserver(policy, self._chunk_received), action_horizon)

    def _chunk_received(self, result):
        stream = self._action_stream
        self._visualization_refs = None
        if stream.enabled:
            refs = stream.chunk_received(result.get("actions"), usable_steps=self._action_horizon)
            self._visualization_refs = refs
            if refs is not None:
                stream.plan_updated(result["actions"][: len(refs)], refs)

    def infer(self, obs):
        index = 0 if self._last_results is None else self._cur_step
        result = super().infer(obs)
        refs = self._visualization_refs
        if self._action_stream.enabled and refs is not None and index < len(refs):
            ref = refs[index]
            self._action_stream.popped(ref, result["actions"])
            result[ACTION_REF_KEY] = ref
        return result

    def reset(self):
        super().reset()
        self._visualization_refs = None
        self._action_stream.reset()


class _ObservedActionQueue(ActionQueue):
    """Mirror the original queue under its lock, without duplicating queue math."""

    def __init__(self, *, action_stream, rtc_enabled=True, blend_steps=0):
        super().__init__(rtc_enabled=rtc_enabled, blend_steps=blend_steps)
        # get()/clear() enclose the parent call so the action and its reference
        # stay atomic. This lock is private to the opt-in queue instance.
        self.lock = threading.RLock()
        self._stream = action_stream
        self._refs = ()
        self._local = threading.local()

    @property
    def last_ref(self):
        return getattr(self._local, "last_ref", None)

    def get(self):
        with self.lock:
            index = self.last_index
            action = super().get()
            ref = self._refs[index] if action is not None and index < len(self._refs) else None
            self._local.last_ref = ref if self._stream.enabled else None
            if action is not None and self._stream.enabled:
                self._stream.popped(ref, action)
            return action

    def merge(self, new_original_actions, new_processed_actions, estimated_delay, action_index_before_inference=0):
        # The parent broker calls merge only for executable chunks, after JIT
        # warmup. Freeze the full arrival before taking the queue lock.
        self._local.incoming_refs = self._stream.chunk_received(new_processed_actions)
        try:
            return super().merge(
                new_original_actions, new_processed_actions, estimated_delay, action_index_before_inference
            )
        finally:
            self._local.incoming_refs = None

    def _replace_actions_queue(
        self, new_original_actions, new_processed_actions, truncate_delay, action_index_before_inference=0
    ):
        # Called by the parent merge with the queue lock already held.
        super()._replace_actions_queue(
            new_original_actions, new_processed_actions, truncate_delay, action_index_before_inference
        )
        refs = self._local.incoming_refs
        if self._stream.enabled and refs is not None:
            # Derive the retained suffix from the actual parent result, rather
            # than maintaining another implementation of delay/cropping math.
            self._refs = refs[len(new_processed_actions) - len(self.queue) :]
            self._stream.plan_updated(self.queue, self._refs)
        else:
            self._refs = ()

    def _append_actions_queue(self, new_original_actions, new_processed_actions):
        retained = self._refs[self.last_index :]
        super()._append_actions_queue(new_original_actions, new_processed_actions)
        refs = self._local.incoming_refs
        if self._stream.enabled and refs is not None:
            self._refs = retained + refs
            self._stream.plan_updated(self.queue, self._refs)
        else:
            self._refs = ()

    def clear(self):
        with self.lock:
            super().clear()
            self._refs = ()
            self._local.last_ref = None
            self._stream.reset()


class ObservedRTCActionChunkBroker(RTCActionChunkBroker):
    """Use the unchanged RTC worker and observe its effective execution queue."""

    def __init__(self, *args, action_stream, **kwargs):
        super().__init__(*args, **kwargs)
        # The parent constructs, but does not start, its worker in __init__.
        queue = self._action_queue
        self._action_queue = _ObservedActionQueue(
            action_stream=action_stream, rtc_enabled=queue.rtc_enabled, blend_steps=queue.blend_steps
        )

    def infer(self, obs):
        result = super().infer(obs)
        ref = self._action_queue.last_ref
        if ref is not None:
            result[ACTION_REF_KEY] = ref
        return result
