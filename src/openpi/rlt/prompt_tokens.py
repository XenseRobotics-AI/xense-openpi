"""Which prompt tokens of the frozen VLA's prefix the RL token sees.

The RLT paper drops the language embeddings before training the RL token, since
each task has a fixed instruction (Sec. IV-A, footnote 1). pi05 with
``discrete_state_input`` renders the prompt as::

    <bos>Task: {instruction}, State: {discretized normalized state};\\nAction:

so the prompt slots also carry the robot state. ``drop_task_tokens`` therefore
masks out only the instruction - everything before the last ``▁State`` token -
and keeps the state section. The discretized state is digits, spaces, ``;`` and
``Action:``, which never tokenize to ``▁State``, so its last occurrence is the
template's even when the instruction itself contains "State". Without the state
in the prompt (pi0 format), or when it was truncated away, the whole prompt is
instruction and is dropped.

The prefix layout is fixed (one 256-token block per camera, then the prompt
right-padded to ``max_token_len``), so the prompt always occupies the last
``max_token_len`` prefix slots and masking a slot never moves another one.
"""

from __future__ import annotations

import jax.numpy as jnp

from openpi.shared import array_typing as at

# "▁State" in the PaliGemma SentencePiece vocabulary (checked in the tests).
STATE_TOKEN_ID = 3040


def drop_task_tokens(
    prefix_mask: at.Bool[at.Array, "b s"],
    tokenized_prompt: at.Int[at.Array, "b l"],
    tokenized_prompt_mask: at.Bool[at.Array, "b l"],
    *,
    state_in_prompt: bool,
) -> at.Bool[at.Array, "b s"]:
    """``prefix_mask`` with the instruction tokens of the prompt slots set to False."""
    length = tokenized_prompt.shape[1]
    if prefix_mask.shape[1] < length:
        raise ValueError(f"Prefix of {prefix_mask.shape[1]} slots cannot hold a {length}-token prompt.")
    index = jnp.arange(length)
    if state_in_prompt:
        marker = (tokenized_prompt == STATE_TOKEN_ID) & tokenized_prompt_mask
        last = jnp.max(jnp.where(marker, index, -1), axis=1)
        start = jnp.where(last >= 0, last, length)
    else:
        start = jnp.full(tokenized_prompt.shape[:1], length)
    keep = index[None, :] >= start[:, None]
    return prefix_mask.at[:, -length:].set(prefix_mask[:, -length:] & keep)
