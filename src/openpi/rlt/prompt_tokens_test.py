import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer
from openpi.rlt import prefix_cache
from openpi.rlt import prompt_tokens

MAX_LEN = 200  # pi05 configs use 128-200; a 20D state alone is ~80 tokens
NUM_IMAGE_TOKENS = 12


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return _tokenizer.PaligemmaTokenizer(MAX_LEN)
    except Exception as exc:
        pytest.skip(f"PaliGemma tokenizer unavailable: {exc}")


def _drop(tokenizer, prompt, state, *, state_in_prompt=True):
    tokens, token_mask = tokenizer.tokenize(prompt, state)
    prefix_mask = np.concatenate([np.ones(NUM_IMAGE_TOKENS, bool), token_mask])[None]
    kept = prompt_tokens.drop_task_tokens(
        jnp.asarray(prefix_mask),
        jnp.asarray(tokens)[None],
        jnp.asarray(token_mask)[None],
        state_in_prompt=state_in_prompt,
    )[0]
    kept = np.asarray(kept)
    pieces = [tokenizer._tokenizer.id_to_piece(int(t)) for t in tokens]
    prompt_kept = kept[NUM_IMAGE_TOKENS:]
    return (
        kept[:NUM_IMAGE_TOKENS],
        [p for p, k, m in zip(pieces, prompt_kept, token_mask, strict=True) if m and not k],
        [p for p, k in zip(pieces, prompt_kept, strict=True) if k],
    )


def test_state_token_id(tokenizer):
    assert tokenizer._tokenizer.piece_to_id("▁State") == prompt_tokens.STATE_TOKEN_ID


@pytest.mark.parametrize(
    ("prompt", "instruction"),
    [
        ("pick up the cube", ["▁pick", "▁up", "▁the", "▁cube"]),
        (
            "insert the ethernet cable into the port",
            ["▁insert", "▁the", "▁ethernet", "▁cable", "▁into", "▁the", "▁port"],
        ),
        # The instruction itself mentions "State": only the template's (last) marker counts.
        ("Task: tricky State: prompt", ["▁Task", ":", "▁tricky", "▁State", ":", "▁prompt"]),
    ],
)
def test_drops_the_instruction_and_keeps_the_state(tokenizer, prompt, instruction):
    for state in (np.zeros(20), np.full(20, -1.0), np.random.default_rng(0).uniform(-1, 1, 20)):
        images, dropped, kept = _drop(tokenizer, prompt, state)
        assert images.all()
        assert dropped == ["<bos>", "Task", ":", *instruction, ","]
        assert kept[:2] == ["▁State", ":"]
        assert "".join(kept).replace("▁", " ").strip().endswith("Action:")


def test_without_state_in_the_prompt_the_whole_prompt_is_dropped(tokenizer):
    images, dropped, kept = _drop(tokenizer, "pick up the cube", None, state_in_prompt=False)
    assert images.all()
    assert kept == []
    assert dropped[0] == "<bos>"


def test_truncated_state_drops_the_whole_prompt():
    tokens = jnp.asarray([[2, 7, 8, 9]])
    kept = prompt_tokens.drop_task_tokens(jnp.ones((1, 6), bool), tokens, jnp.ones((1, 4), bool), state_in_prompt=True)
    np.testing.assert_array_equal(kept, [[True, True, False, False, False, False]])


def test_cache_identity_covers_drop_task_tokens():
    identity = {
        "vla_config": "v",
        "params_fingerprint": "p",
        "norm_stats_fingerprint": "n",
        "repo_id": "r",
        "frame_stride": 1,
    }
    legacy = dict(identity)  # a cache written before the key existed kept the prompt
    prefix_cache.check_identity(legacy, {**identity, "drop_task_tokens": False})
    with pytest.raises(ValueError, match="drop_task_tokens"):
        prefix_cache.check_identity(legacy, {**identity, "drop_task_tokens": True})
    prefix_cache.check_identity({**identity, "drop_task_tokens": True}, {**identity, "drop_task_tokens": True})
