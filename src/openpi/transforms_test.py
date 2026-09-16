import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_repack_transform_alias_key_takes_either_spelling():
    transform = _transforms.RepackTransform(
        structure={
            "images": {
                "left_tactile_top": _transforms.AliasKey(
                    "observation.images.left_tactile_0",
                    "observation.images.left_tactile_left",
                )
            }
        }
    )

    # LeRobot items are flat dicts whose column names contain dots.
    legacy = {"observation.images.left_tactile_0": 7}
    current = {"observation.images.left_tactile_left": 9}

    assert transform(legacy) == {"images": {"left_tactile_top": 7}}
    assert transform(current) == {"images": {"left_tactile_top": 9}}


def test_repack_transform_alias_key_reports_both_spellings_when_absent():
    alias = _transforms.AliasKey("observation.images.left_tactile_0", "observation.images.left_tactile_left")
    transform = _transforms.RepackTransform(structure={"images": {"left_tactile_top": alias}})

    with pytest.raises(KeyError, match="left_tactile_left"):
        transform({"observation.images.head": 1})


def test_repack_transform_source_keys_expands_aliases():
    transform = _transforms.RepackTransform(
        structure={
            "state": "observation.state",
            "images": {"left_tactile_top": _transforms.AliasKey("a", "b")},
        }
    )

    assert transform.source_keys() == {"observation.state", "a", "b"}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def _write_future_label_store(root, *, lengths=(5, 3), num_pads=4, latent_dim=8, horizons=(1, 2)):
    """A tiny label store where frame ``i`` has latent value ``i`` and pixel value ``10 * i``."""
    import json

    num_frames = sum(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    z = np.broadcast_to(np.arange(num_frames)[:, None, None], (num_frames, num_pads, latent_dim)).astype(np.float16)
    field = np.broadcast_to(10 * np.arange(num_frames)[:, None, None, None, None], (num_frames, num_pads, 16, 16, 3))
    np.save(root / "episode_offsets.npy", offsets)
    np.save(root / "z_tac.npy", z)
    np.save(root / "pixel_field.npy", field.astype(np.uint8))
    rms = {str(k): [0.5 * k] * num_pads for k in horizons}
    (root / "meta.json").write_text(json.dumps({"num_pads": num_pads, "horizons": list(horizons), "y_delta_rms": rms}))


def test_inject_tactile_future_labels_latent_lookup_and_mask(tmp_path):
    _write_future_label_store(tmp_path)
    transform = _transforms.InjectTactileFutureLabels(labels_dir=str(tmp_path), horizons=(1, 2))

    # Episode 0 (frames 0-4), frame 3: frame 4 exists, frame 5 does not.
    out = transform({"episode_index": np.int64(0), "frame_index": np.int64(3), "index": np.int64(3)})
    aux = out["aux_targets"]
    assert aux["future_tactile_z"].shape == (2, 4, 8)
    assert aux["future_tactile_z"].dtype == np.float32
    assert aux["future_tactile_mask"].tolist() == [True, False]
    assert aux["future_tactile_z"][0, 0, 0] == 4.0
    assert (aux["future_tactile_z"][1] == 0).all()

    # Episode 1 (global frames 5-7), frame 0: both horizons valid, values from the second episode.
    aux = transform({"episode_index": np.int64(1), "frame_index": np.int64(0)})["aux_targets"]
    assert aux["future_tactile_mask"].tolist() == [True, True]
    assert aux["future_tactile_z"][:, 0, 0].tolist() == [6.0, 7.0]

    # Last frame of the last episode: nothing valid, and no read past the store.
    aux = transform({"episode_index": np.int64(1), "frame_index": np.int64(2)})["aux_targets"]
    assert aux["future_tactile_mask"].tolist() == [False, False]


def test_inject_tactile_future_labels_pixel_delta(tmp_path):
    _write_future_label_store(tmp_path)
    transform = _transforms.InjectTactileFutureLabels(labels_dir=str(tmp_path), horizons=(1, 2), target="pixel_delta")
    aux = transform({"episode_index": np.int64(0), "frame_index": np.int64(0)})["aux_targets"]
    assert aux["future_tactile_z"].shape == (2, 4, 16 * 16 * 3)
    # (10k / 255) / rms_k with rms_k = 0.5k -> 20 / 255 for every horizon.
    np.testing.assert_allclose(aux["future_tactile_z"], 20.0 / 255.0, rtol=1e-5)

    with pytest.raises(ValueError, match="RMS for horizons"):
        _transforms.InjectTactileFutureLabels(labels_dir=str(tmp_path), horizons=(3,), target="pixel_delta")(
            {"episode_index": np.int64(0), "frame_index": np.int64(0)}
        )


def test_inject_tactile_future_labels_rejects_mismatched_store(tmp_path):
    _write_future_label_store(tmp_path)
    transform = _transforms.InjectTactileFutureLabels(labels_dir=str(tmp_path), horizons=(1,))
    with pytest.raises(ValueError, match="disagrees"):
        transform({"episode_index": np.int64(1), "frame_index": np.int64(0), "index": np.int64(4)})
    with pytest.raises(IndexError):
        transform({"episode_index": np.int64(2), "frame_index": np.int64(0)})
    with pytest.raises(IndexError):
        transform({"episode_index": np.int64(0), "frame_index": np.int64(5)})
    with pytest.raises(ValueError, match="needs 'frame_index'"):
        transform({"episode_index": np.int64(0)})
