import pytest

from openpi.policies.task_metadata import with_default_task_instruction


@pytest.mark.parametrize(
    ("server", "model", "expected", "source"),
    [
        ("server instruction", "training instruction", "server instruction", "server"),
        ("", "training instruction", "", "server"),
        (None, "training instruction", "training instruction", "model"),
        (None, None, None, None),
    ],
)
def test_default_instruction_precedence_and_unavailable(server, model, expected, source):
    original = {"name": "checkpoint", "task_instruction": {"text": "unverified label"}}
    result = with_default_task_instruction(original, ((server, "server"), (model, "model")))
    assert original["task_instruction"]["text"] == "unverified label"
    assert result["name"] == "checkpoint"
    if expected is None:
        assert "task_instruction" not in result
    else:
        assert result["task_instruction"] == {
            "text": expected,
            "source": source,
            "applies_when": "request_has_no_prompt",
        }
