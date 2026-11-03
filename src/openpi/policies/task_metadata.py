"""Read-only description of configured defaults, not request-level task analysis."""


def with_default_task_instruction(metadata, candidates):
    result = dict(metadata or {})
    result.pop("task_instruction", None)
    for prompt, source in candidates:
        if prompt is not None:
            result["task_instruction"] = {
                "text": prompt,
                "source": source,
                "applies_when": "request_has_no_prompt",
            }
            break
    return result
