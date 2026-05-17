"""Input-shape router for EXACT 2026 subtasks."""

from __future__ import annotations

from typing import Any

from exact2026.pipeline.domain_classifier import classify_subtask
from exact2026.pipeline.pipeline_edu import run_edu
from exact2026.pipeline.pipeline_physics import run_physics


def route_and_run(input_data: dict, config: dict | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    """Route an input object to Subtask 1 or Subtask 2 and run it."""
    subtask = classify_subtask(input_data)
    if subtask == 2:
        return run_physics(input_data.get("question", ""), config)
    if subtask == 1:
        premises = input_data.get("premises-NL") or input_data.get("premises_nl") or []
        questions = input_data.get("questions") or []
        if isinstance(questions, str):
            questions = [{"question": questions}]
        results = []
        for q_item in questions:
            q_text = q_item.get("question") if isinstance(q_item, dict) else str(q_item)
            results.append(run_edu(premises, q_text, config))
        return results[0] if len(results) == 1 else results
    return {"error": "Cannot determine subtask from input shape", "input_keys": list(input_data.keys())}
