"""Rule-based subtask and physics-domain classifiers."""

from __future__ import annotations

from exact_physics_pipeline.domain import classify_domain


def classify_subtask(input_data: dict) -> int:
    """Classify an input object as Subtask 1, Subtask 2, or unknown."""
    if "premises-NL" in input_data or "premises_nl" in input_data:
        return 1
    if "question" in input_data and "premises-NL" not in input_data and "premises_nl" not in input_data:
        return 2
    return 0


__all__ = ["classify_domain", "classify_subtask"]

