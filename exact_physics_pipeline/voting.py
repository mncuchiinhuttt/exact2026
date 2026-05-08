"""Consensus voting for independently sampled answers."""

from typing import Any


EQUIVALENT_UNITS = {
    "ohms": "ohm",
    "ω": "ohm",
    "Ω": "ohm",
    "v/m": "n/c",
}


def normalize_unit(unit: str | None) -> str | None:
    """Normalize units for grouping equivalent answers."""
    if unit is None:
        return None
    normalized = unit.strip().lower().replace("μ", "u")
    return EQUIVALENT_UNITS.get(normalized, normalized)


def _within_relative_tolerance(a: float, b: float, tolerance: float = 0.02) -> bool:
    """Return whether two values are within a relative tolerance."""
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) <= tolerance * scale


def choose_consensus(parsed_answers: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Select the answer group with the most votes using unit-aware tolerance grouping."""
    groups: list[dict[str, Any]] = []

    for index, answer in enumerate(parsed_answers):
        value = answer.get("answer")
        unit = normalize_unit(answer.get("unit"))
        if value is None or unit is None:
            continue

        for group in groups:
            if group["unit"] == unit and _within_relative_tolerance(group["representative"], value):
                group["items"].append((index, answer))
                values = [item["answer"] for _, item in group["items"]]
                group["representative"] = sum(values) / len(values)
                break
        else:
            groups.append({"unit": unit, "representative": value, "items": [(index, answer)]})

    if not groups:
        return {
            "answer": None,
            "unit": None,
            "confidence": 0.0,
            "winner_index": None,
            "all_answers": parsed_answers,
        }

    winner = max(groups, key=lambda group: len(group["items"]))
    winner_index, first_answer = winner["items"][0]
    return {
        "answer": float(winner["representative"]),
        "unit": first_answer["unit"],
        "confidence": len(winner["items"]) / k,
        "winner_index": winner_index,
        "all_answers": parsed_answers,
    }
