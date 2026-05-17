"""Self-consistency voting for parsed numeric answers."""

from __future__ import annotations

import statistics
from typing import Any


def _value(answer: dict[str, Any]) -> float | None:
    """Return the SI-expanded value when available, otherwise the raw value."""
    value = answer.get("value_si")
    if value is None:
        value = answer.get("value")
    return float(value) if value is not None else None


def _close(a: float, b: float, tolerance: float) -> bool:
    """Return whether two numbers are within relative tolerance."""
    return abs(a - b) / max(abs(a), abs(b), 1e-12) <= tolerance


def vote(parsed_answers: list[dict[str, Any]], tolerance: float = 0.03) -> dict[str, Any]:
    """Cluster answers by unit and value, then return the majority median."""
    usable = [answer for answer in parsed_answers if _value(answer) is not None]
    clusters: list[dict[str, Any]] = []
    for answer in usable:
        value = _value(answer)
        unit = answer.get("unit") or "1"
        for cluster in clusters:
            if cluster["unit"] == unit and _close(cluster["center"], value, tolerance):
                cluster["items"].append(answer)
                values = [_value(item) for item in cluster["items"]]
                cluster["center"] = statistics.median(values)
                break
        else:
            clusters.append({"unit": unit, "center": value, "items": [answer]})

    if not usable:
        return {"answer": None, "unit": "", "confidence": 0.0, "winning_count": 0, "total": 0, "all_clusters": []}

    def cluster_key(cluster: dict[str, Any]) -> tuple[int, float]:
        values = [_value(item) for item in cluster["items"]]
        stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        return (len(cluster["items"]), -stdev)

    winner = max(clusters, key=cluster_key)
    values = [_value(item) for item in winner["items"]]
    return {
        "answer": float(statistics.median(values)),
        "unit": winner["unit"],
        "confidence": len(winner["items"]) / len(usable),
        "winning_count": len(winner["items"]),
        "total": len(usable),
        "all_clusters": [
            {"unit": c["unit"], "values": [_value(item) for item in c["items"]], "count": len(c["items"])}
            for c in clusters
        ],
    }

