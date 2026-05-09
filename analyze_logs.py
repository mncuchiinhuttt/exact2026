"""Analyze evaluation logs for parser failures, low confidence, and rounding drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def relative_error(predicted: float | None, expected: float | None) -> float | None:
    """Return relative error when both values are numeric."""
    if not isinstance(predicted, (int, float)) or not isinstance(expected, (int, float)):
        return None
    scale = max(abs(expected), 1e-12)
    return abs(predicted - expected) / scale


def analyze_log(path: Path) -> dict[str, Any]:
    """Analyze one evaluation JSON log."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    failed = []
    parse_failures = []
    low_confidence = []
    rounding_drift = []

    for case_record in cases:
        scoring = case_record.get("scoring", {})
        case_id = scoring.get("id") or case_record.get("case", {}).get("id")
        expected = scoring.get("expected_answer")
        predicted = scoring.get("predicted_answer")
        rel_error = relative_error(predicted, expected)
        all_answers = scoring.get("all_answers", [])
        failed_answers = [answer for answer in all_answers if answer.get("error") == "parse_failed"]

        if not scoring.get("passed", False):
            failed.append(
                {
                    "id": case_id,
                    "expected": f"{expected} {scoring.get('expected_unit')}",
                    "predicted": f"{predicted} {scoring.get('predicted_unit')}",
                    "confidence": scoring.get("confidence"),
                    "judge": scoring.get("judge"),
                }
            )
        if failed_answers:
            parse_failures.append(
                {
                    "id": case_id,
                    "parse_failed": len(failed_answers),
                    "total_generations": len(all_answers),
                    "confidence": scoring.get("confidence"),
                }
            )
        if scoring.get("confidence", 1.0) <= 0.4:
            low_confidence.append(
                {
                    "id": case_id,
                    "passed": scoring.get("passed"),
                    "confidence": scoring.get("confidence"),
                    "predicted": f"{predicted} {scoring.get('predicted_unit')}",
                }
            )
        if rel_error is not None and rel_error >= 0.005:
            rounding_drift.append(
                {
                    "id": case_id,
                    "relative_error": rel_error,
                    "passed": scoring.get("passed"),
                    "expected": expected,
                    "predicted": predicted,
                }
            )

    return {
        "run": data.get("run", {}),
        "failed": failed,
        "parse_failures": parse_failures,
        "low_confidence": low_confidence,
        "rounding_drift": rounding_drift,
    }


def main() -> None:
    """Print a compact log analysis report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to logs/evaluation_*.json")
    args = parser.parse_args()

    report = analyze_log(args.log_path)
    run = report["run"]
    print(f"Log: {args.log_path}")
    print(f"Accuracy: {run.get('passed_cases')}/{run.get('total_cases')} = {run.get('accuracy')}")
    print(f"Duration: {run.get('duration_seconds')} seconds")

    for section in ["failed", "parse_failures", "low_confidence", "rounding_drift"]:
        print(f"\n{section}:")
        items = report[section]
        if not items:
            print("  none")
            continue
        for item in items:
            print(f"  {item}")


if __name__ == "__main__":
    main()
