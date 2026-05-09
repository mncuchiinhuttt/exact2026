"""Evaluate the live pipeline on curated physics test cases."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

from exact_physics_pipeline.test_cases import get_test_cases
from run import run_structured


DEFAULT_JUDGE_MODEL = "gpt-5.4"
LOG_DIR = Path("logs")


def normalize_unit(unit: str | None) -> str:
    """Normalize unit text for simple exact/equivalent comparisons."""
    if not unit:
        return ""
    return (
        unit.strip()
        .replace("μ", "u")
        .replace("Ω", "ohm")
        .replace("²", "^2")
        .replace("³", "^3")
        .lower()
    )


def unit_matches(predicted: str | None, case: dict[str, Any]) -> bool:
    """Return whether a predicted unit matches expected or accepted units."""
    expected_units = [case["expected_unit"], *case.get("accepted_units", [])]
    normalized_predicted = normalize_unit(predicted)
    return normalized_predicted in {normalize_unit(unit) for unit in expected_units}


def answer_matches(predicted: float | None, expected: float, tolerance: float) -> bool:
    """Return whether a numeric prediction is within relative tolerance."""
    if predicted is None or not math.isfinite(float(predicted)):
        return False
    scale = max(abs(expected), 1e-12)
    return abs(float(predicted) - expected) <= tolerance * scale


def relative_error(predicted: float | None, expected: float) -> float | None:
    """Return relative error for a prediction, or None when unavailable."""
    if predicted is None or not math.isfinite(float(predicted)):
        return None
    scale = max(abs(expected), 1e-12)
    return abs(float(predicted) - expected) / scale


def load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file without overwriting the environment."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse judge JSON, tolerating invalid backslash escapes in feedback text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if isinstance(parsed, dict):
        score = parsed.get("score")
        feedback = parsed.get("feedback", "")
    else:
        score_match = re.search(r'"?score"?\s*:\s*(\d{1,3})', text)
        feedback_match = re.search(r'"?feedback"?\s*:\s*"(.*?)"\s*\}?$', text, re.DOTALL)
        score = int(score_match.group(1)) if score_match else None
        feedback = feedback_match.group(1) if feedback_match else text

    if isinstance(score, float):
        score = round(score)
    if isinstance(score, str) and score.isdigit():
        score = int(score)
    feedback = sanitize_feedback(str(feedback))

    if not isinstance(score, int) or score < 0 or score > 100:
        return {"score": None, "feedback": f"Judge returned invalid score: {text[:300]}"}
    return {"score": score, "feedback": feedback}


def sanitize_feedback(feedback: str) -> str:
    """Clean judge feedback for compact terminal display."""
    cleaned = feedback.replace('\\"', '"').replace("\\n", " ")
    cleaned = re.sub(r"\\mu\\text\{([^}]+)\}", r"micro\1", cleaned)
    cleaned = re.sub(r"\\text\{([^}]+)\}", r"\1", cleaned)
    cleaned = cleaned.replace("\\mu", "micro")
    cleaned = cleaned.replace("\\(", "(").replace("\\)", ")")
    cleaned = cleaned.replace("\\[", "[").replace("\\]", "]")
    cleaned = cleaned.replace("\\times", "x")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def build_judge_prompt(case: dict[str, Any], scored: dict[str, Any]) -> str:
    """Build a compact judge prompt for evaluating physics reasoning quality."""
    return f"""Judge this physics solution on a 0-100 scale.

Focus on correctness of reasoning, formula choice, unit handling, substitutions, and final answer.
Be fair about harmless rounding, but penalize wrong formulas, wrong unit conversions, missing steps, or unsupported final answers.

Return only JSON with this schema:
{{"score": <integer 0-100>, "feedback": "<one or two short sentences>"}}
Do not use Markdown, LaTeX, or escaped backslashes in the feedback.

Problem:
{case["problem"]}

Expected answer:
{case["expected_answer"]} {case["expected_unit"]}

Pipeline prediction:
{scored["predicted_answer"]} {scored["predicted_unit"]}

Numeric check passed: {scored["numeric_ok"]}
Unit check passed: {scored["unit_ok"]}

Pipeline explanation:
{scored.get("explanation", "")[:3500]}
"""


def call_openai_judge(case: dict[str, Any], scored: dict[str, Any], model: str) -> dict[str, Any]:
    """Use an OpenAI model to grade the pipeline explanation and give brief feedback."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"score": None, "feedback": "OPENAI_API_KEY is not set."}

    from openai import OpenAI

    client = OpenAI()
    prompt = build_judge_prompt(case, scored)

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=300,
        )
        text = response.output_text
    except Exception:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict but fair physics solution evaluator."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=300,
        )
        text = completion.choices[0].message.content or "{}"

    return parse_judge_response(text)


def evaluate_case(
    case: dict[str, Any],
    k: int,
    tolerance: float,
    max_tokens: int | None,
    max_formulas: int | None,
    compact_prompt: bool,
    enable_thinking: bool | None,
    direct_first: bool,
) -> dict[str, Any]:
    """Run one test case through the pipeline and score the result."""
    structured = run_structured(
        case["problem"],
        k=k,
        include_debug=True,
        max_tokens=max_tokens,
        max_formulas=max_formulas,
        compact_prompt=compact_prompt,
        enable_thinking=enable_thinking,
        direct_first=direct_first,
    )
    debug = structured["debug"]
    predicted_answer = debug["raw_answer"]
    predicted_unit = debug["unit"]
    numeric_ok = answer_matches(predicted_answer, case["expected_answer"], tolerance)
    unit_ok = unit_matches(predicted_unit, case)
    rel_error = relative_error(predicted_answer, case["expected_answer"])
    return {
        "id": case["id"],
        "difficulty": case["difficulty"],
        "domain": case["domain"],
        "expected_answer": case["expected_answer"],
        "expected_unit": case["expected_unit"],
        "predicted_answer": predicted_answer,
        "predicted_unit": predicted_unit,
        "confidence": structured["confidence"],
        "relative_error": rel_error,
        "tolerance": tolerance,
        "pass_reason": "within_relative_tolerance" if numeric_ok and unit_ok else "failed_numeric_or_unit_check",
        "explanation": structured["explanation"],
        "formulas_used": debug["formulas_used"],
        "all_answers": debug["all_answers"],
        "structured_output": {key: value for key, value in structured.items() if key != "debug"},
        "numeric_ok": numeric_ok,
        "unit_ok": unit_ok,
        "passed": numeric_ok and unit_ok,
    }


def write_evaluation_log(log_data: dict[str, Any]) -> Path:
    """Write a timestamped JSON evaluation log and return its path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    path = LOG_DIR / f"evaluation_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as log_file:
        json.dump(log_data, log_file, indent=2, ensure_ascii=False)
        log_file.write("\n")
    return path


def main() -> None:
    """Run selected evaluation cases and print a compact report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["circuit", "electrostatics", "energy"], help="Only run one domain.")
    parser.add_argument("--difficulty", choices=["simple", "moderate", "complex"], help="Only run one difficulty.")
    parser.add_argument("--limit", type=int, help="Only run the first N selected cases.")
    parser.add_argument("--k", type=int, default=5, help="Self-consistency sample count.")
    parser.add_argument("--max-tokens", type=int, help="Override max output tokens for faster runs.")
    parser.add_argument("--max-formulas", type=int, help="Limit formula references in the prompt.")
    parser.add_argument("--compact", action="store_true", help="Use the shorter latency-oriented prompt.")
    parser.add_argument("--no-thinking", action="store_true", help="Disable Qwen/DeepSeek thinking when supported by vLLM.")
    parser.add_argument("--direct-first", action="store_true", help="Try deterministic formula solvers before calling vLLM.")
    parser.add_argument("--fast", action="store_true", help="Shortcut for --direct-first --k 1 --max-tokens 512 --max-formulas 8 --compact --no-thinking.")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Relative numeric tolerance for scoring.")
    parser.add_argument("--judge", action="store_true", help="Use OpenAI to grade reasoning quality.")
    parser.add_argument("--judge-failures-only", action="store_true", help="Only call the OpenAI judge for failed cases.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="OpenAI model used when --judge is set.")
    args = parser.parse_args()

    cases = get_test_cases(domain=args.domain, difficulty=args.difficulty)
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("No test cases selected.")
        return

    k = 1 if args.fast else args.k
    max_tokens = 512 if args.fast and args.max_tokens is None else args.max_tokens
    max_formulas = 8 if args.fast and args.max_formulas is None else args.max_formulas
    compact_prompt = args.compact or args.fast
    enable_thinking = False if args.no_thinking or args.fast else None
    direct_first = args.direct_first or args.fast

    run_started_at = datetime.now(UTC)
    run_started_monotonic = time.monotonic()
    log_cases: list[dict[str, Any]] = []
    passed = 0
    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case['id']}")
        case_started_at = datetime.now(UTC)
        case_started_monotonic = time.monotonic()
        try:
            scored = evaluate_case(
                case,
                k=k,
                tolerance=args.tolerance,
                max_tokens=max_tokens,
                max_formulas=max_formulas,
                compact_prompt=compact_prompt,
                enable_thinking=enable_thinking,
                direct_first=direct_first,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            log_cases.append(
                {
                    "index": index,
                    "case": case,
                    "started_at": case_started_at.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": time.monotonic() - case_started_monotonic,
                    "error": str(exc),
                }
            )
            continue

        status = "PASS" if scored["passed"] else "FAIL"
        passed += int(scored["passed"])
        print(f"{status} domain={scored['domain']} difficulty={scored['difficulty']}")
        print(f"expected={scored['expected_answer']} {scored['expected_unit']}")
        print(f"predicted={scored['predicted_answer']} {scored['predicted_unit']} confidence={scored['confidence']:.2f}")
        should_judge = args.judge and (not args.judge_failures_only or not scored["passed"])
        if should_judge:
            judge = call_openai_judge(case, scored, model=args.judge_model)
            scored["judge"] = judge
            print(f"judge_score={judge['score']}/100")
            print(f"feedback={judge['feedback']}")

        structured_output = scored.pop("structured_output")
        log_cases.append(
            {
                "index": index,
                "case": case,
                "started_at": case_started_at.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.monotonic() - case_started_monotonic,
                "scoring": scored,
                "structured_output": structured_output,
            }
        )

    print(f"\nScore: {passed}/{len(cases)} passed")
    run_finished_at = datetime.now(UTC)
    log_data = {
        "run": {
            "started_at": run_started_at.isoformat(),
            "finished_at": run_finished_at.isoformat(),
            "duration_seconds": time.monotonic() - run_started_monotonic,
            "total_cases": len(cases),
            "passed_cases": passed,
            "failed_cases": len(cases) - passed,
            "accuracy": passed / len(cases),
            "settings": {
                "domain": args.domain,
                "difficulty": args.difficulty,
                "limit": args.limit,
                "k": k,
                "max_tokens": max_tokens,
                "max_formulas": max_formulas,
                "compact_prompt": compact_prompt,
                "enable_thinking": enable_thinking,
                "direct_first": direct_first,
                "fast": args.fast,
                "tolerance": args.tolerance,
                "judge": args.judge,
                "judge_failures_only": args.judge_failures_only,
                "judge_model": args.judge_model if args.judge else None,
            },
        },
        "cases": log_cases,
    }
    log_path = write_evaluation_log(log_data)
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
