"""Run the physics pipeline and emit contest-style structured JSON output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from exact_physics_pipeline import run_pipeline
from exact_physics_pipeline.formulas import FORMULA_DB


def extract_cot_steps(explanation: str) -> list[str]:
    """Extract visible STEP 1-4 lines from the final explanation."""
    if not explanation:
        return []
    normalized = re.sub(r"\*\*", "", explanation)
    markers = list(
        re.finditer(
            r"STEP\s+([1-4])\s*-\s*(READ|PLAN|SOLVE|ANSWER)\s*:",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if markers:
        steps = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(normalized)
            step_text = normalized[marker.start() : end]
            if index == len(markers) - 1:
                step_text = re.sub(r"\n\s*ANSWER\s*:.*\Z", "", step_text, flags=re.IGNORECASE | re.DOTALL)
            steps.append(" ".join(step_text.split()))
        return steps
    return [" ".join(line.split()) for line in explanation.splitlines() if line.strip()][:4]


def build_fol_statement(domain: str) -> str:
    """Return a compact first-order-logic style statement for the problem domain."""
    statements = {
        "circuit": "∀x (CircuitProblem(x) → Uses(OhmsLaw, x) ∨ Uses(KirchhoffLaw, x))",
        "electrostatics": "∀x (ElectrostaticsProblem(x) → Uses(CoulombLaw, x) ∨ Uses(CapacitanceRelations, x))",
        "energy": "∀x (EnergyProblem(x) → ConservesOrComputesEnergy(x))",
    }
    return statements.get(domain, "∀x (PhysicsProblem(x) → Requires(UnitConsistentFormulaApplication(x)))")


def build_premises(formula_ids: list[str]) -> list[str]:
    """Build human-readable premises from formulas selected by the pipeline."""
    premises = []
    for formula_id in formula_ids[:8]:
        formula = FORMULA_DB.get(formula_id)
        if formula:
            premises.append(f"{formula['name']}: {formula['formula']}")
    return premises


def build_structured_output(pipeline_result: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw pipeline result into contest-style structured output."""
    if pipeline_result["answer"] is None:
        answer = None
    else:
        answer = f"{pipeline_result['answer']} {pipeline_result['unit']}"

    return {
        "answer": answer,
        "explanation": pipeline_result.get("explanation", ""),
        "fol": build_fol_statement(pipeline_result["domain"]),
        "cot": extract_cot_steps(pipeline_result.get("explanation", "")),
        "premises": build_premises(pipeline_result.get("formulas_used", [])),
        "confidence": pipeline_result["confidence"],
    }


def run_structured(
    problem: str,
    k: int = 5,
    include_debug: bool = False,
    max_tokens: int | None = None,
    max_formulas: int | None = None,
    compact_prompt: bool = False,
    enable_thinking: bool | None = None,
    direct_first: bool = False,
) -> dict[str, Any]:
    """Run the pipeline and return structured output suitable for contest APIs."""
    pipeline_result = run_pipeline(
        problem,
        k=k,
        max_tokens=max_tokens,
        max_formulas=max_formulas,
        compact_prompt=compact_prompt,
        enable_thinking=enable_thinking,
        direct_first=direct_first,
    )
    structured = build_structured_output(pipeline_result)
    if include_debug:
        structured["debug"] = {
            "domain": pipeline_result["domain"],
            "formulas_used": pipeline_result["formulas_used"],
            "unit": pipeline_result["unit"],
            "raw_answer": pipeline_result["answer"],
            "raw_think": pipeline_result["raw_think"],
            "all_answers": pipeline_result["all_answers"],
            "source": pipeline_result.get("source", "vllm"),
            "model_answer": pipeline_result.get("model_answer"),
        }
    return structured


def main() -> None:
    """CLI entrypoint for one-off structured runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", help="Problem text. If omitted, stdin is used.")
    parser.add_argument("--k", type=int, default=5, help="Self-consistency sample count.")
    parser.add_argument("--max-tokens", type=int, help="Override max output tokens for faster runs.")
    parser.add_argument("--max-formulas", type=int, help="Limit formula references in the prompt.")
    parser.add_argument("--compact", action="store_true", help="Use the shorter latency-oriented prompt.")
    parser.add_argument("--no-thinking", action="store_true", help="Disable Qwen/DeepSeek thinking when supported by vLLM.")
    parser.add_argument("--direct-first", action="store_true", help="Try deterministic formula solvers before calling vLLM.")
    parser.add_argument("--fast", action="store_true", help="Shortcut for --direct-first --k 1 --max-tokens 512 --max-formulas 8 --compact --no-thinking.")
    parser.add_argument("--debug", action="store_true", help="Include raw pipeline debug fields.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of compact endpoint JSON.")
    args = parser.parse_args()

    problem = args.problem if args.problem is not None else sys.stdin.read().strip()
    if not problem:
        raise SystemExit("Provide --problem or pipe problem text on stdin.")

    k = 1 if args.fast else args.k
    max_tokens = 512 if args.fast and args.max_tokens is None else args.max_tokens
    max_formulas = 8 if args.fast and args.max_formulas is None else args.max_formulas
    compact_prompt = args.compact or args.fast
    enable_thinking = False if args.no_thinking or args.fast else None
    direct_first = args.direct_first or args.fast
    result = run_structured(
        problem,
        k=k,
        include_debug=args.debug,
        max_tokens=max_tokens,
        max_formulas=max_formulas,
        compact_prompt=compact_prompt,
        enable_thinking=enable_thinking,
        direct_first=direct_first,
    )
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
