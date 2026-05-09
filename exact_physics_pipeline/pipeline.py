"""Top-level orchestration for the EXACT 2026 physics pipeline."""

from typing import Any

from exact_physics_pipeline import config
from exact_physics_pipeline.direct_solver import try_direct_solve
from exact_physics_pipeline.domain import classify_domain
from exact_physics_pipeline.formulas import DEFAULT_FAST_FORMULA_LIMIT, retrieve_formulas
from exact_physics_pipeline.inference import run_self_consistency
from exact_physics_pipeline.prompts import build_messages, build_system_prompt
from exact_physics_pipeline.voting import choose_consensus, normalize_unit


def _same_answer_family(model_answer: dict[str, Any], direct_result: dict[str, Any], tolerance: float = 0.05) -> bool:
    """Return whether a model answer is close enough for deterministic refinement."""
    model_value = model_answer.get("answer")
    direct_value = direct_result.get("answer")
    if model_value is None or direct_value is None:
        return False
    if normalize_unit(model_answer.get("unit")) != normalize_unit(direct_result.get("unit")):
        return False
    scale = max(abs(float(model_value)), abs(float(direct_value)), 1e-12)
    return abs(float(model_value) - float(direct_value)) <= tolerance * scale


def _apply_direct_refinement(result: dict[str, Any], direct_result: dict[str, Any] | None) -> dict[str, Any]:
    """Use deterministic arithmetic to clean up rounded or explanation-less model results."""
    if direct_result is None:
        return result
    if result["answer"] is not None and not _same_answer_family(result, direct_result):
        return result

    refined = dict(result)
    refined.update(
        {
            "formulas_used": direct_result["formulas_used"],
            "answer": direct_result["answer"],
            "unit": direct_result["unit"],
            "confidence": direct_result["confidence"],
            "explanation": direct_result["explanation"],
            "raw_think": result.get("raw_think", ""),
            "source": "direct_refinement",
            "model_answer": {
                "answer": result["answer"],
                "unit": result["unit"],
                "confidence": result["confidence"],
            },
        }
    )
    refined["all_answers"] = [
        *result.get("all_answers", []),
        {
            "answer": direct_result["answer"],
            "unit": direct_result["unit"],
            "raw": f"DIRECT_REFINEMENT: {direct_result['answer']} {direct_result['unit']}",
            "source": "direct_refinement",
        },
    ]
    return refined


def run_pipeline(
    problem: str,
    k: int = config.DEFAULT_K,
    max_tokens: int | None = None,
    max_formulas: int | None = None,
    compact_prompt: bool = False,
    enable_thinking: bool | None = None,
    direct_first: bool = False,
) -> dict[str, Any]:
    """Run the full physics reasoning pipeline for one problem."""
    domain = classify_domain(problem)
    direct_result = try_direct_solve(problem, domain)
    if direct_first:
        if direct_result is not None:
            return direct_result

    if compact_prompt and max_formulas is None:
        max_formulas = DEFAULT_FAST_FORMULA_LIMIT
    formulas = retrieve_formulas(domain, problem=problem, max_formulas=max_formulas)
    system_prompt = build_system_prompt(formulas, compact=compact_prompt)
    messages = build_messages(problem, system_prompt)

    generations = run_self_consistency(
        messages,
        k=k,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    parsed_answers = [generation["parsed_answer"] for generation in generations]
    if parsed_answers and all(str(answer.get("error", "")).startswith("inference_failed") for answer in parsed_answers):
        first_error = parsed_answers[0].get("error", "inference_failed")
        raise RuntimeError(first_error)

    consensus = choose_consensus(parsed_answers, k=k)

    winner_index = consensus.get("winner_index")
    winning_generation = generations[winner_index] if winner_index is not None else None
    explanation = winning_generation["final_text"] if winning_generation else ""
    raw_think = winning_generation["think_text"] if winning_generation else ""

    result = {
        "domain": domain,
        "formulas_used": [formula["id"] for formula in formulas],
        "answer": consensus["answer"],
        "unit": consensus["unit"],
        "confidence": consensus["confidence"],
        "explanation": explanation,
        "raw_think": raw_think,
        "all_answers": consensus["all_answers"],
    }
    return _apply_direct_refinement(result, direct_result)
