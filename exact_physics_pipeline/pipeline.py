"""Top-level orchestration for the EXACT 2026 physics pipeline."""

from typing import Any

from exact_physics_pipeline import config
from exact_physics_pipeline.domain import classify_domain
from exact_physics_pipeline.formulas import retrieve_formulas
from exact_physics_pipeline.inference import run_self_consistency
from exact_physics_pipeline.prompts import build_messages, build_system_prompt
from exact_physics_pipeline.voting import choose_consensus


def run_pipeline(problem: str, k: int = config.DEFAULT_K) -> dict[str, Any]:
    """Run the full physics reasoning pipeline for one problem."""
    domain = classify_domain(problem)
    formulas = retrieve_formulas(domain)
    system_prompt = build_system_prompt(formulas)
    messages = build_messages(problem, system_prompt)

    generations = run_self_consistency(messages, k=k)
    parsed_answers = [generation["parsed_answer"] for generation in generations]
    if parsed_answers and all(str(answer.get("error", "")).startswith("inference_failed") for answer in parsed_answers):
        first_error = parsed_answers[0].get("error", "inference_failed")
        raise RuntimeError(first_error)

    consensus = choose_consensus(parsed_answers, k=k)

    winner_index = consensus.get("winner_index")
    winning_generation = generations[winner_index] if winner_index is not None else None
    explanation = winning_generation["final_text"] if winning_generation else ""
    raw_think = winning_generation["think_text"] if winning_generation else ""

    return {
        "domain": domain,
        "formulas_used": [formula["id"] for formula in formulas],
        "answer": consensus["answer"],
        "unit": consensus["unit"],
        "confidence": consensus["confidence"],
        "explanation": explanation,
        "raw_think": raw_think,
        "all_answers": consensus["all_answers"],
    }
