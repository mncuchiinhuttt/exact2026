"""Full Subtask 2 physics inference pipeline."""

from __future__ import annotations

import re
from typing import Any

from exact2026.pipeline.answer_parser import parse_answer
from exact2026.pipeline.direct_solver import try_direct_solve
from exact2026.pipeline.domain_classifier import classify_domain
from exact2026.pipeline.formula_db import retrieve_formulas
from exact2026.pipeline.inference import VLLMClient
from exact2026.pipeline.prompts import build_physics_prompt
from exact2026.pipeline.voting import vote


def extract_cot_block(text: str) -> str:
    """Extract the COT section without overlapping EXPLANATION or ANSWER."""
    match = re.search(r"COT:\s*(.*?)(?:\n\s*EXPLANATION:|\n\s*ANSWER:|\Z)", text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_explanation_block(text: str) -> str:
    """Extract the EXPLANATION section without overlapping ANSWER."""
    match = re.search(r"EXPLANATION:\s*(.*?)(?:\n\s*ANSWER:|\Z)", text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _matches(parsed: dict[str, Any], vote_result: dict[str, Any], tolerance: float = 0.03) -> bool:
    """Return whether a parsed answer belongs to the winning cluster."""
    value = parsed.get("value_si") if parsed.get("value_si") is not None else parsed.get("value")
    if value is None or vote_result.get("answer") is None:
        return False
    if (parsed.get("unit") or "1") != vote_result.get("unit"):
        return False
    scale = max(abs(float(value)), abs(float(vote_result["answer"])), 1e-12)
    return abs(float(value) - float(vote_result["answer"])) <= tolerance * scale


def _format_direct(result: dict[str, Any], source: str) -> dict[str, Any]:
    """Convert a direct-solver result to the new pipeline schema."""
    explanation = result.get("explanation", "")
    return {
        "answer": result.get("answer"),
        "unit": result.get("unit", ""),
        "confidence": result.get("confidence", 1.0),
        "explanation": explanation,
        "cot": "\n".join(line for line in explanation.splitlines() if line.startswith("STEP")),
        "raw_think": "",
        "domain": result.get("domain", "general"),
        "formulas_used": result.get("formulas_used", []),
        "source": source,
        "all_answers": result.get("all_answers", []),
    }


def run_physics(question: str, config: dict | None = None) -> dict[str, Any]:
    """Run Subtask 2 with optional direct solving and vLLM self-consistency."""
    cfg = {"k": 5, "mode": "full", "fast": False, "direct_first": True, "max_tokens": 4096}
    cfg.update(config or {})
    if cfg["fast"]:
        cfg.update({"k": 1, "mode": "compact"})

    domain = classify_domain(question)
    direct_result = try_direct_solve(question, domain) if cfg.get("direct_first", True) else None
    if direct_result and cfg.get("fast"):
        return _format_direct(direct_result, "direct_solver")

    formulas = retrieve_formulas(domain, question, max_formulas=int(cfg.get("max_formulas", 8)))
    messages = build_physics_prompt(question, formulas, mode=cfg["mode"])
    generations = VLLMClient().generate_k(
        messages,
        k=int(cfg["k"]),
        max_tokens=int(cfg.get("max_tokens", 4096)),
        enable_thinking=not cfg.get("no_thinking", False),
    )
    parsed = [parse_answer(gen["content"], gen.get("reasoning")) for gen in generations]
    vote_result = vote(parsed)
    source = "vllm"

    if direct_result and vote_result["answer"] is not None and direct_result.get("unit") == vote_result.get("unit"):
        scale = max(abs(float(vote_result["answer"])), abs(float(direct_result["answer"])), 1e-12)
        if abs(float(vote_result["answer"]) - float(direct_result["answer"])) <= 0.05 * scale:
            direct = _format_direct(direct_result, "direct_refinement")
            direct["all_answers"] = parsed
            return direct

    winning_gen = next((gen for gen, ans in zip(generations, parsed) if _matches(ans, vote_result)), generations[0] if generations else {"content": "", "reasoning": ""})
    return {
        "answer": vote_result["answer"],
        "unit": vote_result["unit"],
        "confidence": vote_result["confidence"],
        "explanation": extract_explanation_block(winning_gen["content"]),
        "cot": extract_cot_block(winning_gen["content"]),
        "raw_think": winning_gen.get("reasoning", ""),
        "domain": domain,
        "formulas_used": [formula["id"] for formula in formulas],
        "source": source,
        "all_answers": parsed,
    }

