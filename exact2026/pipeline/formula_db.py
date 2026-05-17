"""Static physics formula database and keyword retrieval."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from exact_physics_pipeline.formulas import FORMULA_DB as _BASE_FORMULA_DB


Formula = dict[str, Any]


def _default_keywords(formula: Formula) -> list[str]:
    """Infer formula keywords from id, name, formula, variables, and conditions."""
    text = " ".join(
        [
            formula.get("id", "").replace("_", " "),
            formula.get("name", ""),
            formula.get("formula", ""),
            formula.get("conditions", ""),
            " ".join(formula.get("variables", {}).values()),
        ]
    )
    return sorted({token.lower() for token in re.findall(r"[A-Za-z]{4,}", text)})


def _latex(formula_text: str) -> str:
    """Return a lightweight LaTeX-ready formula string."""
    return formula_text.replace("*", r"\cdot ").replace("ε0", r"\epsilon_0")


FORMULA_DB: dict[str, Formula] = deepcopy(_BASE_FORMULA_DB)
for formula in FORMULA_DB.values():
    formula.setdefault("domains", ["general"])
    formula.setdefault("keywords", _default_keywords(formula))
    formula.setdefault("latex", _latex(formula.get("formula", "")))


def _score(formula: Formula, problem_text: str) -> int:
    """Score a formula by keyword overlap with problem text."""
    tokens = set(re.findall(r"[a-zA-Z]+", problem_text.lower()))
    score = len(tokens.intersection(set(formula.get("keywords", []))))
    for keyword in formula.get("keywords", []):
        if keyword in problem_text.lower() and len(keyword) >= 5:
            score += 1
    return score


def retrieve_formulas(domain: str, problem_text: str, max_formulas: int = 8) -> list[Formula]:
    """Return top formulas for a domain using simple keyword-overlap ranking."""
    candidates = [
        formula
        for formula in FORMULA_DB.values()
        if domain == "general" or domain in formula.get("domains", [])
    ]
    ranked = sorted(candidates, key=lambda formula: _score(formula, problem_text), reverse=True)
    return ranked[:max_formulas]


def format_formula_table(formulas: list[Formula]) -> str:
    """Format formulas as a prompt reference table."""
    rows = ["| ID | Name | Formula | Variables | Conditions |", "|---|---|---|---|---|"]
    for formula in formulas:
        variables = "; ".join(f"{key}: {value}" for key, value in formula.get("variables", {}).items())
        rows.append(
            f"| {formula['id']} | {formula['name']} | {formula['formula']} | {variables} | {formula['conditions']} |"
        )
    return "\n".join(rows)

