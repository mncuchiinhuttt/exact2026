"""Prompt builders for EXACT 2026 subtasks."""

from __future__ import annotations

from exact2026.pipeline.formula_db import format_formula_table


def build_physics_prompt(question: str, formulas: list[dict], mode: str = "full") -> list[dict[str, str]]:
    """Build OpenAI-compatible messages for Subtask 2 physics."""
    if mode == "compact":
        system = "Solve the physics problem. Show formula ID and computation. End with: ANSWER: <value> <unit>"
    else:
        system = f"""You are an expert physics tutor solving university-level problems.

Reference formulas:
{format_formula_table(formulas)}

Physical constants:
- k = 8.99×10^9 N·m²/C² (Coulomb constant)
- ε₀ = 8.85×10^-12 F/m (permittivity of free space)
- e = 1.6×10^-19 C (elementary charge)

Solve using EXACTLY this structure:

COT:
- Step 0 (Tool call - calculate): Identify which numeric computation will be offloaded to the calculator tool.
- Step 1: Extract all given quantities with values and SI units.
- Step 2: Convert all values to SI base units.
- Step 3: Identify and state the formula to use (reference formula ID if applicable).
- Step 4: Substitute values and compute.
- Step 5: State the final answer.

EXPLANATION:
[1-2 sentences summarizing the solution in plain language for a student.
 Do NOT repeat the step-by-step format. Do NOT include raw formulas or "Step N" markers.
 This section should read as a human-friendly summary, not a technical walkthrough.]

ANSWER: <numeric_value> <SI_unit>

The ANSWER line must be the very last line."""
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def build_edu_prompt(premises: list[str], question: str, mode: str = "full") -> list[dict[str, str]]:
    """Build OpenAI-compatible messages for Subtask 1 logic/education."""
    numbered = "\n".join(f"{index + 1}. {premise}" for index, premise in enumerate(premises))
    if mode == "compact":
        system = "Answer the educational-regulations logic question. End with: ANSWER: <Yes/No/True/False/A/B/C/D>"
    else:
        system = """You are an educational regulations reasoning assistant.
Given a set of premises and a question, reason step by step to find the answer.

Respond using EXACTLY this structure:

COT:
- Step 0 (Tool call - Z3): [PLACEHOLDER — symbolic verification not yet implemented. Will encode premises as FOL and check entailment.]
- Step 1: Identify which premises are relevant to the question.
- Step 2: Reason through the implications.
- Step 3: Reach a conclusion.

EXPLANATION:
[2-4 sentences explaining the answer in plain language. No Step markers. No FOL notation.
 This should be readable by someone unfamiliar with formal logic.]

ANSWER: <Yes/No/True/False/A/B/C/D>"""
    user = f"Premises:\n{numbered}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

