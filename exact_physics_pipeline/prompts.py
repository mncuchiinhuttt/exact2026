"""Prompt construction for the physics reasoning pipeline."""

from exact_physics_pipeline.formulas import Formula, format_formula_table


def build_system_prompt(formulas: list[Formula]) -> str:
    """Build the system prompt with formula references and answer format constraints."""
    formula_table = format_formula_table(formulas)
    return f"""You are an expert physics tutor solving university-level problems.

Use the following physics formula reference table for this problem domain:

{formula_table}

Use these constants and unit conversions exactly unless the problem gives a different value:
- Coulomb constant: k = 8.99e9 N·m²/C²
- Vacuum permittivity: ε0 = 8.854e-12 F/m
- micro = 1e-6
- milli = 1e-3
- kilo = 1e3

Follow this strict 4-step reasoning structure:

STEP 1 - READ: Extract all given quantities with values and SI units. Identify the unknown.
STEP 2 - PLAN: Select the relevant formula(s) from the reference. State why each applies.
STEP 3 - SOLVE: Write Python-style pseudocode or arithmetic to calculate the answer. Show every substitution.
STEP 4 - ANSWER: State the final answer on its own line in EXACTLY this format:
ANSWER: <numeric_value> <SI_unit>

Do not over-round the final numeric value. Keep at least 3 significant figures, and use scientific notation when it makes the answer clearer.
For multi-part questions, put the primary requested value in the ANSWER line and include any additional requested values in the explanation.
Examples:
ANSWER: 4.5e-6 F
ANSWER: 12.0 V
"""


def build_messages(problem: str, system_prompt: str) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for vLLM."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
