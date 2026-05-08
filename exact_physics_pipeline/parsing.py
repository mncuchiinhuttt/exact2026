"""Answer parsing helpers for model outputs."""

import re
from typing import Any


ANSWER_RE = re.compile(r"ANSWER:\s*([\d\.\+\-eE]+)\s*([A-Za-zΩμ°²³/·]+)")


def parse_answer_text(text: str | None) -> dict[str, Any] | None:
    """Parse an ANSWER line from text and return a structured answer record."""
    if not text:
        return None
    match = ANSWER_RE.search(text)
    if not match:
        return None
    value_text, unit = match.groups()
    try:
        value = float(value_text)
    except ValueError:
        return None
    return {"answer": value, "unit": unit, "raw": match.group(0)}


def parse_generation_answer(final_text: str | None, think_text: str | None) -> dict[str, Any]:
    """Parse an answer from final text, falling back to the end of reasoning text."""
    parsed = parse_answer_text(final_text)
    if parsed is not None:
        return parsed

    fallback_text = think_text[-300:] if think_text else None
    parsed = parse_answer_text(fallback_text)
    if parsed is not None:
        parsed["source"] = "reasoning_fallback"
        return parsed

    return {"answer": None, "unit": None, "error": "parse_failed"}


def extract_step_section(text: str | None) -> str:
    """Extract the STEP 1 through STEP 4 section when present."""
    if not text:
        return ""
    match = re.search(r"(STEP 1\s*-\s*READ:.*?STEP 4\s*-\s*ANSWER:.*)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""
