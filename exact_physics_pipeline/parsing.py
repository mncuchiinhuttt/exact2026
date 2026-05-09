"""Answer parsing helpers for model outputs."""

import re
from typing import Any


UNIT_PATTERN = r"(?:N\s*/\s*C|V\s*/\s*m|ohms?|volts?|amps?|amperes?|farads?|joules?|newtons?|[AVFJNCΩ]\b)"
ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.{0,220})", re.IGNORECASE | re.DOTALL)
NUMERIC_UNIT_RE = re.compile(
    rf"([-+]?(?:\d+(?:,\d{{3}})*(?:\.\d*)?|\.\d+)(?:\s*(?:e|E)\s*[-+]?\d+)?(?:\s*(?:x|×|\*)\s*10\s*\^?\s*[-+]?\d+)?)\s*({UNIT_PATTERN})",
    re.IGNORECASE,
)


UNIT_ALIASES = {
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "volt": "V",
    "volts": "V",
    "farad": "F",
    "farads": "F",
    "joule": "J",
    "joules": "J",
    "newton": "N",
    "newtons": "N",
    "ohm": "Ω",
    "ohms": "Ω",
}


def normalize_text_for_parsing(text: str) -> str:
    """Normalize common model answer formatting before regex parsing."""
    normalized = text.replace("−", "-").replace("–", "-")
    normalized = normalized.replace("\\,", " ").replace("\\;", " ").replace("~", " ")
    normalized = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = normalized.replace("\\mathrm", "")
    normalized = normalized.replace("{", " ").replace("}", " ")
    normalized = normalized.replace("\\", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_numeric(value_text: str) -> float:
    """Parse decimal, scientific, or x10^ numeric text into a float."""
    cleaned = value_text.replace(",", "").replace(" ", "")
    scientific_match = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:x|×|\*)10\^?([-+]?\d+)", cleaned)
    if scientific_match:
        base, exponent = scientific_match.groups()
        return float(base) * (10 ** int(exponent))
    return float(cleaned)


def normalize_parsed_unit(unit: str) -> str:
    """Normalize common written unit names to compact SI symbols."""
    cleaned = re.sub(r"\s+", "", unit.strip())
    lowered = cleaned.lower()
    if lowered in {"n/c", "v/m"}:
        return cleaned.upper().replace("V/M", "V/m")
    return UNIT_ALIASES.get(lowered, cleaned)


def parse_answer_text(text: str | None) -> dict[str, Any] | None:
    """Parse an ANSWER line from text and return a structured answer record."""
    if not text:
        return None
    normalized = normalize_text_for_parsing(text)
    answer_line = ANSWER_LINE_RE.search(normalized)
    search_text = answer_line.group(1) if answer_line else normalized
    matches = list(NUMERIC_UNIT_RE.finditer(search_text))
    if not matches:
        return None

    match = matches[-1]
    value_text, unit = match.groups()
    try:
        value = parse_numeric(value_text)
    except ValueError:
        return None
    return {"answer": value, "unit": normalize_parsed_unit(unit), "raw": match.group(0)}


def parse_generation_answer(final_text: str | None, think_text: str | None) -> dict[str, Any]:
    """Parse an answer from final text, falling back to the end of reasoning text."""
    parsed = parse_answer_text(final_text)
    if parsed is not None:
        return parsed

    fallback_text = think_text[-1200:] if think_text else None
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
