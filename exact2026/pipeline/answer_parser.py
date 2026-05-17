"""Answer and unit extraction for model generations."""

from __future__ import annotations

import re
from typing import Any

from exact2026.pipeline.unit_normalizer import expand_prefix, normalize_unit


VALUE_RE = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*[×x*]\s*10\^?[-+]?\d+)?"
UNIT_RE = r"[A-Za-zΩμ°²³/·\-]+"
ANSWER_RE = re.compile(rf"ANSWER:\s*({VALUE_RE})\s*({UNIT_RE})?", re.IGNORECASE)


def normalize_value_string(s: str) -> float:
    """Parse decimal, exponent, and multiplication-by-ten notation."""
    cleaned = s.strip().replace("−", "-").replace(",", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    match = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:[×x*])10\^?([-+]?\d+)", cleaned)
    if match:
        base, exponent = match.groups()
        return float(base) * (10 ** int(exponent))
    return float(cleaned)


def _record(value_raw: str | None, unit_raw: str | None, source: str) -> dict[str, Any]:
    """Build a normalized parse result record."""
    value = None
    value_si = None
    unit = normalize_unit(unit_raw or "")
    if value_raw is not None:
        try:
            value = normalize_value_string(value_raw)
            value_si, unit = expand_prefix(value, unit_raw or "")
        except ValueError:
            value = None
            value_si = None
    return {
        "value_raw": value_raw or "",
        "unit_raw": unit_raw or "",
        "value": value,
        "unit": unit,
        "value_si": value_si,
        "source": source,
    }


def _search(text: str, source: str) -> dict[str, Any] | None:
    """Search text with the ANSWER regex and return the last match."""
    matches = list(ANSWER_RE.finditer(text or ""))
    if not matches:
        return None
    value_raw, unit_raw = matches[-1].groups()
    return _record(value_raw, unit_raw, source)


def parse_answer(text: str, reasoning: str | None = None) -> dict[str, Any]:
    """Extract a numeric ANSWER line, falling back to tails and last-line numbers."""
    parsed = _search(text or "", "answer_line")
    if parsed:
        return parsed
    parsed = _search((text or "")[-500:], "text_tail")
    if parsed:
        return parsed
    if reasoning is not None:
        parsed = _search(reasoning[-500:], "reasoning_tail")
        if parsed:
            return parsed

    tail_lines = "\n".join((text or "").splitlines()[-3:])
    number_matches = list(re.finditer(VALUE_RE, tail_lines))
    if number_matches:
        return _record(number_matches[-1].group(0), "", "last_number")
    return _record(None, None, "failed")

