"""Unit canonicalization and SI-prefix expansion utilities."""

from __future__ import annotations

import re


UNIT_MAP = {
    "v": "V",
    "volt": "V",
    "volts": "V",
    "a": "A",
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "ohm": "Ω",
    "ohms": "Ω",
    "ω": "Ω",
    "Ω": "Ω",
    "f": "F",
    "farad": "F",
    "farads": "F",
    "j": "J",
    "joule": "J",
    "joules": "J",
    "n": "N",
    "newton": "N",
    "newtons": "N",
    "c": "C",
    "coulomb": "C",
    "coulombs": "C",
    "w": "W",
    "watt": "W",
    "watts": "W",
    "n/c": "N/C",
    "v/m": "V/m",
    "hz": "Hz",
    "hertz": "Hz",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "1": "1",
    "": "1",
    "-": "1",
    "—": "1",
    "–": "1",
}

SI_PREFIX_MAP = {
    "p": 1e-12,
    "n": 1e-9,
    "μ": 1e-6,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}


def _clean_unit(raw_unit: str | None) -> str:
    """Return a whitespace-normalized unit string."""
    return re.sub(r"\s+", "", str(raw_unit or "").strip())


def normalize_unit(raw_unit: str | None) -> str:
    """Return a canonical SI-ish unit symbol without changing numeric scale."""
    cleaned = _clean_unit(raw_unit)
    lowered = cleaned.lower()
    if lowered in UNIT_MAP:
        return UNIT_MAP[lowered]

    for prefix in sorted(SI_PREFIX_MAP, key=len, reverse=True):
        if not cleaned.startswith(prefix) or len(cleaned) == len(prefix):
            continue
        base = cleaned[len(prefix) :]
        canonical = UNIT_MAP.get(base.lower())
        if canonical and canonical != "1":
            return f"{prefix}{canonical}"
    return cleaned


def expand_prefix(value: float, raw_unit: str | None) -> tuple[float, str]:
    """Apply an SI prefix in raw_unit to value and return base-unit value/unit."""
    cleaned = _clean_unit(raw_unit)
    canonical = normalize_unit(cleaned)
    for prefix, factor in SI_PREFIX_MAP.items():
        if not cleaned.startswith(prefix) or len(cleaned) == len(prefix):
            continue
        base = cleaned[len(prefix) :]
        base_unit = UNIT_MAP.get(base.lower())
        if base_unit and base_unit != "1":
            return value * factor, base_unit
    return value, UNIT_MAP.get(canonical.lower(), canonical)

