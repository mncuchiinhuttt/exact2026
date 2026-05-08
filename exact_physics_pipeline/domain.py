"""Rule-based domain classification for physics problems."""


DOMAIN_KEYWORDS = {
    "circuit": [
        "resistor",
        "circuit",
        "current",
        "voltage",
        "ohm",
        "kirchhoff",
        "series",
        "parallel",
    ],
    "electrostatics": [
        "charge",
        "electric field",
        "coulomb",
        "capacitor",
        "capacitance",
        "potential",
        "conductor",
    ],
    "energy": [
        "energy",
        "power",
        "work",
        "joule",
        "stored",
        "dissipated",
    ],
}


def classify_domain(problem: str) -> str:
    """Classify a physics problem into circuit, electrostatics, energy, or general."""
    text = problem.lower()
    scores = {
        domain: sum(1 for keyword in keywords if keyword.lower() in text)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    if "what charge" in text or "charge is stored" in text:
        return "electrostatics"
    if scores["energy"]:
        return "energy"
    if any(keyword in text for keyword in ["capacitor", "capacitance", "coulomb", "electric field", "point charge"]):
        return "electrostatics"
    if scores["circuit"]:
        return "circuit"
    if scores["electrostatics"]:
        return "electrostatics"
    return "general"
