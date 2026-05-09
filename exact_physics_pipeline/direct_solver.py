"""Deterministic fast-path solvers for common physics formula problems."""

from __future__ import annotations

import math
import re
from typing import Any


K_COULOMB = 8.99e9


def _numbers_before_unit(problem: str, unit_patterns: list[str]) -> list[float]:
    """Extract numeric values immediately followed by one of the unit patterns."""
    unit = "|".join(unit_patterns)
    values = []
    for match in re.finditer(rf"([-+]?\d+(?:\.\d+)?)\s*(?:{unit})\b", problem, flags=re.IGNORECASE):
        values.append(float(match.group(1)))
    return values


def _micro_values(problem: str, base_unit: str) -> list[float]:
    """Extract micro-prefixed values and convert them to SI base units."""
    return [
        float(match.group(1)) * 1e-6
        for match in re.finditer(rf"([-+]?\d+(?:\.\d+)?)\s*(?:micro{base_unit}|μ{base_unit})s?\b", problem, flags=re.IGNORECASE)
    ]


def _count_from_words(problem: str) -> int | None:
    """Extract a small count from digits or common number words."""
    text = problem.lower()
    for word, count in {"two": 2, "three": 3, "four": 4, "five": 5}.items():
        if word in text:
            return count
    match = re.search(r"\b([2-9])\b", text)
    return int(match.group(1)) if match else None


def _answer(value: float, unit: str, explanation: str, formulas: list[str], domain: str) -> dict[str, Any]:
    """Build a pipeline-compatible direct-solver result."""
    return {
        "domain": domain,
        "formulas_used": formulas,
        "answer": float(value),
        "unit": unit,
        "confidence": 1.0,
        "explanation": explanation,
        "raw_think": "",
        "all_answers": [{"answer": float(value), "unit": unit, "raw": f"DIRECT_SOLVER: {value} {unit}"}],
        "source": "direct_solver",
    }


def _format_value(value: float) -> str:
    """Format computed values without throwing away useful precision."""
    return f"{value:.12g}"


def _format_steps(read: str, plan: str, solve: str, value: float, unit: str) -> str:
    """Format direct-solver explanation with the same visible STEP structure."""
    formatted_value = _format_value(value)
    return "\n".join(
        [
            f"STEP 1 - READ: {read}",
            f"STEP 2 - PLAN: {plan}",
            f"STEP 3 - SOLVE: {solve}",
            f"STEP 4 - ANSWER: Final answer is {formatted_value} {unit}.",
            f"ANSWER: {formatted_value} {unit}",
        ]
    )


def try_direct_solve(problem: str, domain: str) -> dict[str, Any] | None:
    """Try to solve common formula problems without calling the LLM."""
    text = problem.lower()
    resistances = _numbers_before_unit(problem, ["ohm", "ohms", "Ω"])
    voltages = _numbers_before_unit(problem, ["v"])
    currents = _numbers_before_unit(problem, ["a"])
    powers = _numbers_before_unit(problem, ["w"])
    times = _numbers_before_unit(problem, ["s", "sec", "second", "seconds"])
    distances = _numbers_before_unit(problem, ["m", "meter", "meters"])
    energies = _numbers_before_unit(problem, ["j", "joule", "joules"])
    micro_coulombs = _micro_values(problem, "coulomb")
    micro_farads = _micro_values(problem, "farad")

    if domain == "circuit":
        if "node" in text and "enter" in text and "outgoing" in text:
            entering = [float(value) for value in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*a\s+enter", text)]
            if len(entering) < 2:
                entering = currents[:2]
            known_out = [float(value) for value in re.findall(r"carries\s+([-+]?\d+(?:\.\d+)?)\s*a", text)]
            if len(entering) >= 2 and known_out:
                value = sum(entering) - known_out[0]
                explanation = _format_steps(
                    f"Incoming currents are {entering}; known outgoing current is {known_out[0]} A.",
                    "Use Kirchhoff's current law.",
                    f"I_unknown = sum(I_in) - I_known_out = {sum(entering)} - {known_out[0]} = {value}",
                    value,
                    "A",
                )
                return _answer(value, "A", explanation, ["kirchhoff_current"], domain)

        if "series" in text and "parallel" in text and len(resistances) >= 3 and voltages:
            parallel_equivalent = 1.0 / sum(1.0 / resistance for resistance in resistances[1:])
            total = resistances[0] + parallel_equivalent
            value = voltages[0] / total
            explanation = _format_steps(
                f"Voltage is {voltages[0]} V; series resistor is {resistances[0]} ohm; parallel pair is {resistances[1:]} ohm.",
                "Reduce the parallel branch, add the series resistor, then use Ohm's law.",
                f"R_parallel = {parallel_equivalent}; R_total = {resistances[0]} + {parallel_equivalent} = {total}; I = {voltages[0]}/{total} = {value}",
                value,
                "A",
            )
            return _answer(value, "A", explanation, ["parallel_resistance", "series_resistance", "ohms_law"], domain)

        if "parallel" in text and resistances and voltages:
            equivalent = 1.0 / sum(1.0 / resistance for resistance in resistances)
            value = voltages[0] / equivalent
            explanation = _format_steps(
                f"Voltage is {voltages[0]} V; parallel resistors are {resistances} ohm.",
                "Find equivalent parallel resistance, then use Ohm's law.",
                f"R_eq = 1/sum(1/R_i) = {equivalent}; I = V/R_eq = {voltages[0]}/{equivalent} = {value}",
                value,
                "A",
            )
            return _answer(value, "A", explanation, ["parallel_resistance", "ohms_law"], domain)

        if "series" in text and resistances and voltages:
            total = sum(resistances)
            value = voltages[0] / total
            explanation = _format_steps(
                f"Voltage is {voltages[0]} V; series resistors are {resistances} ohm.",
                "Add series resistances, then use Ohm's law.",
                f"R_total = {total}; I = V/R_total = {voltages[0]}/{total} = {value}",
                value,
                "A",
            )
            return _answer(value, "A", explanation, ["series_resistance", "ohms_law"], domain)

        if resistances and voltages and "current" in text:
            value = voltages[0] / resistances[0]
            explanation = _format_steps(
                f"Voltage is {voltages[0]} V and resistance is {resistances[0]} ohm.",
                "Use Ohm's law.",
                f"I = V/R = {voltages[0]}/{resistances[0]} = {value}",
                value,
                "A",
            )
            return _answer(value, "A", explanation, ["ohms_law"], domain)

        if voltages and currents and "resistance" in text:
            value = voltages[0] / currents[0]
            explanation = _format_steps(
                f"Voltage is {voltages[0]} V and current is {currents[0]} A.",
                "Rearrange Ohm's law.",
                f"R = V/I = {voltages[0]}/{currents[0]} = {value}",
                value,
                "Ω",
            )
            return _answer(value, "Ω", explanation, ["ohms_law"], domain)

        if resistances and currents and "voltage" in text:
            value = currents[0] * resistances[0]
            explanation = _format_steps(
                f"Resistance is {resistances[0]} ohm and current is {currents[0]} A.",
                "Use Ohm's law.",
                f"V = I*R = {currents[0]}*{resistances[0]} = {value}",
                value,
                "V",
            )
            return _answer(value, "V", explanation, ["ohms_law"], domain)

    if domain == "electrostatics":
        if "electric field" in text and micro_coulombs and distances:
            value = K_COULOMB * abs(micro_coulombs[0]) / (distances[0] ** 2)
            explanation = _format_steps(
                f"Point charge is {micro_coulombs[0]} C and distance is {distances[0]} m.",
                "Use point-charge electric field.",
                f"E = kQ/r^2 = {K_COULOMB}*{abs(micro_coulombs[0])}/{distances[0]}^2 = {value}",
                value,
                "N/C",
            )
            return _answer(value, "N/C", explanation, ["electric_field_point"], domain)

        if "potential" in text and micro_coulombs and distances:
            if len(micro_coulombs) >= 2 and len(distances) == 1:
                value = K_COULOMB * sum(micro_coulombs) / distances[0]
            else:
                value = K_COULOMB * micro_coulombs[0] / distances[0]
            explanation = _format_steps(
                f"Charges are {micro_coulombs} C and distance is {distances[0]} m.",
                "Use scalar electric potential superposition.",
                f"V = k*sum(q_i/r_i) = {value}",
                value,
                "V",
            )
            return _answer(value, "V", explanation, ["electric_potential", "electric_potential_superposition"], domain)

        if "force" in text and len(micro_coulombs) >= 2 and distances:
            value = K_COULOMB * abs(micro_coulombs[0] * micro_coulombs[1]) / (distances[0] ** 2)
            explanation = _format_steps(
                f"Charges are {micro_coulombs[:2]} C and separation is {distances[0]} m.",
                "Use Coulomb's law.",
                f"F = k*q1*q2/r^2 = {value}",
                value,
                "N",
            )
            return _answer(value, "N", explanation, ["coulombs_law"], domain)

        if "capacitance" in text and "parallel" in text and micro_farads:
            value = sum(micro_farads)
            explanation = _format_steps(
                f"Parallel capacitances are {micro_farads} F.",
                "Add parallel capacitors.",
                f"C_total = sum(C_i) = {value}",
                value,
                "F",
            )
            return _answer(value, "F", explanation, ["parallel_capacitance"], domain)

        if "capacitance" in text and "series" in text and micro_farads:
            if "identical" in text and len(micro_farads) == 1:
                count = _count_from_words(problem)
                if count:
                    micro_farads = micro_farads * count
            value = 1.0 / sum(1.0 / capacitance for capacitance in micro_farads)
            explanation = _format_steps(
                f"Series capacitances are {micro_farads} F.",
                "Use reciprocal sum for series capacitors.",
                f"C_total = 1/sum(1/C_i) = {value}",
                value,
                "F",
            )
            return _answer(value, "F", explanation, ["series_capacitance"], domain)

        if "charge" in text and micro_farads and voltages:
            value = micro_farads[0] * voltages[0]
            explanation = _format_steps(
                f"Capacitance is {micro_farads[0]} F and voltage is {voltages[0]} V.",
                "Use capacitor charge relation.",
                f"Q = C*V = {micro_farads[0]}*{voltages[0]} = {value}",
                value,
                "C",
            )
            return _answer(value, "C", explanation, ["capacitor_charge"], domain)

        if "capacitance" in text and micro_coulombs and voltages:
            value = micro_coulombs[0] / voltages[0]
            explanation = _format_steps(
                f"Charge is {micro_coulombs[0]} C and voltage is {voltages[0]} V.",
                "Use capacitance definition.",
                f"C = Q/V = {micro_coulombs[0]}/{voltages[0]} = {value}",
                value,
                "F",
            )
            return _answer(value, "F", explanation, ["capacitance_def"], domain)

    if domain == "energy":
        if "voltage" in text and micro_farads and energies:
            value = math.sqrt(2 * energies[0] / micro_farads[0])
            explanation = _format_steps(
                f"Energy is {energies[0]} J and capacitance is {micro_farads[0]} F.",
                "Rearrange capacitor energy.",
                f"V = sqrt(2E/C) = sqrt(2*{energies[0]}/{micro_farads[0]}) = {value}",
                value,
                "V",
            )
            return _answer(value, "V", explanation, ["capacitor_energy_voltage_from_energy"], domain)

        if "energy" in text and micro_farads and voltages:
            capacitance = micro_farads[0]
            if "series" in text and len(micro_farads) >= 2:
                capacitance = 1.0 / sum(1.0 / c for c in micro_farads)
            value = 0.5 * capacitance * (voltages[0] ** 2)
            explanation = _format_steps(
                f"Capacitance is {capacitance} F and voltage is {voltages[0]} V.",
                "Use capacitor energy.",
                f"E = 0.5*C*V^2 = 0.5*{capacitance}*{voltages[0]}^2 = {value}",
                value,
                "J",
            )
            formulas = ["series_capacitance", "energy_capacitor"] if "series" in text else ["energy_capacitor"]
            return _answer(value, "J", explanation, formulas, domain)

        if "energy" in text and powers and times:
            value = powers[0] * times[0]
            explanation = _format_steps(
                f"Power is {powers[0]} W and time is {times[0]} s.",
                "Use energy from constant power.",
                f"E = P*t = {powers[0]}*{times[0]} = {value}",
                value,
                "J",
            )
            return _answer(value, "J", explanation, ["power_basic"], domain)

        if "dissipated" in text and resistances and currents and times:
            value = (currents[0] ** 2) * resistances[0] * times[0]
            explanation = _format_steps(
                f"Current is {currents[0]} A, resistance is {resistances[0]} ohm, time is {times[0]} s.",
                "Use Joule heating.",
                f"E = I^2*R*t = {currents[0]}^2*{resistances[0]}*{times[0]} = {value}",
                value,
                "J",
            )
            return _answer(value, "J", explanation, ["joule_heating"], domain)

        if "energy" in text and micro_farads and micro_coulombs:
            value = (micro_coulombs[0] ** 2) / (2 * micro_farads[0])
            explanation = _format_steps(
                f"Charge is {micro_coulombs[0]} C and capacitance is {micro_farads[0]} F.",
                "Use capacitor energy from charge.",
                f"E = Q^2/(2C) = {micro_coulombs[0]}^2/(2*{micro_farads[0]}) = {value}",
                value,
                "J",
            )
            return _answer(value, "J", explanation, ["energy_capacitor"], domain)

    return None
