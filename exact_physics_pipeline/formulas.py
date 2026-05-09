"""Static formula database and retrieval helpers."""

import re
from typing import Any


Formula = dict[str, Any]


FORMULA_DB: dict[str, Formula] = {
    "ohms_law": {
        "id": "ohms_law",
        "name": "Ohm's Law",
        "formula": "V = I * R",
        "variables": {"V": "voltage in volts", "I": "current in amperes", "R": "resistance in ohms"},
        "conditions": "Applies to ohmic resistors.",
        "domains": ["circuit"],
    },
    "kirchhoff_voltage": {
        "id": "kirchhoff_voltage",
        "name": "Kirchhoff Voltage Law",
        "formula": "sum(V) = 0 around any closed loop",
        "variables": {"V": "voltage rises or drops in volts"},
        "conditions": "Applies to closed loops in lumped circuits.",
        "domains": ["circuit"],
    },
    "kirchhoff_current": {
        "id": "kirchhoff_current",
        "name": "Kirchhoff Current Law",
        "formula": "sum(I_in) = sum(I_out) at any node",
        "variables": {"I_in": "incoming current", "I_out": "outgoing current"},
        "conditions": "Applies at circuit nodes under charge conservation.",
        "domains": ["circuit"],
    },
    "series_resistance": {
        "id": "series_resistance",
        "name": "Series Resistance",
        "formula": "R_total = R1 + R2 + ... + Rn",
        "variables": {"R_total": "equivalent resistance", "Rn": "individual resistor"},
        "conditions": "Applies when resistors carry the same current in series.",
        "domains": ["circuit"],
    },
    "parallel_resistance": {
        "id": "parallel_resistance",
        "name": "Parallel Resistance",
        "formula": "1/R_total = 1/R1 + 1/R2 + ... + 1/Rn",
        "variables": {"R_total": "equivalent resistance", "Rn": "individual resistor"},
        "conditions": "Applies when resistors share the same voltage in parallel.",
        "domains": ["circuit"],
    },
    "capacitance_def": {
        "id": "capacitance_def",
        "name": "Capacitance Definition",
        "formula": "C = Q / V",
        "variables": {"C": "capacitance in farads", "Q": "charge in coulombs", "V": "potential difference in volts"},
        "conditions": "Applies to capacitors and conductors with defined potential difference.",
        "domains": ["electrostatics"],
    },
    "series_capacitance": {
        "id": "series_capacitance",
        "name": "Series Capacitance",
        "formula": "1/C_total = 1/C1 + 1/C2",
        "variables": {"C_total": "equivalent capacitance", "C1": "first capacitor", "C2": "second capacitor"},
        "conditions": "Applies to capacitors connected in series.",
        "domains": ["electrostatics"],
    },
    "parallel_capacitance": {
        "id": "parallel_capacitance",
        "name": "Parallel Capacitance",
        "formula": "C_total = C1 + C2 + ... + Cn",
        "variables": {"C_total": "equivalent capacitance", "Cn": "individual capacitor"},
        "conditions": "Applies to capacitors connected in parallel.",
        "domains": ["electrostatics"],
    },
    "energy_capacitor": {
        "id": "energy_capacitor",
        "name": "Energy Stored in a Capacitor",
        "formula": "E = 0.5 * C * V^2; E = Q^2/(2C); E = 0.5 * Q * V",
        "variables": {"E": "energy in joules", "C": "capacitance in farads", "V": "voltage in volts", "Q": "charge in coulombs"},
        "conditions": "Applies to ideal capacitors.",
        "domains": ["energy", "electrostatics"],
    },
    "coulombs_law": {
        "id": "coulombs_law",
        "name": "Coulomb's Law",
        "formula": "F = k * q1 * q2 / r^2",
        "variables": {"F": "force in newtons", "k": "8.99e9 N·m²/C²", "q1": "charge 1", "q2": "charge 2", "r": "separation in meters"},
        "conditions": "Applies to point charges in vacuum or air approximation.",
        "domains": ["electrostatics"],
    },
    "electric_field_point": {
        "id": "electric_field_point",
        "name": "Electric Field of a Point Charge",
        "formula": "E = k * Q / r^2",
        "variables": {"E": "electric field in N/C or V/m", "k": "8.99e9 N·m²/C²", "Q": "source charge", "r": "distance in meters"},
        "conditions": "Applies to a point charge in vacuum or air approximation.",
        "domains": ["electrostatics"],
    },
    "electric_potential": {
        "id": "electric_potential",
        "name": "Electric Potential of a Point Charge",
        "formula": "V = k * Q / r",
        "variables": {"V": "electric potential in volts", "k": "8.99e9 N·m²/C²", "Q": "source charge", "r": "distance in meters"},
        "conditions": "Applies to a point charge relative to zero potential at infinity.",
        "domains": ["electrostatics"],
    },
    "current_definition": {
        "id": "current_definition",
        "name": "Current Definition",
        "formula": "I = ΔQ / Δt",
        "variables": {"I": "current in amperes", "ΔQ": "charge flow in coulombs", "Δt": "time interval in seconds"},
        "conditions": "Applies to steady average current through a cross-section.",
        "domains": ["circuit"],
    },
    "charge_from_current": {
        "id": "charge_from_current",
        "name": "Charge Transport",
        "formula": "Q = I * t",
        "variables": {"Q": "charge in coulombs", "I": "current in amperes", "t": "time in seconds"},
        "conditions": "Applies when current is constant over the time interval.",
        "domains": ["circuit", "electrostatics"],
    },
    "power_basic": {
        "id": "power_basic",
        "name": "Power and Energy",
        "formula": "P = E / t; E = P * t",
        "variables": {"P": "power in watts", "E": "energy in joules", "t": "time in seconds"},
        "conditions": "Applies when power is constant or an average power is given.",
        "domains": ["energy"],
    },
    "electric_power": {
        "id": "electric_power",
        "name": "Electric Power",
        "formula": "P = I * V",
        "variables": {"P": "power in watts", "I": "current in amperes", "V": "potential difference in volts"},
        "conditions": "Applies to electrical devices using passive sign convention.",
        "domains": ["circuit", "energy"],
    },
    "resistor_power_current": {
        "id": "resistor_power_current",
        "name": "Resistor Power from Current",
        "formula": "P = I^2 * R",
        "variables": {"P": "power in watts", "I": "current in amperes", "R": "resistance in ohms"},
        "conditions": "Applies to ohmic resistors.",
        "domains": ["circuit", "energy"],
    },
    "resistor_power_voltage": {
        "id": "resistor_power_voltage",
        "name": "Resistor Power from Voltage",
        "formula": "P = V^2 / R",
        "variables": {"P": "power in watts", "V": "voltage in volts", "R": "resistance in ohms"},
        "conditions": "Applies to ohmic resistors.",
        "domains": ["circuit", "energy"],
    },
    "joule_heating": {
        "id": "joule_heating",
        "name": "Joule Heating Energy",
        "formula": "E = I^2 * R * t = V * I * t = V^2 * t / R",
        "variables": {"E": "thermal energy in joules", "I": "current in amperes", "R": "resistance in ohms", "V": "voltage in volts", "t": "time in seconds"},
        "conditions": "Applies to energy dissipated by an ohmic resistor.",
        "domains": ["circuit", "energy"],
    },
    "work_charge_potential": {
        "id": "work_charge_potential",
        "name": "Electrical Work on Charge",
        "formula": "W = q * ΔV",
        "variables": {"W": "work or energy in joules", "q": "charge in coulombs", "ΔV": "potential difference in volts"},
        "conditions": "Applies to moving charge through an electric potential difference.",
        "domains": ["electrostatics", "energy"],
    },
    "series_voltage_divider": {
        "id": "series_voltage_divider",
        "name": "Voltage Divider",
        "formula": "V_i = V_total * R_i / R_total",
        "variables": {"V_i": "voltage drop across resistor i", "V_total": "source voltage", "R_i": "selected series resistor", "R_total": "total series resistance"},
        "conditions": "Applies to resistors in series carrying the same current.",
        "domains": ["circuit"],
    },
    "parallel_current_divider_two": {
        "id": "parallel_current_divider_two",
        "name": "Two-Branch Current Divider",
        "formula": "I_1 = I_total * R_2 / (R_1 + R_2); I_2 = I_total * R_1 / (R_1 + R_2)",
        "variables": {"I_1": "current through branch 1", "I_2": "current through branch 2", "I_total": "total current", "R_1": "branch 1 resistance", "R_2": "branch 2 resistance"},
        "conditions": "Applies to two resistors in parallel.",
        "domains": ["circuit"],
    },
    "conductance_parallel": {
        "id": "conductance_parallel",
        "name": "Conductance Form of Parallel Resistance",
        "formula": "G_total = G1 + G2 + ... + Gn; G = 1/R",
        "variables": {"G_total": "total conductance in siemens", "G": "conductance", "R": "resistance"},
        "conditions": "Useful for parallel resistor networks.",
        "domains": ["circuit"],
    },
    "two_resistor_parallel": {
        "id": "two_resistor_parallel",
        "name": "Two-Resistor Parallel Equivalent",
        "formula": "R_total = (R1 * R2) / (R1 + R2)",
        "variables": {"R_total": "equivalent resistance", "R1": "first resistor", "R2": "second resistor"},
        "conditions": "Shortcut for exactly two resistors in parallel.",
        "domains": ["circuit"],
    },
    "capacitor_charge": {
        "id": "capacitor_charge",
        "name": "Capacitor Charge",
        "formula": "Q = C * V",
        "variables": {"Q": "charge in coulombs", "C": "capacitance in farads", "V": "potential difference in volts"},
        "conditions": "Applies to ideal capacitors.",
        "domains": ["electrostatics", "energy"],
    },
    "capacitor_voltage": {
        "id": "capacitor_voltage",
        "name": "Capacitor Voltage",
        "formula": "V = Q / C",
        "variables": {"V": "potential difference in volts", "Q": "charge in coulombs", "C": "capacitance in farads"},
        "conditions": "Applies to ideal capacitors.",
        "domains": ["electrostatics", "energy"],
    },
    "capacitor_energy_voltage_from_energy": {
        "id": "capacitor_energy_voltage_from_energy",
        "name": "Capacitor Voltage from Stored Energy",
        "formula": "V = sqrt(2 * E / C)",
        "variables": {"V": "voltage in volts", "E": "stored energy in joules", "C": "capacitance in farads"},
        "conditions": "Applies to ideal capacitors when stored energy and capacitance are known.",
        "domains": ["energy", "electrostatics"],
    },
    "capacitor_energy_capacitance_from_energy": {
        "id": "capacitor_energy_capacitance_from_energy",
        "name": "Capacitance from Energy and Voltage",
        "formula": "C = 2 * E / V^2",
        "variables": {"C": "capacitance in farads", "E": "stored energy in joules", "V": "voltage in volts"},
        "conditions": "Applies to ideal capacitors when stored energy and voltage are known.",
        "domains": ["energy", "electrostatics"],
    },
    "capacitor_series_charge_same": {
        "id": "capacitor_series_charge_same",
        "name": "Charge on Series Capacitors",
        "formula": "Q1 = Q2 = ... = Qn = C_total * V_total",
        "variables": {"Qn": "charge on each series capacitor", "C_total": "series equivalent capacitance", "V_total": "total applied voltage"},
        "conditions": "Applies to ideal capacitors in series.",
        "domains": ["electrostatics", "energy"],
    },
    "capacitor_parallel_voltage_same": {
        "id": "capacitor_parallel_voltage_same",
        "name": "Voltage on Parallel Capacitors",
        "formula": "V1 = V2 = ... = Vn = V_total",
        "variables": {"Vn": "voltage across each parallel capacitor", "V_total": "applied voltage"},
        "conditions": "Applies to ideal capacitors in parallel.",
        "domains": ["electrostatics", "energy"],
    },
    "parallel_plate_capacitance": {
        "id": "parallel_plate_capacitance",
        "name": "Parallel-Plate Capacitance",
        "formula": "C = ε0 * A / d",
        "variables": {"C": "capacitance in farads", "ε0": "8.854e-12 F/m", "A": "plate area in square meters", "d": "plate separation in meters"},
        "conditions": "Applies to ideal parallel plates in vacuum ignoring fringe fields.",
        "domains": ["electrostatics"],
    },
    "parallel_plate_dielectric": {
        "id": "parallel_plate_dielectric",
        "name": "Parallel-Plate Capacitance with Dielectric",
        "formula": "C = κ * ε0 * A / d",
        "variables": {"C": "capacitance in farads", "κ": "relative permittivity", "ε0": "8.854e-12 F/m", "A": "plate area", "d": "plate separation"},
        "conditions": "Applies when a uniform dielectric fully fills an ideal parallel-plate capacitor.",
        "domains": ["electrostatics"],
    },
    "electric_field_uniform": {
        "id": "electric_field_uniform",
        "name": "Uniform Electric Field",
        "formula": "E = V / d",
        "variables": {"E": "electric field in V/m or N/C", "V": "potential difference in volts", "d": "separation in meters"},
        "conditions": "Applies to approximately uniform fields such as ideal parallel plates.",
        "domains": ["electrostatics"],
    },
    "electric_force_on_charge": {
        "id": "electric_force_on_charge",
        "name": "Force on a Charge in an Electric Field",
        "formula": "F = q * E",
        "variables": {"F": "force in newtons", "q": "charge in coulombs", "E": "electric field in N/C"},
        "conditions": "Applies to a charge in an external electric field.",
        "domains": ["electrostatics"],
    },
    "electric_field_superposition": {
        "id": "electric_field_superposition",
        "name": "Electric Field Superposition",
        "formula": "E_net = Σ E_i",
        "variables": {"E_net": "vector sum of electric fields", "E_i": "field contribution from source i"},
        "conditions": "Use vector addition; in one-dimensional symmetric cases include signs.",
        "domains": ["electrostatics"],
    },
    "electric_potential_superposition": {
        "id": "electric_potential_superposition",
        "name": "Electric Potential Superposition",
        "formula": "V_net = Σ k * q_i / r_i",
        "variables": {"V_net": "net electric potential", "q_i": "source charge i", "r_i": "distance from charge i"},
        "conditions": "Potential is scalar; include charge signs.",
        "domains": ["electrostatics"],
    },
    "electric_potential_energy_two_charges": {
        "id": "electric_potential_energy_two_charges",
        "name": "Potential Energy of Two Point Charges",
        "formula": "U = k * q1 * q2 / r",
        "variables": {"U": "electric potential energy in joules", "k": "8.99e9 N·m²/C²", "q1": "first charge", "q2": "second charge", "r": "separation"},
        "conditions": "Applies to two point charges relative to infinite separation.",
        "domains": ["electrostatics", "energy"],
    },
    "gauss_law": {
        "id": "gauss_law",
        "name": "Gauss's Law",
        "formula": "Φ_E = ∮ E · dA = Q_enclosed / ε0",
        "variables": {"Φ_E": "electric flux", "E": "electric field", "dA": "area element", "Q_enclosed": "enclosed charge", "ε0": "8.854e-12 F/m"},
        "conditions": "Most useful with high symmetry: spherical, cylindrical, or planar.",
        "domains": ["electrostatics"],
    },
    "electric_flux_uniform": {
        "id": "electric_flux_uniform",
        "name": "Uniform Electric Flux",
        "formula": "Φ_E = E * A * cos(θ)",
        "variables": {"Φ_E": "electric flux", "E": "uniform electric field", "A": "surface area", "θ": "angle between field and area normal"},
        "conditions": "Applies to a flat surface in a uniform electric field.",
        "domains": ["electrostatics"],
    },
    "field_infinite_line_charge": {
        "id": "field_infinite_line_charge",
        "name": "Electric Field of an Infinite Line Charge",
        "formula": "E = λ / (2 * π * ε0 * r)",
        "variables": {"E": "electric field", "λ": "linear charge density", "ε0": "8.854e-12 F/m", "r": "radial distance"},
        "conditions": "Applies to an ideal infinite straight line charge.",
        "domains": ["electrostatics"],
    },
    "field_infinite_sheet_charge": {
        "id": "field_infinite_sheet_charge",
        "name": "Electric Field of an Infinite Sheet",
        "formula": "E = σ / (2 * ε0)",
        "variables": {"E": "electric field", "σ": "surface charge density", "ε0": "8.854e-12 F/m"},
        "conditions": "Applies to an ideal infinite nonconducting sheet of charge.",
        "domains": ["electrostatics"],
    },
    "field_conducting_surface": {
        "id": "field_conducting_surface",
        "name": "Electric Field Near a Conductor Surface",
        "formula": "E = σ / ε0",
        "variables": {"E": "electric field just outside conductor", "σ": "surface charge density", "ε0": "8.854e-12 F/m"},
        "conditions": "Applies just outside a conductor in electrostatic equilibrium.",
        "domains": ["electrostatics"],
    },
    "resistivity_wire": {
        "id": "resistivity_wire",
        "name": "Resistance of a Uniform Wire",
        "formula": "R = ρ * L / A",
        "variables": {"R": "resistance in ohms", "ρ": "resistivity in ohm meters", "L": "length in meters", "A": "cross-sectional area"},
        "conditions": "Applies to a uniform material and cross-section at fixed temperature.",
        "domains": ["circuit"],
    },
    "conductivity_relation": {
        "id": "conductivity_relation",
        "name": "Conductivity and Resistivity",
        "formula": "σ = 1 / ρ",
        "variables": {"σ": "conductivity in S/m", "ρ": "resistivity in ohm meters"},
        "conditions": "Applies to linear isotropic materials.",
        "domains": ["circuit"],
    },
    "rc_time_constant": {
        "id": "rc_time_constant",
        "name": "RC Time Constant",
        "formula": "τ = R * C",
        "variables": {"τ": "time constant in seconds", "R": "resistance in ohms", "C": "capacitance in farads"},
        "conditions": "Applies to first-order RC charging or discharging circuits.",
        "domains": ["circuit", "energy", "electrostatics"],
    },
    "rc_charging_voltage": {
        "id": "rc_charging_voltage",
        "name": "RC Charging Voltage",
        "formula": "V_C(t) = V0 * (1 - exp(-t/(R*C)))",
        "variables": {"V_C": "capacitor voltage at time t", "V0": "source voltage", "t": "time", "R": "resistance", "C": "capacitance"},
        "conditions": "Applies to an initially uncharged capacitor charging through a resistor.",
        "domains": ["circuit", "electrostatics"],
    },
    "rc_discharging_voltage": {
        "id": "rc_discharging_voltage",
        "name": "RC Discharging Voltage",
        "formula": "V_C(t) = V_initial * exp(-t/(R*C))",
        "variables": {"V_C": "capacitor voltage at time t", "V_initial": "initial capacitor voltage", "t": "time", "R": "resistance", "C": "capacitance"},
        "conditions": "Applies to a capacitor discharging through a resistor.",
        "domains": ["circuit", "electrostatics", "energy"],
    },
    "efficiency": {
        "id": "efficiency",
        "name": "Efficiency",
        "formula": "η = E_useful / E_input = P_output / P_input",
        "variables": {"η": "efficiency as a fraction", "E_useful": "useful output energy", "E_input": "input energy", "P_output": "output power", "P_input": "input power"},
        "conditions": "Applies to energy conversion problems.",
        "domains": ["energy"],
    },
}


DEFAULT_FAST_FORMULA_LIMIT = 10


def _formula_score(formula: Formula, problem: str) -> int:
    """Score a formula against problem text using lightweight keyword overlap."""
    text = problem.lower()
    haystack = " ".join(
        [
            formula["id"].replace("_", " "),
            formula["name"],
            formula["formula"],
            formula["conditions"],
            " ".join(formula["variables"].values()),
        ]
    ).lower()
    score = 0
    for token in set(re.findall(r"[a-zA-Z]+", haystack)):
        if len(token) >= 4 and token in text:
            score += 1
    for strong_token in ["series", "parallel", "capacitor", "charge", "energy", "power", "field", "potential", "current", "voltage"]:
        if strong_token in text and strong_token in haystack:
            score += 3
    return score


def retrieve_formulas(domain: str, problem: str | None = None, max_formulas: int | None = None) -> list[Formula]:
    """Return formulas relevant to a classified domain, optionally ranked for a problem."""
    if domain == "general":
        formulas = list(FORMULA_DB.values())
    else:
        formulas = [formula for formula in FORMULA_DB.values() if domain in formula["domains"]]

    if problem and max_formulas:
        formulas = sorted(formulas, key=lambda formula: _formula_score(formula, problem), reverse=True)
        return formulas[:max_formulas]
    return formulas


def format_formula_table(formulas: list[Formula]) -> str:
    """Format formulas as a compact reference table for the system prompt."""
    rows = ["| ID | Name | Formula | Conditions |", "|---|---|---|---|"]
    for formula in formulas:
        rows.append(
            f"| {formula['id']} | {formula['name']} | {formula['formula']} | {formula['conditions']} |"
        )
    return "\n".join(rows)
