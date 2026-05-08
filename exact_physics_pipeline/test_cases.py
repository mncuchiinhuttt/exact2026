"""Curated evaluation cases for the EXACT 2026 physics pipeline."""

from typing import Any


TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "circuit_simple_series_current",
        "difficulty": "simple",
        "domain": "circuit",
        "problem": "A 12 V battery is connected to two resistors in series, 4 ohm and 8 ohm. What is the current through the circuit?",
        "expected_answer": 1.0,
        "expected_unit": "A",
    },
    {
        "id": "circuit_simple_ohms_resistance",
        "difficulty": "simple",
        "domain": "circuit",
        "problem": "A resistor has 24 V across it and carries a current of 3 A. What is its resistance?",
        "expected_answer": 8.0,
        "expected_unit": "Ω",
        "accepted_units": ["Ω", "ohm", "ohms"],
    },
    {
        "id": "circuit_simple_voltage_drop",
        "difficulty": "simple",
        "domain": "circuit",
        "problem": "A 5 ohm resistor carries a current of 2.4 A. What is the voltage drop across the resistor?",
        "expected_answer": 12.0,
        "expected_unit": "V",
    },
    {
        "id": "circuit_parallel_current",
        "difficulty": "moderate",
        "domain": "circuit",
        "problem": "Two resistors, 6 ohm and 3 ohm, are connected in parallel across a 12 V battery. What is the total current supplied by the battery?",
        "expected_answer": 6.0,
        "expected_unit": "A",
    },
    {
        "id": "circuit_three_series_current",
        "difficulty": "moderate",
        "domain": "circuit",
        "problem": "A 24 V source is connected to three series resistors of 6 ohm, 3 ohm, and 1 ohm. What current flows through the circuit?",
        "expected_answer": 2.4,
        "expected_unit": "A",
    },
    {
        "id": "circuit_series_parallel_current",
        "difficulty": "complex",
        "domain": "circuit",
        "problem": "A 10.8 V battery is connected to a 3 ohm resistor in series with a parallel pair of 4 ohm and 6 ohm resistors. What is the total current from the battery?",
        "expected_answer": 2.0,
        "expected_unit": "A",
    },
    {
        "id": "circuit_kcl_unknown_current",
        "difficulty": "complex",
        "domain": "circuit",
        "problem": "At a circuit node, currents of 2.0 A and 3.5 A enter. One outgoing branch carries 1.2 A. What current must leave through the second outgoing branch?",
        "expected_answer": 4.3,
        "expected_unit": "A",
    },
    {
        "id": "electrostatics_point_field",
        "difficulty": "simple",
        "domain": "electrostatics",
        "problem": "A point charge of +2.0 microcoulomb is in vacuum. What is the electric field strength 0.30 m from the charge?",
        "expected_answer": 1.9977777777777778e5,
        "expected_unit": "N/C",
        "accepted_units": ["N/C", "V/m"],
    },
    {
        "id": "electrostatics_capacitance_definition",
        "difficulty": "simple",
        "domain": "electrostatics",
        "problem": "A capacitor stores 3.0 microcoulomb of charge when connected to 12 V. What is its capacitance?",
        "expected_answer": 2.5e-7,
        "expected_unit": "F",
    },
    {
        "id": "electrostatics_charge_from_capacitance",
        "difficulty": "simple",
        "domain": "electrostatics",
        "problem": "A 47 microfarad capacitor is charged to 12 V. What charge is stored on the capacitor?",
        "expected_answer": 5.64e-4,
        "expected_unit": "C",
    },
    {
        "id": "electrostatics_parallel_capacitance",
        "difficulty": "moderate",
        "domain": "electrostatics",
        "problem": "Capacitors of 2 microfarad, 3 microfarad, and 5 microfarad are connected in parallel. What is the equivalent capacitance?",
        "expected_answer": 1.0e-5,
        "expected_unit": "F",
    },
    {
        "id": "electrostatics_series_capacitance",
        "difficulty": "moderate",
        "domain": "electrostatics",
        "problem": "A 6 microfarad capacitor and a 3 microfarad capacitor are connected in series. What is the equivalent capacitance?",
        "expected_answer": 2.0e-6,
        "expected_unit": "F",
    },
    {
        "id": "electrostatics_coulomb_force",
        "difficulty": "moderate",
        "domain": "electrostatics",
        "problem": "Two point charges of +2.0 microcoulomb and +3.0 microcoulomb are separated by 0.50 m in vacuum. What is the magnitude of the electrostatic force between them?",
        "expected_answer": 0.21576,
        "expected_unit": "N",
    },
    {
        "id": "electrostatics_point_potential",
        "difficulty": "moderate",
        "domain": "electrostatics",
        "problem": "What is the electric potential 0.20 m from a +5.0 microcoulomb point charge in vacuum?",
        "expected_answer": 224750.0,
        "expected_unit": "V",
    },
    {
        "id": "electrostatics_two_charge_potential",
        "difficulty": "complex",
        "domain": "electrostatics",
        "problem": "At a point, a +3.0 microcoulomb charge and a -1.0 microcoulomb charge are each 0.50 m away. What is the net electric potential at that point?",
        "expected_answer": 35960.0,
        "expected_unit": "V",
    },
    {
        "id": "electrostatics_three_series_capacitance",
        "difficulty": "complex",
        "domain": "electrostatics",
        "problem": "Three identical 2.0 microfarad capacitors are connected in series. What is their equivalent capacitance?",
        "expected_answer": 6.666666666666667e-7,
        "expected_unit": "F",
    },
    {
        "id": "energy_resistor_work",
        "difficulty": "simple",
        "domain": "energy",
        "problem": "A 60 W lamp runs for 120 s. How much energy does it use?",
        "expected_answer": 7200.0,
        "expected_unit": "J",
    },
    {
        "id": "energy_capacitor_basic",
        "difficulty": "simple",
        "domain": "energy",
        "problem": "A 10 microfarad capacitor is charged to 50 V. What energy is stored in the capacitor?",
        "expected_answer": 0.0125,
        "expected_unit": "J",
    },
    {
        "id": "energy_resistor_dissipated",
        "difficulty": "moderate",
        "domain": "energy",
        "problem": "A 5 ohm resistor carries a steady current of 2 A for 10 s. How much energy is dissipated in the resistor?",
        "expected_answer": 200.0,
        "expected_unit": "J",
    },
    {
        "id": "energy_capacitor_from_charge",
        "difficulty": "moderate",
        "domain": "energy",
        "problem": "A capacitor has capacitance 8 microfarad and stores charge 40 microcoulomb. What energy is stored in it?",
        "expected_answer": 1.0e-4,
        "expected_unit": "J",
    },
    {
        "id": "energy_find_capacitor_voltage",
        "difficulty": "complex",
        "domain": "energy",
        "problem": "A 20 microfarad capacitor stores 0.040 J of energy. What voltage is across the capacitor?",
        "expected_answer": 63.245553203367585,
        "expected_unit": "V",
    },
    {
        "id": "energy_series_capacitor_energy",
        "difficulty": "complex",
        "domain": "energy",
        "problem": "A 4 microfarad capacitor and a 12 microfarad capacitor are connected in series across 100 V. What total energy is stored in the equivalent capacitance?",
        "expected_answer": 0.015,
        "expected_unit": "J",
    },
]


def get_test_cases(domain: str | None = None, difficulty: str | None = None) -> list[dict[str, Any]]:
    """Return test cases optionally filtered by domain and difficulty."""
    cases = TEST_CASES
    if domain:
        cases = [case for case in cases if case["domain"] == domain]
    if difficulty:
        cases = [case for case in cases if case["difficulty"] == difficulty]
    return cases
