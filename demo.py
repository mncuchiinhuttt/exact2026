"""Demo runner for the EXACT 2026 physics pipeline."""

from exact_physics_pipeline import run_pipeline


PROBLEMS = [
    (
        "Problem 1",
        """A 12V battery is connected to two resistors in series: R1 = 4Ω and R2 = 8Ω.
What is the current flowing through the circuit and the voltage drop across R2?""",
    ),
    (
        "Problem 2",
        """A point charge of +2.0 μC is placed in a vacuum.
Calculate the electric field strength at a distance of 0.30 m from the charge.""",
    ),
    (
        "Problem 3",
        """A capacitor of 10 μF is charged to a potential difference of 50 V.
Calculate the energy stored in the capacitor.""",
    ),
]


def main() -> None:
    """Run the hardcoded demo problems and print compact results."""
    for name, problem in PROBLEMS:
        print(f"\n{name}")
        print("-" * len(name))
        try:
            result = run_pipeline(problem)
        except Exception as exc:
            if "No module named 'openai'" in str(exc):
                print("The OpenAI Python client is not installed.")
                print("Install dependencies with: pip install -r requirements.txt")
                return
            print("Could not reach the local vLLM server.")
            print("Start vLLM on port 8000 before running this demo.")
            print(f"Error: {exc}")
            return

        print(f"domain: {result['domain']}")
        print(f"answer: {result['answer']}")
        print(f"unit: {result['unit']}")
        print(f"confidence: {result['confidence']:.2f}")
        print(f"explanation: {result['explanation'][:500]}")


if __name__ == "__main__":
    main()
