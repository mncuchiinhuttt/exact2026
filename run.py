"""CLI entrypoint for EXACT 2026 Subtask 1 and Subtask 2 inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from exact2026.pipeline.router import route_and_run


def _cot_to_list(cot: Any) -> list[str] | str:
    """Convert a multiline COT block into a list of step strings when possible."""
    if not isinstance(cot, str):
        return cot
    lines = [line.strip() for line in cot.splitlines() if line.strip()]
    if lines:
        return lines
    parts = [part.strip() for part in cot.split("- Step ") if part.strip()]
    return [f"- Step {part}" for part in parts] if parts else []


def _contest_shape(result: dict[str, Any], debug: bool = False) -> dict[str, Any]:
    """Return the required contest output schema."""
    answer = result.get("answer")
    if "unit" in result and result.get("unit") and answer is not None:
        answer_out: str | float | None = f"{answer} {result['unit']}"
    else:
        answer_out = answer
    shaped = {
        "answer": answer_out,
        "explanation": result.get("explanation", ""),
        "cot": _cot_to_list(result.get("cot", "")),
        "fol": result.get("fol", ""),
        "premises": result.get("premises"),
        "confidence": result.get("confidence", 0.0),
    }
    if debug:
        for key in ["domain", "formulas_used", "source", "all_answers", "raw_think"]:
            if key in result:
                shaped[key] = result[key]
    return shaped


def run_structured(input_data: dict[str, Any], config: dict | None = None, debug: bool = False) -> dict[str, Any] | list[dict[str, Any]]:
    """Run routed inference and return the contest/API response shape."""
    result = route_and_run(input_data, config)
    if isinstance(result, list):
        return [_contest_shape(item, debug=debug) for item in result]
    if "error" in result:
        return result
    return _contest_shape(result, debug=debug)


def _parse_input(raw_input: str) -> dict[str, Any]:
    """Parse CLI input as JSON, falling back to a physics question string."""
    try:
        parsed = json.loads(raw_input)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"question": raw_input}


def _config(args: argparse.Namespace) -> dict[str, Any]:
    """Build pipeline config from CLI flags."""
    return {
        "fast": args.fast,
        "k": 1 if args.fast else args.k,
        "mode": "compact" if args.fast else args.mode,
        "max_tokens": args.max_tokens,
        "max_formulas": args.max_formulas,
        "no_thinking": args.no_thinking,
        "direct_first": True,
    }


def main() -> None:
    """Run one input or a JSONL batch and print JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON object or raw physics question.")
    parser.add_argument("--input_file", help="JSONL input file for batch mode.")
    parser.add_argument("--output_file", help="JSONL output file for batch mode.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["full", "compact"], default="full")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-formulas", type=int, default=8)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    cfg = _config(args)
    if args.input_file:
        if not args.output_file:
            raise SystemExit("--output_file is required with --input_file")
        with Path(args.input_file).open(encoding="utf-8") as source, Path(args.output_file).open("w", encoding="utf-8") as sink:
            for line in source:
                if not line.strip():
                    continue
                output = run_structured(json.loads(line), cfg, debug=args.debug)
                sink.write(json.dumps(output, ensure_ascii=False) + "\n")
        return

    raw = args.input if args.input is not None else sys.stdin.read().strip()
    if not raw:
        raise SystemExit("Provide --input, --input_file, or stdin.")
    output = run_structured(_parse_input(raw), cfg, debug=args.debug)
    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()

