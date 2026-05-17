"""Prepare EXACT 2026 SFT JSONL datasets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from zipfile import ZipFile
from pathlib import Path
from typing import Any

from exact2026.pipeline.answer_parser import parse_answer
from exact2026.pipeline.unit_normalizer import expand_prefix, normalize_unit
from exact2026.train.data_config import BTC_LOGIC, BTC_PHYSICS, EXTERNAL_LOGIC, EXTERNAL_PHYSICS, DataConfig


LOGGER = logging.getLogger(__name__)
BAD_UNITS = {"", "-", "—", "–", "N/A", "none", "null"}
LOGIC_ANSWERS = {"Yes", "No", "True", "False", "A", "B", "C", "D"}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_btc_physics(path: str | Path) -> list[dict[str, Any]]:
    """Read BTC physics CSV rows as raw dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_answer(answer: Any) -> float:
    """Parse a BTC numeric answer."""
    return float(str(answer).strip().replace(",", "."))


def filter_physics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep parseable physics rows and save excluded rows for inspection."""
    kept = []
    excluded = []
    for record in records:
        reason = None
        try:
            _float_answer(record.get("answer"))
        except (TypeError, ValueError):
            reason = "answer_not_float"
        unit = str(record.get("unit", "")).strip()
        cot = record.get("cot")
        if reason is None and unit in BAD_UNITS:
            reason = "missing_unit"
        if reason is None and (not isinstance(cot, str) or len(cot.strip()) <= 20):
            reason = "missing_cot"
        if reason:
            excluded.append({**record, "excluded": reason})
        else:
            kept.append(record)
    _write_jsonl(Path("excluded_physics.jsonl"), excluded)
    return kept


def normalize_physics_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one BTC physics row to the internal training shape."""
    answer_num, unit_canonical = expand_prefix(_float_answer(record["answer"]), record.get("unit", ""))
    if unit_canonical == normalize_unit(record.get("unit", "")):
        answer_num = _float_answer(record["answer"])
    return {
        "id": str(record.get("id", "")).strip(),
        "question": str(record.get("question", "")).strip(),
        "cot": str(record.get("cot", "")).strip(),
        "answer_num": float(answer_num),
        "unit": unit_canonical,
        "answer_str": f"{answer_num:g} {unit_canonical}",
    }


def _sentences(text: str) -> list[str]:
    """Split text into rough natural-language sentences."""
    return [part.strip(" -\n\t") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def summarize_cot_to_explanation(cot: str, answer_str: str) -> str:
    """Create a short student-facing explanation from full COT without LLM calls."""
    step_matches = re.findall(r"Step\s+\d+\s*:\s*(.*?)(?=\n?\s*Step\s+\d+\s*:|\Z)", cot, flags=re.IGNORECASE | re.DOTALL)
    parts = [re.sub(r"\s+", " ", match).strip() for match in step_matches] or _sentences(cot)
    first = parts[0] if parts else "identify the relevant given quantities"
    return f"To solve this, {first.rstrip('.')}. After computing, the result is {answer_str}."


def _cot_bullets(cot: str) -> str:
    """Format COT text as bullet steps."""
    step_matches = re.findall(r"Step\s+(\d+)\s*:\s*(.*?)(?=\n?\s*Step\s+\d+\s*:|\Z)", cot, flags=re.IGNORECASE | re.DOTALL)
    if step_matches:
        return "\n".join(f"- Step {number}: {re.sub(r'\\s+', ' ', text).strip()}" for number, text in step_matches)
    return "\n".join(f"- Step {index + 1}: {sentence}" for index, sentence in enumerate(_sentences(cot)))


def build_physics_sft_sample(record: dict[str, Any]) -> dict[str, Any]:
    """Build one TRL ChatML-format physics SFT sample."""
    answer_str = record["answer_str"]
    system = """You are an expert physics tutor. Solve the problem step by step.
End your response with EXACTLY this structure:

COT:
- Step 1: [extract given quantities with values and SI units]
- Step 2: [convert all values to SI base units]
- Step 3: [identify and state the formula]
- Step 4: [substitute values and compute]
- Step 5: [state the final answer]

EXPLANATION:
[1-2 sentences summarizing the solution in plain language for a student, no Step markers, no raw formulas]

ANSWER: <numeric_value> <SI_unit>"""
    assistant = f"""COT:
{_cot_bullets(record["cot"])}

EXPLANATION:
{summarize_cot_to_explanation(record["cot"], answer_str)}

ANSWER: {answer_str}"""
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": record["question"]}, {"role": "assistant", "content": assistant}]}


def load_btc_logic(path: str | Path) -> list[dict[str, Any]]:
    """Load BTC logic records from JSONL or JSON array and flatten questions."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    flattened = []
    for record in rows:
        questions = record.get("questions", [])
        answers = record.get("answers", [])
        explanations = record.get("explanations", record.get("explanation", []))
        idx_values = record.get("idx", [])
        for index, question_item in enumerate(questions):
            if isinstance(question_item, dict):
                question_text = question_item.get("question", "")
                answer = question_item.get("answer", "")
                explanation = question_item.get("explanation", "")
                support_idx = question_item.get("idx", idx_values[index] if index < len(idx_values) else [])
            else:
                question_text = str(question_item)
                answer = answers[index] if index < len(answers) else ""
                explanation = explanations[index] if isinstance(explanations, list) and index < len(explanations) else ""
                support_idx = idx_values[index] if index < len(idx_values) else []
            flattened.append(
                {
                    "premises-NL": record.get("premises-NL", []),
                    "premises-FOL": record.get("premises-FOL", []),
                    "question": question_text,
                    "answer": answer,
                    "explanation": explanation,
                    "idx": support_idx,
                }
            )
    return flattened


def filter_logic(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep logic rows with supported answers, explanation, and support indices."""
    return [
        record
        for record in records
        if record.get("answer") in LOGIC_ANSWERS
        and str(record.get("explanation", "")).strip()
        and isinstance(record.get("idx"), list)
        and len(record.get("idx")) > 0
    ]


def build_logic_sft_sample(record: dict[str, Any]) -> dict[str, Any]:
    """Build one TRL ChatML-format logic SFT sample."""
    premises = "\n".join(f"{index + 1}. {premise}" for index, premise in enumerate(record.get("premises-NL", [])))
    explanation = str(record.get("explanation", "")).strip()
    steps = "\n".join(f"- Step {index + 1}: {sentence}" for index, sentence in enumerate(_sentences(explanation)))
    system = """You are an educational regulations reasoning assistant.
Given premises and a question, provide a clear answer and explanation.
Format your response as:

COT:
- Step 0 (Tool call - Z3): [placeholder — symbolic verification not yet implemented]
- Step 1: [first reasoning step]
- ...

EXPLANATION:
[2-4 sentences explaining the answer in plain language, no Step markers]

ANSWER: <answer>"""
    user = f"Premises:\n{premises}\n\nQuestion: {record.get('question', '')}"
    assistant = f"""COT:
- Step 0 (Tool call - Z3): Encode premises into FOL and verify entailment. [placeholder]
{steps}

EXPLANATION:
{explanation}

ANSWER: {record.get("answer", "")}"""
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def _clean_latex_text(text: Any) -> str:
    """Remove common LaTeX wrappers while preserving readable content."""
    cleaned = str(text or "").strip()
    cleaned = cleaned.strip("$").strip()
    cleaned = re.sub(r"\\(?:mathrm|mathsf|text)\{([^{}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("\\,", "").replace("~", "").replace("−", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_numeric_answer(answer: Any, unit: Any = None) -> tuple[float, str, str] | None:
    """Parse an answer field that is already expected to be a plain final number."""
    answer_text = _clean_latex_text(answer)
    if not answer_text:
        return None
    if "=" in answer_text:
        answer_text = answer_text.rsplit("=", maxsplit=1)[-1].strip()
    if re.search(r"\\(?:frac|sqrt)|\b(?:frac|sqrt|sin|cos|tan|lambda|theta|alpha|beta|gamma|mu|pi)\b", answer_text):
        return None
    match = re.fullmatch(r"([-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zΩμ°%/·^\-{}]*)", answer_text)
    if not match:
        return None
    try:
        value = _float_answer(match.group(1))
    except ValueError:
        return None
    raw_unit = _clean_latex_text(unit) or match.group(2) or "1"
    return value, raw_unit, f"{value:g} {raw_unit}".strip()


def _load_scibench(config: DataConfig) -> list[dict[str, Any]]:
    """Map SciBench rows, whose schema differs from the generic external loader."""
    from datasets import load_dataset

    dataset = load_dataset(config.hf_id, split="train")
    samples = []
    skipped = 0
    for index, row in enumerate(dataset):
        question = row.get("problem_text") or row.get("problem") or row.get("question")
        solution = str(row.get("solution", "")).strip()
        parsed = _parse_numeric_answer(row.get("answer_number"), row.get("unit"))
        if not question or not solution or parsed is None:
            skipped += 1
            continue
        value, unit, answer_str = parsed
        normalized = {
            "id": str(row.get("problemid") or f"{config.name}-{index}").strip(),
            "question": str(question).strip(),
            "cot": solution,
            "answer_num": value,
            "unit": normalize_unit(unit),
            "answer_str": answer_str,
        }
        samples.append(build_physics_sft_sample(normalized))
    LOGGER.info("Mapped %s: kept %s rows, skipped %s rows without solution/numeric answer", config.name, len(samples), skipped)
    return samples


def _ordered_subquestions(question_structure: dict[str, Any]) -> list[tuple[str, str]]:
    """Return PhysReason sub-questions in numeric order."""
    items = []
    for key, value in question_structure.items():
        match = re.fullmatch(r"sub_question_(\d+)", key)
        if match:
            items.append((int(match.group(1)), str(value).strip()))
    return [(f"sub_question_{number}", question) for number, question in sorted(items)]


def _flatten_steps(steps: Any) -> str:
    """Join PhysReason explanation step dictionaries into one explanation string."""
    if isinstance(steps, dict):
        ordered = sorted(steps.items(), key=lambda item: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", item[0])])
        return "\n".join(str(value).strip() for _, value in ordered if str(value).strip())
    return str(steps or "").strip()


def _load_physreason(config: DataConfig) -> list[dict[str, Any]]:
    """Load PhysReason directly from its zip archive to avoid HF schema casting issues."""
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(config.hf_id, "PhysReason-full.zip", repo_type="dataset")
    samples = []
    skipped = 0
    with ZipFile(zip_path) as archive:
        problem_files = sorted(name for name in archive.namelist() if name.endswith("problem.json"))
        for problem_file in problem_files:
            record = json.loads(archive.read(problem_file))
            question_structure = record.get("question_structure") or {}
            context = str(question_structure.get("context", "")).strip()
            answers = record.get("answer") or []
            if not isinstance(answers, list):
                answers = [answers]
            explanation_steps = record.get("explanation_steps") or {}
            for answer_index, (sub_key, sub_question) in enumerate(_ordered_subquestions(question_structure)):
                parsed = _parse_numeric_answer(answers[answer_index] if answer_index < len(answers) else None)
                cot = _flatten_steps(explanation_steps.get(sub_key))
                if parsed is None or not context or not sub_question or not cot:
                    skipped += 1
                    continue
                value, unit, answer_str = parsed
                normalized = {
                    "id": f"{Path(problem_file).parent.name}-{sub_key}",
                    "question": f"{context}\n\n{sub_question}",
                    "cot": cot,
                    "answer_num": value,
                    "unit": normalize_unit(unit),
                    "answer_str": answer_str,
                }
                samples.append(build_physics_sft_sample(normalized))
                if config.max_samples and len(samples) >= config.max_samples:
                    LOGGER.info("Mapped %s: kept %s numeric sub-questions, skipped %s non-numeric/unmapped sub-questions", config.name, len(samples), skipped)
                    return samples
    LOGGER.info("Mapped %s: kept %s numeric sub-questions, skipped %s non-numeric/unmapped sub-questions", config.name, len(samples), skipped)
    return samples


def load_external_hf(config: DataConfig) -> list[dict[str, Any]]:
    """Load and map a HuggingFace dataset into internal SFT sample format."""
    try:
        from datasets import load_dataset
    except ImportError:
        LOGGER.warning("datasets is not installed; skipping %s", config.hf_id)
        return []
    if config.hf_id == "xw27/scibench":
        try:
            return _load_scibench(config)
        except Exception as exc:
            LOGGER.warning("Could not map %s: %s", config.hf_id, exc)
            return []
    if config.hf_id == "zhibei1204/PhysReason":
        try:
            return _load_physreason(config)
        except Exception as exc:
            LOGGER.warning("Could not map %s: %s", config.hf_id, exc)
            return []
    try:
        dataset = load_dataset(config.hf_id, split="train")
    except Exception as exc:
        LOGGER.warning("Could not load %s: %s", config.hf_id, exc)
        return []
    rows = list(dataset)
    if config.max_samples:
        rows = rows[: config.max_samples]
    samples = []
    for index, row in enumerate(rows):
        try:
            if config.subtask == 2:
                question = row.get("problem") or row.get("question")
                cot = row.get("solution") or row.get("answer")
                if not question or not cot:
                    continue
                parsed = parse_answer(str(cot))
                value = parsed.get("value_si") if parsed.get("value_si") is not None else parsed.get("value")
                if value is None:
                    LOGGER.warning("Skipping %s row %s: final numeric answer not found", config.name, index)
                    continue
                unit = parsed.get("unit") or "1"
                normalized = {
                    "id": f"{config.name}-{index}",
                    "question": question,
                    "cot": str(cot),
                    "answer_num": float(value),
                    "unit": unit,
                    "answer_str": f"{float(value):g} {unit}",
                }
                samples.append(build_physics_sft_sample(normalized))
            elif config.hf_id == "yale-nlp/FOLIO":
                story = row.get("story", "")
                premises = _sentences(story)
                label = str(row.get("label", "")).strip()
                mapped = {"premises-NL": premises, "question": row.get("conclusion", ""), "answer": label, "explanation": row.get("explanation", label), "idx": [0]}
                samples.append(build_logic_sft_sample(mapped))
        except Exception as exc:
            LOGGER.warning("Skipping %s row %s: %s", config.name, index, exc)
    return samples


def split_and_save(records: list[dict[str, Any]], split_ratios: tuple[float, float, float], seed: int, output_dir: str | Path, prefix: str) -> dict[str, int]:
    """Shuffle, split, and save train/dev/test JSONL files."""
    output = Path(output_dir)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * split_ratios[0])
    dev_end = train_end + int(len(shuffled) * split_ratios[1])
    splits = {"train": shuffled[:train_end], "dev": shuffled[train_end:dev_end], "test": shuffled[dev_end:]}
    test_path = output / f"{prefix}_test.jsonl"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("# DO NOT USE FOR TRAINING\n", encoding="utf-8")
    with test_path.open("a", encoding="utf-8") as handle:
        for record in splits["test"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_jsonl(output / f"{prefix}_train.jsonl", splits["train"])
    _write_jsonl(output / f"{prefix}_dev.jsonl", splits["dev"])
    stats = {split: len(rows) for split, rows in splits.items()}
    stats_path = output / "split_stats.json"
    existing = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    existing[prefix] = stats
    stats_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    """Prepare local BTC and stage-1 external datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="output/data")
    parser.add_argument("--skip_external", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    summaries = []
    physics_raw = load_btc_physics(BTC_PHYSICS.path)
    physics = [build_physics_sft_sample(normalize_physics_record(row)) for row in filter_physics(physics_raw)]
    stats = split_and_save(physics, BTC_PHYSICS.split_ratios, BTC_PHYSICS.seed, args.output_dir, "physics")
    summaries.append(("physics", len(physics_raw), len(physics), stats))

    logic_raw = load_btc_logic(BTC_LOGIC.path)
    logic = [build_logic_sft_sample(row) for row in filter_logic(logic_raw)]
    stats = split_and_save(logic, BTC_LOGIC.split_ratios, BTC_LOGIC.seed, args.output_dir, "logic")
    summaries.append(("logic", len(logic_raw), len(logic), stats))

    if not args.skip_external:
        for config in [*EXTERNAL_PHYSICS, *EXTERNAL_LOGIC]:
            samples = load_external_hf(config)
            stats = split_and_save(samples, config.split_ratios, config.seed, args.output_dir, f"ext_{config.name}")
            summaries.append((config.name, len(samples), len(samples), stats))

    LOGGER.info("dataset | total | filtered_kept | train | dev | test")
    for name, total, kept, stats in summaries:
        LOGGER.info("%s | %s | %s | %s | %s | %s", name, total, kept, stats["train"], stats["dev"], stats["test"])


if __name__ == "__main__":
    main()
