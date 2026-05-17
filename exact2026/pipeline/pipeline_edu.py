"""Full Subtask 1 logic/education inference pipeline."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from exact2026.pipeline.inference import VLLMClient
from exact2026.pipeline.pipeline_physics import extract_cot_block, extract_explanation_block
from exact2026.pipeline.prompts import build_edu_prompt


ANSWER_RE = re.compile(r"ANSWER:\s*(Yes|No|True|False|A|B|C|D)\b", re.IGNORECASE)


def _parse_logic_answer(text: str) -> str | None:
    """Extract a categorical logic answer from a generation."""
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1]
    return answer[0].upper() + answer[1:].lower() if len(answer) > 1 else answer.upper()


def run_edu(premises: list[str], question: str, config: dict | None = None) -> dict[str, Any]:
    """Run Subtask 1 with majority voting over exact answer strings."""
    cfg = {"k": 3, "mode": "full", "max_tokens": 2048}
    cfg.update(config or {})
    if cfg.get("fast"):
        cfg.update({"k": 1, "mode": "compact"})

    messages = build_edu_prompt(premises, question, mode=cfg.get("mode", "full"))
    generations = VLLMClient().generate_k(messages, k=int(cfg["k"]), max_tokens=int(cfg.get("max_tokens", 2048)))
    answers = [_parse_logic_answer(gen["content"]) for gen in generations]
    counts = Counter(answer for answer in answers if answer)
    majority_answer, winning_count = counts.most_common(1)[0] if counts else (None, 0)
    winning_gen = next((gen for gen, ans in zip(generations, answers) if ans == majority_answer), generations[0] if generations else {"content": ""})
    return {
        "answer": majority_answer,
        "confidence": winning_count / max(1, int(cfg["k"])),
        "explanation": extract_explanation_block(winning_gen["content"]),
        "cot": extract_cot_block(winning_gen["content"]),
        "fol": "",
        "premises": premises,
        "source": "vllm",
    }

