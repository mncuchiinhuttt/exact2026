"""Dataset source configuration for EXACT 2026 SFT data preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Configuration for one local or HuggingFace dataset source."""

    name: str
    type: str | None = None
    path: str | None = None
    hf_id: str | None = None
    subtask: int = 0
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42
    stage: int = 1
    max_samples: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


BTC_PHYSICS = DataConfig(
    name="physics",
    type="csv",
    path="dataset/Physics_Problems_Text_Only/Physics_Problems_Text_Only.csv",
    subtask=2,
)

BTC_LOGIC = DataConfig(
    name="logic",
    type="jsonl",
    path="dataset/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json",
    subtask=1,
)

EXTERNAL_PHYSICS = [
    DataConfig(name="scibench", hf_id="xw27/scibench", subtask=2, stage=1),
    DataConfig(name="PhysReason", hf_id="zhibei1204/PhysReason", subtask=2, stage=1, max_samples=5000),
]

EXTERNAL_LOGIC = [
    DataConfig(name="FOLIO", hf_id="yale-nlp/FOLIO", subtask=1, stage=1),
]

SOURCE_CONFIGS = [BTC_PHYSICS, BTC_LOGIC, *EXTERNAL_PHYSICS, *EXTERNAL_LOGIC]

