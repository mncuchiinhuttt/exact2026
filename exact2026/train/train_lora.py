"""DoRA SFT training for DeepSeek-R1-0528-Qwen3-8B."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LOGGER = logging.getLogger(__name__)
_LOG_HANDLES = []


class _Tee:
    """Write terminal output to both the console and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _setup_logging(output_dir: str | Path) -> Path:
    """Tee each training run's console output to a timestamped log file."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"train_{timestamp}.log"
    log_file = log_path.open("a", encoding="utf-8")
    _LOG_HANDLES.append(log_file)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)
    return log_path


def detect_backend() -> str:
    """Detect available GPU backend: cuda, rocm, or cpu."""
    if torch.cuda.is_available():
        if torch.version.hip is not None:
            return "rocm"
        return "cuda"
    return "cpu"


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records, ignoring comment lines."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            records.append(json.loads(line))
    return records


def _dataset_prefixes(stage: int, subtask: str) -> list[str]:
    """Return dataset file prefixes for the selected stage and subtask."""
    prefixes = []
    if subtask in {"1", "both"}:
        prefixes.append("logic")
    if subtask in {"2", "both"}:
        prefixes.append("physics")
    if stage == 1:
        if subtask in {"1", "both"}:
            prefixes.append("ext_FOLIO")
        if subtask in {"2", "both"}:
            prefixes.extend(["ext_scibench", "ext_PhysReason"])
    return prefixes


def _load_split(data_dir: Path, prefixes: list[str], split: str):
    """Load and concatenate selected split files as a datasets.Dataset."""
    from datasets import Dataset, concatenate_datasets

    datasets = []
    for prefix in prefixes:
        rows = _load_jsonl(data_dir / f"{prefix}_{split}.jsonl")
        if rows:
            datasets.append(Dataset.from_list(rows))
    if not datasets:
        raise FileNotFoundError(f"No {split} data found for prefixes {prefixes} in {data_dir}")
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def _load_model(model_name_or_path: str, backend: str, local_files_only: bool = False):
    """Load the base model with backend-appropriate precision/quantization."""
    from transformers import AutoModelForCausalLM

    common_kwargs = {"device_map": "auto", "local_files_only": local_files_only, "trust_remote_code": True}
    if backend == "cuda":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config, **common_kwargs)
        return prepare_model_for_kbit_training(model)
    if backend == "rocm":
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.bfloat16, **common_kwargs)
    LOGGER.warning("No GPU detected. Training on CPU will be extremely slow.")
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32, local_files_only=local_files_only, trust_remote_code=True)


def main() -> None:
    """Train a DoRA adapter from prepared ChatML JSONL data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[0, 1], default=1)
    parser.add_argument("--subtask", choices=["1", "2", "both"], default="both")
    parser.add_argument("--data_dir", default="output/data")
    parser.add_argument("--output_dir", default="output/lora")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--model_name_or_path",
        default=os.environ.get("TRAIN_MODEL_NAME_OR_PATH", os.environ.get("MODEL_NAME_OR_PATH", MODEL_ID)),
        help="Local model directory or Hugging Face model ID to use as the DoRA base model.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load the base model/tokenizer only from local files or the Hugging Face cache.",
    )
    args = parser.parse_args()
    log_path = _setup_logging(args.output_dir)
    LOGGER.info("Writing training log to %s", log_path)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    backend = detect_backend()
    LOGGER.info("Training base model: %s", args.model_name_or_path)
    LOGGER.info("Backend: %s", backend)
    LOGGER.info("Local files only: %s", args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_peft_model(
        _load_model(args.model_name_or_path, backend, args.local_files_only),
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True,
        ),
    )

    prefixes = _dataset_prefixes(args.stage, args.subtask)
    train_dataset = _load_split(Path(args.data_dir), prefixes, "train")
    dev_dataset = _load_split(Path(args.data_dir), prefixes, "dev")

    def render(example: dict) -> dict:
        """Apply the tokenizer chat template for SFTTrainer."""
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    train_dataset = train_dataset.map(render, remove_columns=train_dataset.column_names)
    dev_dataset = dev_dataset.map(render, remove_columns=dev_dataset.column_names)

    epochs = args.epochs if args.epochs is not None else (2 if args.stage == 1 else 1)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=5e-5 if args.stage == 1 else 2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=backend in {"cuda", "rocm"},
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        logging_dir=str(Path(args.output_dir) / "logs" / "trainer"),
        max_length=4096,
        dataset_text_field="text",
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    LOGGER.info("Saved DoRA adapter, adapter_config.json, and tokenizer to %s", args.output_dir)


if __name__ == "__main__":
    main()
