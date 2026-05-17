"""LoRA SFT training for DeepSeek-R1-Distill-Qwen3-8B."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen3-8B"
LOGGER = logging.getLogger(__name__)


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


def _load_model(model_id: str, backend: str):
    """Load the base model with backend-appropriate precision/quantization."""
    from transformers import AutoModelForCausalLM

    if backend == "cuda":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        return prepare_model_for_kbit_training(model)
    if backend == "rocm":
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    LOGGER.warning("No GPU detected. Training on CPU will be extremely slow.")
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)


def main() -> None:
    """Train a LoRA adapter from prepared ChatML JSONL data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[0, 1], default=1)
    parser.add_argument("--subtask", choices=["1", "2", "both"], default="both")
    parser.add_argument("--data_dir", default="output/data")
    parser.add_argument("--output_dir", default="output/lora")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    backend = detect_backend()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_peft_model(
        _load_model(MODEL_ID, backend),
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
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
    training_args = TrainingArguments(
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
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        max_seq_length=4096,
        dataset_text_field="text",
        packing=False,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "lora_config.json").write_text(model.peft_config["default"].to_json_string(), encoding="utf-8")


if __name__ == "__main__":
    main()

