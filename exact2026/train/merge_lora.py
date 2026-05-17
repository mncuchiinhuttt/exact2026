"""Merge a trained LoRA adapter into the DeepSeek base model."""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact2026.train.train_lora import MODEL_ID, detect_backend


def main() -> None:
    """Merge LoRA weights and save a standalone model directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_dir", default="output/lora")
    parser.add_argument("--output_dir", default="output/merged")
    parser.add_argument("--base_model", default=MODEL_ID)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    backend = detect_backend()
    dtype = torch.bfloat16 if backend in {"cuda", "rocm"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map="auto" if backend != "cpu" else None, trust_remote_code=True)
    merged = PeftModel.from_pretrained(model, args.adapter_dir).merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
