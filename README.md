# EXACT 2026 Training and Inference

Production-oriented Python codebase for the EXACT 2026 challenge:

- Subtask 1: logic/educational-regulations reasoning
- Subtask 2: physics reasoning
- LoRA SFT training for `deepseek-ai/DeepSeek-R1-Distill-Qwen3-8B`
- vLLM OpenAI-compatible inference for `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

## Layout

```text
exact2026/
  train/
    prepare_data.py
    train_lora.py
    merge_lora.py
    data_config.py
  pipeline/
    router.py
    pipeline_physics.py
    pipeline_edu.py
    formula_db.py
    domain_classifier.py
    prompts.py
    inference.py
    voting.py
    answer_parser.py
    unit_normalizer.py
    direct_solver.py
run.py
requirements.txt
.env.example
```

The previous `exact_physics_pipeline/` package is retained for backwards compatibility and for the deterministic direct solver implementation.

## Install

```bash
pip install -r requirements.txt
```

For NVIDIA CUDA QLoRA training, install bitsandbytes separately:

```bash
pip install bitsandbytes
```

## Start vLLM

For NVIDIA CUDA GPUs:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --port 8000
```

For AMD ROCm GPUs, for example MI300X:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --port 8000
```

Install vLLM with ROCm support:

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.3
```

For a LoRA-trained model, replace the model ID in `vllm serve` with the local path to the merged model, such as `output/merged/`.

## Run Inference

Subtask 2, auto-detected from `question`:

```bash
python run.py --input '{"question":"A 12 V battery is connected to a 6 ohm resistor. What is the current?"}' --fast --pretty
```

Subtask 1, auto-detected from premises:

```bash
python run.py --input '{"premises-NL":["If a course is approved, it satisfies the policy.","This course is approved."],"questions":[{"question":"Does the course satisfy the policy?"}]}' --pretty
```

Batch JSONL:

```bash
python run.py --input_file problems.jsonl --output_file results.jsonl --fast
```

`--debug` includes `domain`, `formulas_used`, `source`, `all_answers`, and `raw_think`.

## Prepare SFT Data

```bash
python -m exact2026.train.prepare_data --output_dir output/data --skip_external
```

Remove `--skip_external` to fetch the configured HuggingFace stage-1 datasets.

## Train LoRA

```bash
python -m exact2026.train.train_lora \
  --stage 1 \
  --subtask both \
  --data_dir output/data \
  --output_dir output/lora \
  --epochs 2 \
  --batch_size 2
```

Backend behavior:

- CUDA: 4-bit NF4 QLoRA with bitsandbytes and `prepare_model_for_kbit_training`
- ROCm: full bf16 model, no bitsandbytes
- CPU: full fp32 with a warning

## Merge LoRA

```bash
python -m exact2026.train.merge_lora --adapter_dir output/lora --output_dir output/merged
```
