# EXACT 2026 Training and Inference

Production-oriented Python codebase for the EXACT 2026 challenge:

- Subtask 1: logic/educational-regulations reasoning
- Subtask 2: physics reasoning
- DoRA SFT training for `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
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

## Current Structure

```text
.
├── exact2026/
│   ├── train/
│   │   ├── data_config.py      # Local BTC and external HuggingFace dataset configs
│   │   ├── prepare_data.py     # Filtering, normalization, ChatML SFT export
│   │   ├── train_lora.py       # DeepSeek-R1-0528-Qwen3-8B DoRA SFT
│   │   └── merge_lora.py       # Merge DoRA adapter into a standalone model
│   └── pipeline/
│       ├── router.py           # Detect input shape and route to Subtask 1 or 2
│       ├── pipeline_physics.py # Subtask 2 orchestration
│       ├── pipeline_edu.py     # Subtask 1 orchestration
│       ├── inference.py        # vLLM OpenAI-compatible client
│       ├── prompts.py          # Prompt builders for both subtasks
│       ├── formula_db.py       # Formula DB and keyword retrieval
│       ├── domain_classifier.py# Physics domain and subtask classifiers
│       ├── answer_parser.py    # Numeric ANSWER parsing and SI prefix handling
│       ├── unit_normalizer.py  # Unit canonicalization
│       ├── voting.py           # Self-consistency clustering and median voting
│       └── direct_solver.py    # Wrapper around existing deterministic solver
├── exact_physics_pipeline/     # Existing physics implementation retained
├── dataset/                    # Local EXACT 2026 dataset files
├── run.py                      # Root CLI for single and batch inference
├── requirements.txt
└── .env.example
```

## Install

Create a virtual environment and install dependencies inside it:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are on Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you are on Windows CMD:

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

```bash
pip install -r requirements.txt
```

PEFT must be version `0.9.0` or newer because DoRA support is required.

For NVIDIA CUDA quantized DoRA training, install bitsandbytes separately:

```bash
pip install bitsandbytes
```

## Start vLLM

For NVIDIA CUDA GPUs:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --trust-remote-code \
  --tokenizer Qwen/Qwen3-8B \
  --reasoning-parser deepseek_r1 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --port 8000
```

For AMD ROCm GPUs, for example MI300X:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser deepseek_r1 \
  --port 8000
```

Install vLLM with ROCm support:

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.3
```

For a DoRA-trained model, replace the model ID in `vllm serve` with the local path to the merged model, such as `output/merged/`.

## Current LLM Layer Implementation

The current LLM layer is implemented in `exact2026/pipeline/`. It does not load model weights inside the Python process. The Python code sends OpenAI-compatible chat requests to a running vLLM server.

### Runtime model

- Inference model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Training model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Client module: `exact2026/pipeline/inference.py`
- Client class: `VLLMClient`
- Default endpoint: `http://localhost:8000/v1`

The endpoint can be overridden with:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
export VLLM_API_KEY=EMPTY
```

### Request flow

```text
run.py
  -> exact2026.pipeline.router.route_and_run()
    -> classify_subtask()
      -> Subtask 2: run_physics()
      -> Subtask 1: run_edu()
```

### Subtask 2 physics flow

```text
run_physics(question, config)
  1. classify_domain(question)
  2. try_direct_solve(question, domain)
  3. retrieve_formulas(domain, question)
  4. build_physics_prompt(question, formulas)
  5. VLLMClient.generate_k(...)
  6. parse_answer(content, reasoning)
  7. vote(parsed_answers)
  8. optionally replace close model answer with deterministic direct-solver result
  9. return contest-shaped answer, explanation, cot, confidence, debug fields
```

Important behavior:

- `--fast` sets `k=1`, uses compact prompts, and returns the direct solver result immediately when the direct solver matches the problem.
- Without `--fast`, the pipeline still computes a direct-solver result when possible, but uses it only as a refinement if the LLM consensus is within 5 percent and the units match.
- Physics self-consistency uses numeric parsing, unit normalization, SI prefix expansion, relative-tolerance clustering, and median selection.

### Subtask 1 logic/education flow

```text
run_edu(premises, question, config)
  1. build_edu_prompt(premises, question)
  2. VLLMClient.generate_k(...)
  3. parse final ANSWER line as Yes/No/True/False/A/B/C/D
  4. majority vote by exact string match
  5. return answer, explanation, cot, premises, confidence
```

Current limitation:

- The `fol` field is present but empty.
- Z3/FOL verification is a placeholder in the prompt and has not been integrated into runtime execution yet.

### vLLM call details

`VLLMClient.generate_one()` sends:

```python
client.chat.completions.create(
    model=VLLM_MODEL,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096,
    n=1,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
```

`generate_k()` runs `k` independent calls in parallel with `ThreadPoolExecutor`. The returned fields are:

```text
content    # final assistant text
reasoning  # DeepSeek/vLLM reasoning_content when available
```

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

## Train DoRA

```bash
python -m exact2026.train.train_lora \
  --stage 1 \
  --subtask both \
  --data_dir output/data \
  --output_dir output/lora \
  --epochs 2 \
  --batch_size 2
```

For AMD MI300X, the script loads the 8B model in bf16 without bitsandbytes. The MI300X has enough memory to use a larger per-device batch size than the default.

Recommended MI300X starting command:

```bash
python -m exact2026.train.train_lora \
  --stage 1 \
  --subtask both \
  --data_dir output/data \
  --output_dir output/lora \
  --epochs 2 \
  --batch_size 8
```

To train from a self-hosted/local copy of the base model, point the trainer at the local model directory:

```bash
python -m exact2026.train.train_lora \
  --stage 1 \
  --subtask both \
  --data_dir output/data \
  --output_dir output/lora \
  --epochs 2 \
  --batch_size 32 \
  --model_name_or_path /models/DeepSeek-R1-0528-Qwen3-8B \
  --local_files_only
```

You can also set `TRAIN_MODEL_NAME_OR_PATH=/models/DeepSeek-R1-0528-Qwen3-8B` instead of passing `--model_name_or_path`.

The script currently uses:

```text
gradient_accumulation_steps = 8
effective_batch_size = batch_size * 8
```

Practical MI300X tuning:

```text
--batch_size 8   -> effective batch 64
--batch_size 12  -> effective batch 96
--batch_size 16  -> effective batch 128
```

Start with `--batch_size 8`, then increase while monitoring memory:

```bash
rocm-smi
```

If training is stable and memory is underused, try `--batch_size 12` or `--batch_size 16`. If loss behavior gets worse after increasing batch size, lower the batch size or reduce the learning rate.

Backend behavior:

- CUDA: 4-bit NF4 quantized DoRA with bitsandbytes and `prepare_model_for_kbit_training`
- ROCm: full bf16 model, no bitsandbytes
- CPU: full fp32 with a warning

## Merge DoRA

```bash
python -m exact2026.train.merge_lora \
  --adapter_dir output/lora \
  --output_dir output/merged \
  --base_model /models/DeepSeek-R1-0528-Qwen3-8B \
  --local_files_only
```
