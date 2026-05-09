# EXACT 2026 Physics Pipeline

Modular physics reasoning pipeline for EXACT 2026 Subtask 2. The Python pipeline does not load the model directly. It sends OpenAI-compatible chat requests to a local vLLM server running `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.

## Project Layout

```text
exact_physics_pipeline/
  domain.py       # rule-based domain classifier
  formulas.py     # static formula database and formula retrieval
  prompts.py      # system prompt builder
  inference.py    # OpenAI-compatible vLLM calls
  parsing.py      # ANSWER parsing
  voting.py       # self-consistency voting
  pipeline.py     # run_pipeline orchestration
scripts/
  model_server.sh # download and serve the model with vLLM
demo.py           # runs 3 example physics problems
requirements.txt
requirements-server.txt
```

## Recommended Setup

Use a fresh environment for vLLM. This avoids dependency conflicts with Lightning, pandas, matplotlib, scipy, scikit-learn, and OpenCV.

```bash
pip install -r requirements-server.txt
pip install -r requirements.txt
```

The lightweight client dependency is:

```bash
pip install -r requirements.txt
```

The vLLM server dependencies are:

```bash
pip install -r requirements-server.txt
```

## Download The Model

The server can download the model automatically on first launch, but pre-downloading is useful:

```bash
scripts/model_server.sh download
```

Optional cache location:

```bash
HF_HOME=/path/to/hf-cache scripts/model_server.sh download
```

If Hugging Face authentication is needed:

```bash
export HF_TOKEN=your_token_here
scripts/model_server.sh download
```

## Start The vLLM Server

FP8 mode, recommended when the GPU has enough VRAM:

```bash
scripts/model_server.sh serve-fp8
```

Lower-VRAM AWQ mode:

```bash
scripts/model_server.sh serve-awq
```

The launcher automatically adapts to vLLM versions where `--enable-reasoning`
has been removed. It always passes `--reasoning-parser deepseek_r1`, and only
adds `--enable-reasoning` if your installed `vllm serve --help` supports it.

Useful overrides:

```bash
PORT=8001 scripts/model_server.sh serve-fp8
GPU_MEMORY_UTILIZATION=0.85 scripts/model_server.sh serve-fp8
MAX_MODEL_LEN_FP8=8192 scripts/model_server.sh serve-fp8
```

The pipeline expects the server at:

```text
http://localhost:8000/v1
```

## Speed Tips

The largest speed cost is self-consistency: `k=5` means five model calls per
problem. Use smaller `k` and shorter output while debugging:

```bash
python evaluate.py --fast --limit 3
python run.py --fast --problem "A 12 V battery is connected to a 6 ohm resistor. What is the current?"
```

`--fast` is shorthand for:

```text
--direct-first --k 1 --max-tokens 512 --max-formulas 8 --compact --no-thinking
```

The direct solver covers common Ohm's law, resistor network, capacitor,
electrostatics, and energy formulas without calling the LLM. If it cannot match
the problem confidently, `run.py --fast` falls back to the compact vLLM path.

In the standard vLLM path, the same deterministic solver is also used as a
precision refinement pass. When the self-consistency consensus is in the same
unit family and close to a known formula result, the pipeline returns the exact
computed value and a visible four-step explanation. This reduces harmless but
score-costly rounding drift such as `63 V` versus `63.2455532034 V`.

You can also set environment defaults:

```bash
EXACT_K=1 EXACT_MAX_TOKENS=1024 python evaluate.py --limit 3
```

For final scoring, return to:

```bash
python evaluate.py --k 5 --max-tokens 4096
```

Benchmark the accuracy/speed tradeoff:

```bash
python evaluate.py --fast
python evaluate.py --k 3 --max-tokens 1024 --max-formulas 12 --compact
python evaluate.py --k 5 --max-tokens 4096
```

Server-side, shorter context reduces KV-cache memory and can improve throughput:

```bash
MAX_MODEL_LEN_FP8=8192 scripts/model_server.sh serve-fp8
```

Keeping the vLLM server warm between runs matters; do not restart it for every
evaluation unless you changed server settings.

## Run The Demo

Start the vLLM server in one terminal. In another terminal:

```bash
python demo.py
```

The demo runs three hardcoded examples:

1. Circuit problem
2. Electrostatics problem
3. Capacitor energy problem

Each result prints the domain, answer, unit, confidence, and the first 500 characters of the explanation.

## Structured Contest Output

Use `run.py` when you need the contest/API response shape directly:

```bash
python run.py --problem "A 12 V battery is connected to a 6 ohm resistor. What is the current?"
```

For latency-sensitive API endpoints, use fast mode:

```bash
python run.py --fast --problem "A 12 V battery is connected to a 6 ohm resistor. What is the current?"
```

or pipe input through stdin:

```bash
echo "A 10 microfarad capacitor is charged to 50 V. What energy is stored?" | python run.py
```

The output shape is:

```json
{
  "answer": "2.0 A",
  "explanation": "Full visible explanation from the model or deterministic refinement.",
  "fol": "∀x (CircuitProblem(x) → Uses(OhmsLaw, x) ∨ Uses(KirchhoffLaw, x))",
  "cot": [
    "STEP 1 - READ: ...",
    "STEP 2 - PLAN: ...",
    "STEP 3 - SOLVE: ...",
    "STEP 4 - ANSWER: ..."
  ],
  "premises": [
    "Ohm's Law: V = I * R"
  ],
  "confidence": 0.8
}
```

For an API endpoint, import the function:

```python
from run import run_structured

def solve_endpoint(problem: str) -> dict:
    return run_structured(
        problem,
        k=1,
        max_tokens=512,
        max_formulas=8,
        compact_prompt=True,
        enable_thinking=False,
        direct_first=True,
    )
```

## Run Evaluation Cases

The repository includes curated test cases with expected answers in
`exact_physics_pipeline/test_cases.py`. They cover simple, moderate, and complex
problems across circuits, electrostatics, and energy.

Run all cases against the live vLLM server:

```bash
python evaluate.py
```

Run a smaller subset while debugging:

```bash
python evaluate.py --limit 3 --k 1
python evaluate.py --domain circuit
python evaluate.py --difficulty complex
```

The evaluator checks the predicted numeric answer with relative tolerance and
compares the unit against the expected or accepted equivalent units.

To grade the model's explanation quality with OpenAI, add `OPENAI_API_KEY` to
`.env` and run:

```bash
python evaluate.py --judge
```

By default this uses `gpt-5.4` and prints a judge score out of 100 plus a
short feedback note. You can override the judge model:

```bash
python evaluate.py --judge --judge-model gpt-5.4-mini
python evaluate.py --judge --judge-failures-only
```

Every evaluation writes a timestamped JSON log to `logs/`, for example:

```text
logs/evaluation_20260507_163000Z.json
```

Each log contains run timing, evaluator settings, per-case timing, raw scoring,
optional judge feedback, and a structured output block:

```json
{
  "run": {
    "started_at": "2026-05-07T16:30:00+00:00",
    "finished_at": "2026-05-07T16:34:20+00:00",
    "duration_seconds": 260.1,
    "total_cases": 22,
    "passed_cases": 20,
    "failed_cases": 2,
    "accuracy": 0.91,
    "settings": {
      "k": 5,
      "tolerance": 0.03,
      "judge": true,
      "judge_failures_only": false,
      "judge_model": "gpt-5.4-mini"
    }
  },
  "cases": [
    {
      "index": 1,
      "case": {
        "id": "circuit_simple_series_current",
        "difficulty": "simple",
        "domain": "circuit",
        "problem": "..."
      },
      "duration_seconds": 12.4,
      "scoring": {
        "expected_answer": 1.0,
        "expected_unit": "A",
        "predicted_answer": 1.0,
        "predicted_unit": "A",
        "confidence": 0.8,
        "numeric_ok": true,
        "unit_ok": true,
        "passed": true,
        "judge": {
          "score": 98,
          "feedback": "Correct formula and unit handling with clear substitutions."
        }
      },
      "structured_output": {
        "answer": "1.0 A",
        "explanation": "The visible final explanation from the pipeline.",
        "fol": "∀x (CircuitProblem(x) → Uses(OhmsLaw, x) ∨ Uses(KirchhoffLaw, x))",
        "cot": [
          "STEP 1 - READ: ...",
          "STEP 2 - PLAN: ...",
          "STEP 3 - SOLVE: ...",
          "STEP 4 - ANSWER: ..."
        ],
        "premises": [
          "Ohm's Law: V = I * R",
          "Series Resistance: R_total = R1 + R2 + ... + Rn"
        ],
        "confidence": 0.8
      }
    }
  ]
}
```

To summarize an existing log and find parser failures or rounding drift:

```bash
python analyze_logs.py logs/evaluation_20260508_031026Z.json
```

## Use The Pipeline In Python

```python
from exact_physics_pipeline import run_pipeline

problem = "A 12V battery is connected to two resistors in series: R1 = 4Ω and R2 = 8Ω. What is the current?"
result = run_pipeline(problem)

print(result["answer"], result["unit"], result["confidence"])
```

`run_pipeline(problem: str)` returns:

```python
{
    "domain": str,
    "formulas_used": list[str],
    "answer": float | None,
    "unit": str | None,
    "confidence": float,
    "explanation": str,
    "raw_think": str,
    "all_answers": list[dict],
    "source": "vllm" | "direct_solver" | "direct_refinement",
    "model_answer": dict | None,
}
```

## Dependency Conflict Notes

If you install vLLM into a shared Lightning environment, pip may report conflicts such as:

```text
pandas requires numpy<2
opencv-python-headless requires numpy>=2
lightning-sdk requires urllib3<=2.5.0
flashinfer-cubin version does not match flashinfer version
```

The best fix is a separate vLLM environment:

```bash
pip install -r requirements-server.txt
pip install -r requirements.txt
```

The `requirements-server.txt` file pins a few packages to reduce conflicts in shared environments, but a clean environment is still strongly recommended.

If vLLM fails with a FlashInfer mismatch, repair those two packages directly:

```bash
pip uninstall -y flashinfer flashinfer-python flashinfer-cubin
pip install --force-reinstall flashinfer-python==0.5.3 flashinfer-cubin==0.5.3
```

Then retry:

```bash
scripts/model_server.sh serve-fp8
```

## Quick Checks

Run these checks before switching to the GPU server:

```bash
python -m py_compile demo.py exact_physics_pipeline/*.py
python -c "from exact_physics_pipeline import run_pipeline; print(run_pipeline)"
```

If the server is not running, `demo.py` will print a clear message instead of loading a model in Python.
