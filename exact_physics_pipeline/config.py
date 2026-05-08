"""Configuration for the vLLM-backed EXACT 2026 physics pipeline.

The model is served outside Python. You can use scripts/model_server.sh, or start
vLLM directly before running the demo:

FP8, recommended when VRAM is sufficient:
    vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --quantization fp8 \
      --kv-cache-dtype fp8 \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.90 \
      --enable-reasoning \
      --reasoning-parser deepseek_r1 \
      --port 8000

AWQ Marlin, useful when GPU VRAM is below roughly 20GB:
    vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --quantization awq_marlin \
      --max-model-len 12288 \
      --gpu-memory-utilization 0.90 \
      --enable-reasoning \
      --reasoning-parser deepseek_r1 \
      --port 8000
"""

# === DEPENDENCIES ===
# pip install openai
# vLLM server must be running before executing this pipeline.

MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

DEFAULT_K = 5
MAX_WORKERS = 5
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 4096

EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": True}}
