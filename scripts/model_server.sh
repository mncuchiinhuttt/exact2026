#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

usage() {
  cat <<EOF
Usage:
  scripts/model_server.sh download
  scripts/model_server.sh serve-fp8
  scripts/model_server.sh serve-awq

Environment overrides:
  MODEL_NAME                default: ${MODEL_NAME}
  PORT                      default: ${PORT}
  GPU_MEMORY_UTILIZATION    default: ${GPU_MEMORY_UTILIZATION}
  HF_HOME                   optional Hugging Face cache directory
  HF_TOKEN                  optional Hugging Face token

Install server dependencies first:
  pip install -r requirements-server.txt

Notes:
  - vLLM will download the model automatically on first serve.
  - The download command prefetches model files into the Hugging Face cache.
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    echo "Install server dependencies with: pip install -r requirements-server.txt" >&2
    exit 1
  fi
}

reasoning_args() {
  local args=(--reasoning-parser deepseek_r1)
  if vllm serve --help 2>&1 | grep -q -- "--enable-reasoning"; then
    args=(--enable-reasoning "${args[@]}")
  fi
  printf '%s\n' "${args[@]}"
}

check_flashinfer_versions() {
  python - <<'PY'
from importlib.metadata import PackageNotFoundError, version


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


flashinfer_version = package_version("flashinfer-python") or package_version("flashinfer")
cubin_version = package_version("flashinfer-cubin")

if flashinfer_version and cubin_version and flashinfer_version != cubin_version:
    print("FlashInfer package mismatch detected:", flush=True)
    print(f"  flashinfer-python/flashinfer: {flashinfer_version}", flush=True)
    print(f"  flashinfer-cubin:             {cubin_version}", flush=True)
    print("", flush=True)
    print("Repair with:", flush=True)
    print("  pip uninstall -y flashinfer flashinfer-python flashinfer-cubin", flush=True)
    print("  pip install --force-reinstall flashinfer-python==0.5.3 flashinfer-cubin==0.5.3", flush=True)
    raise SystemExit(1)
PY
}

download_model() {
  require_command huggingface-cli
  echo "Downloading ${MODEL_NAME} into the Hugging Face cache..."
  huggingface-cli download "${MODEL_NAME}"
}

serve_fp8() {
  require_command vllm
  check_flashinfer_versions
  mapfile -t reasoning_cli_args < <(reasoning_args)
  vllm serve "${MODEL_NAME}" \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 16384 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${reasoning_cli_args[@]}" \
    --port "${PORT}"
}

serve_awq() {
  require_command vllm
  check_flashinfer_versions
  mapfile -t reasoning_cli_args < <(reasoning_args)
  vllm serve "${MODEL_NAME}" \
    --quantization awq_marlin \
    --max-model-len 12288 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${reasoning_cli_args[@]}" \
    --port "${PORT}"
}

case "${1:-}" in
  download)
    download_model
    ;;
  serve-fp8)
    serve_fp8
    ;;
  serve-awq)
    serve_awq
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
