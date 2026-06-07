#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/opt/conda/bin/python}
export PYTHONPATH=src

"$PYTHON" -m wake_lora.run_three_way \
  --config configs/qwen_medical_o1_small.json \
  "$@"
