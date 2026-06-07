#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${PYTHONPATH:-src}"

MODEL_PATH="${MODEL_PATH:-/home/jovyan/imagenet-1k/qwen_model}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

"${PYTHON_BIN}" -m wake_lora.mcqa_eval \
  --model_path "${MODEL_PATH}" \
  --output_dir outputs/medxpertqa_text_50_chat_option/base \
  --max_samples "${MAX_SAMPLES}" \
  --prompt_style chat \
  --scoring_method option_likelihood \
  --length_normalize \
  --save_examples 10

"${PYTHON_BIN}" -m wake_lora.mcqa_eval \
  --model_path "${MODEL_PATH}" \
  --adapter_path outputs/pilot_medical_o1_en_32_e1_lam01/standard_lora/adapter_final \
  --output_dir outputs/medxpertqa_text_50_chat_option/standard_lora_32_seed42 \
  --max_samples "${MAX_SAMPLES}" \
  --prompt_style chat \
  --scoring_method option_likelihood \
  --length_normalize \
  --save_examples 10

"${PYTHON_BIN}" -m wake_lora.mcqa_eval \
  --model_path "${MODEL_PATH}" \
  --adapter_path outputs/pilot_medical_o1_en_32_e1_lam01/wake_lora/adapter_final \
  --output_dir outputs/medxpertqa_text_50_chat_option/wake_lora_32_seed42 \
  --max_samples "${MAX_SAMPLES}" \
  --prompt_style chat \
  --scoring_method option_likelihood \
  --length_normalize \
  --save_examples 10
