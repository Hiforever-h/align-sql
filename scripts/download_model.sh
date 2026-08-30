#!/usr/bin/env bash
set -euo pipefail

model_name="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"

if [[ -z "${HF_HOME:-}" ]]; then
  echo "HF_HOME must be set before downloading model weights." >&2
  exit 1
fi
if [[ -z "${HF_HUB_CACHE:-}" ]]; then
  echo "HF_HUB_CACHE must be set before downloading model weights." >&2
  exit 1
fi

hf download "${model_name}"

