#!/usr/bin/env bash
# Dedicated vLLM generation server for the `multi` profile. Run on the GPU NOT used for training.
#   CUDA_VISIBLE_DEVICES=0 scripts/vllm_server.sh Qwen/Qwen2.5-7B-Instruct
# then train on the rest:
#   CUDA_VISIBLE_DEVICES=1,2,3 scripts/train.sh multi
set -euo pipefail
MODEL="${1:?usage: scripts/vllm_server.sh <model> [port]}"
PORT="${2:-8000}"
exec trl vllm-serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --gpu_memory_utilization 0.9 --dtype bfloat16 --enable_prefix_caching true
