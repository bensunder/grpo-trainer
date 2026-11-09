#!/usr/bin/env bash
# Full lifecycle: preflight -> profile -> train -> eval(base) -> eval(trained) -> gate.
# Every stage is idempotent: profile is cached, train resumes, eval overwrites its own report.
#   scripts/pipeline.sh <profile> [--set key=value ...]
set -euo pipefail

PROFILE="${1:?usage: scripts/pipeline.sh <profile> [--set k=v ...]}"
shift || true
CONFIG="${CONFIG:-configs/base.yaml}"
COMMON=(--config "$CONFIG" --profile "$PROFILE" "$@")

echo "== preflight"
grpo preflight "${COMMON[@]}"

echo "== profile"
grpo profile "${COMMON[@]}"

echo "== train"
scripts/train.sh "$PROFILE" "$@"

RUN_DIR="$(grpo validate "${COMMON[@]}" --print | python3 -c \
  'import json,sys; c=json.load(sys.stdin); print(c["run"]["output_root"] + "/" + c["run"]["name"])')"
TRAINED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_model_path"])' \
  "$RUN_DIR/final/COMPLETE")"
BASE_MODEL="$(grpo validate "${COMMON[@]}" --print | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["model"]["name_or_path"])')"

echo "== eval base ($BASE_MODEL)"
grpo eval "${COMMON[@]}" --model "$BASE_MODEL" --label base

echo "== eval trained ($TRAINED)"
grpo eval "${COMMON[@]}" --model "$TRAINED" --label trained

echo "== gate"
grpo gate "${COMMON[@]}"
