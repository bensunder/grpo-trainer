#!/usr/bin/env bash
# Full lifecycle: [spec check] -> preflight -> profile -> train -> eval(base) -> eval(trained)
#                 -> [baseline] -> gate -> report -> [bundle if the gate passed].
# Every stage is idempotent: profile is cached, train resumes, eval overwrites its own report,
# baseline responses are cached.
#   scripts/pipeline.sh <profile> [--set key=value ...]
# Env:
#   CONFIG=configs/extraction.yaml   config file (default configs/base.yaml)
#   BASELINE_PROVIDERS="a b"         frontier providers to run (PAID; off unless set)
#   BUNDLE=1                         package the model after a passing gate
set -euo pipefail

PROFILE="${1:?usage: scripts/pipeline.sh <profile> [--set k=v ...]}"
shift || true
CONFIG="${CONFIG:-configs/base.yaml}"
COMMON=(--config "$CONFIG" --profile "$PROFILE" "$@")

SPEC="$(grpo validate "${COMMON[@]}" --print | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["task"].get("spec") or "")')"
if [ -n "$SPEC" ]; then
  echo "== spec check ($SPEC)"
  grpo spec check --spec "$SPEC"
fi

echo "== preflight"
grpo preflight "${COMMON[@]}"

echo "== profile"
grpo profile "${COMMON[@]}"

echo "== train"
bash scripts/train.sh "$PROFILE" "$@"

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

for provider in ${BASELINE_PROVIDERS:-}; do
  echo "== baseline $provider (paid API)"
  grpo baseline "${COMMON[@]}" --provider "$provider" --dry-run
  grpo baseline "${COMMON[@]}" --provider "$provider"
done

echo "== gate"
GATE_RC=0
grpo gate "${COMMON[@]}" || GATE_RC=$?

echo "== report"
grpo report "${COMMON[@]}"

if [ "$GATE_RC" -ne 0 ]; then
  echo "gate failed (exit $GATE_RC); report written, no bundle" >&2
  exit "$GATE_RC"
fi
if [ "${BUNDLE:-0}" = "1" ]; then
  echo "== bundle"
  grpo bundle "${COMMON[@]}" --force
fi
