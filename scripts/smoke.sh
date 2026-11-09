#!/usr/bin/env bash
# End-to-end smoke test on a real GPU (~20-40 min on a 24 GB card for the 1.5B default).
# Proves: preflight, profiling, training, checkpointing, SIGTERM -> graceful checkpoint -> resume,
# adapter merge, eval, gate plumbing. It does NOT prove the model improves (20 steps is too few);
# the gate is run with min_accuracy_delta=-1 so only the plumbing is asserted.
#   scripts/smoke.sh [profile] [--set key=value ...]   (default gpu24)
#   e.g. scripts/smoke.sh gpu16 --set vllm.gpu_memory_utilization=0.25
set -euo pipefail

PROFILE="${1:-gpu24}"
shift || true
CONFIG="${CONFIG:-configs/base.yaml}"
SMOKE=(
  --set run.name=smoke-"$PROFILE"
  --set grpo.max_steps=20
  --set grpo.save_steps=5
  --set grpo.logging_steps=1
  --set profiler.max_prompts=64
  --set eval.limit=100
  --set eval.pass_k=1
  --set eval.gate.min_accuracy_delta=-1.0
  --set eval.gate.max_format_regression=1.0
  "$@"  # caller overrides win (applied last)
)
COMMON=(--config "$CONFIG" --profile "$PROFILE" "${SMOKE[@]}")
RUN_DIR="runs/smoke-$PROFILE"
START=$(date +%s)

rm -rf "$RUN_DIR"
echo "== [1/6] preflight";  grpo preflight "${COMMON[@]}"
echo "== [2/6] profile";    grpo profile "${COMMON[@]}"

echo "== [3/6] train, interrupted with SIGTERM after first checkpoint"
setsid bash scripts/train.sh "$PROFILE" "${SMOKE[@]}" &
TRAIN_PID=$!
for _ in $(seq 1 360); do
  if compgen -G "$RUN_DIR/checkpoints/checkpoint-*" > /dev/null; then break; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then echo "trainer exited before first checkpoint"; exit 1; fi
  sleep 5
done
compgen -G "$RUN_DIR/checkpoints/checkpoint-*" > /dev/null || { echo "no checkpoint after 30 min"; exit 1; }
echo "   checkpoint found; sending SIGTERM to trainer process group"
kill -TERM -- -"$TRAIN_PID" 2>/dev/null || true
while pgrep -g "$TRAIN_PID" > /dev/null 2>&1; do sleep 2; done
[ -f "$RUN_DIR/final/COMPLETE" ] && { echo "run completed before interrupt; resume not exercised"; }

echo "== [4/6] resume to completion"
bash scripts/train.sh "$PROFILE" "${SMOKE[@]}"
[ -f "$RUN_DIR/final/COMPLETE" ] || { echo "FAIL: final/COMPLETE missing"; exit 1; }
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("resumes"), "no resume recorded"; print("   resume recorded:", m["resumes"][-1])' "$RUN_DIR/manifest.json"

TRAINED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_model_path"])' "$RUN_DIR/final/COMPLETE")"
BASE_MODEL="$(grpo validate "${COMMON[@]}" --print | python3 -c 'import json,sys; print(json.load(sys.stdin)["model"]["name_or_path"])')"
echo "== [5/6] eval base + trained"
grpo eval "${COMMON[@]}" --model "$BASE_MODEL" --label base
grpo eval "${COMMON[@]}" --model "$TRAINED" --label trained
echo "== [6/6] gate (plumbing only)"
grpo gate "${COMMON[@]}"

echo "SMOKE PASSED in $(( ($(date +%s) - START) / 60 )) min. Reward curves: tensorboard --logdir $RUN_DIR/checkpoints"
