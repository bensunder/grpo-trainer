#!/usr/bin/env bash
# Launch GRPO training under accelerate.
#   scripts/train.sh <profile> [extra grpo args...]
# e.g. scripts/train.sh gpu24
#      scripts/train.sh gpu24 --set grpo.max_steps=50
set -euo pipefail

PROFILE="${1:?usage: scripts/train.sh <profile> [args...]}"
shift || true
CONFIG="${CONFIG:-configs/base.yaml}"

case "$PROFILE" in
  multi) ACCEL="configs/accelerate/multi_zero3.yaml" ;;
  *)     ACCEL="configs/accelerate/single.yaml" ;;
esac

exec accelerate launch --config_file "$ACCEL" -m grpo_trainer.cli train \
  --config "$CONFIG" --profile "$PROFILE" "$@"
