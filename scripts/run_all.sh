#!/usr/bin/env bash
# Server-side all-in-one. Safe to re-run.
#   bash scripts/run_all.sh          # ensure install, re-gate existing smoke results
#   bash scripts/run_all.sh --full   # ...and launch the full pipeline in the background (survives SSH loss)
#
# Background run writes $ROOT/pipeline-<timestamp>.log and $ROOT/pipeline.pid.
set -uo pipefail

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${GRPO_ROOT:-}" ]; then ROOT="$GRPO_ROOT"
elif [ -d /workspace ]; then ROOT=/workspace
else ROOT="$HOME/grpo"; fi

say() { printf '\n== %s\n' "$*"; }

# ---- 1. environment (installs only if missing; install.sh is idempotent)
if [ ! -f "$ROOT/grpo-activate.sh" ]; then
  say "no environment yet: running installer"
  GRPO_ROOT="$ROOT" bash "$REPO/scripts/install.sh" || { echo "install failed; see $ROOT/grpo-install.log"; exit 1; }
fi
# shellcheck disable=SC1091
. "$ROOT/grpo-activate.sh" || { echo "cannot source $ROOT/grpo-activate.sh"; exit 1; }
cd "$REPO" || exit 1
export VLLM_USE_FLASHINFER_SAMPLER=0
PROFILE="${GRPO_PROFILE:-gpu16}"
say "code: $(grpo --help >/dev/null 2>&1 && echo ok) | profile: $PROFILE | repo: $REPO"

# ---- 2. re-gate existing smoke results with the current gate (format vs reasoning split)
if [ -f "runs/smoke-$PROFILE/eval/base.json" ] && [ -f "runs/smoke-$PROFILE/eval/trained.json" ]; then
  say "gate on existing smoke results (runs/smoke-$PROFILE)"
  grpo gate --config configs/base.yaml --profile "$PROFILE" --set "run.name=smoke-$PROFILE" \
    || echo "(smoke gate did not pass with production thresholds; see reasons above)"
else
  say "no smoke results found for $PROFILE; skipping smoke gate"
fi

# ---- 3. optional full run, detached
if [ "$FULL" = 1 ]; then
  PIDFILE="$ROOT/pipeline.pid"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    say "a pipeline is already running (pid $(cat "$PIDFILE")); not starting another"
    exit 0
  fi
  LOG="$ROOT/pipeline-$(date +%Y%m%d-%H%M%S).log"
  say "launching full pipeline in background -> $LOG"
  nohup setsid bash scripts/pipeline.sh "$PROFILE" > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 2
  if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "running (pid $(cat "$PIDFILE"))."
    echo "watch:  tail -f $LOG"
    echo "done when the log ends with the gate JSON and 'gate passed' (or 'gate failed: ...')."
  else
    echo "pipeline exited immediately; last lines:"; tail -n 30 "$LOG"; exit 1
  fi
fi
