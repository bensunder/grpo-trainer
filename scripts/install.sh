#!/usr/bin/env bash
# One-command installer for grpo-trainer on a GPU Linux box.
#
#   bash scripts/install.sh            # install + verify + preflight
#   bash scripts/install.sh --check    # only check the machine (GPU, driver, disk); installs nothing
#
# What it does, in order (each step is idempotent -- re-running skips finished work):
#   1. Detects the GPU and VRAM, picks a hardware profile (gpu16 / gpu24 / gpu80).
#   2. Checks the NVIDIA driver can run the CUDA 13 wheels (driver >= 580) and disk space.
#   3. Installs uv and a Python 3.12 venv INSIDE the install root (survives GridShare Stop/Resume).
#   4. Installs requirements.lock + this package.
#   5. Verifies torch sees the GPU, then runs `grpo preflight` with the detected profile.
#   6. Writes $ROOT/grpo-activate.sh so later sessions need one `source` line.
#
# Env overrides: GRPO_ROOT (default /workspace if it exists, else $HOME/grpo), GRPO_PROFILE (skip detection),
#                GRPO_MIN_DISK_GB (default 30).
# Full log: $ROOT/grpo-install.log
set -uo pipefail

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${GRPO_ROOT:-}" ]; then ROOT="$GRPO_ROOT"
elif [ -d /workspace ] && [ -w /workspace ]; then ROOT=/workspace
else ROOT="$HOME/grpo"; fi
mkdir -p "$ROOT"
LOG="$ROOT/grpo-install.log"
VENV="$ROOT/grpo-venv"
BOOT="$ROOT/.grpo-bootstrap"
MIN_DRIVER_MAJOR=580      # first driver branch supporting CUDA 13 (torch 2.13 wheels are cu130)
NEED_DISK_GB="${GRPO_MIN_DISK_GB:-30}"   # ~10 GB packages + model + checkpoints + headroom

c_ok() { printf '\033[32m[ ok ]\033[0m %s\n' "$*"; }
c_info() { printf '\033[36m[....]\033[0m %s\n' "$*"; }
c_warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; echo "Full log: $LOG"; exit 1; }
run() {  # run a command, append output to the log, fail with its tail on error
  echo "+ $*" >> "$LOG"
  if ! "$@" >> "$LOG" 2>&1; then
    echo "----- last 25 log lines -----"; tail -n 25 "$LOG"; echo "-----------------------------"
    die "command failed: $*"
  fi
}

echo "grpo-trainer installer  (repo: $REPO, install root: $ROOT)" | tee -a "$LOG"

# ---------------------------------------------------------------- 1. GPU + profile
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver on this machine"
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>>"$LOG" | head -1)"
[ -n "$GPU_LINE" ] || die "nvidia-smi returned no GPU"
GPU_NAME="$(echo "$GPU_LINE" | cut -d, -f1 | xargs)"
VRAM_MIB="$(echo "$GPU_LINE" | cut -d, -f2 | xargs)"
DRIVER="$(echo "$GPU_LINE" | cut -d, -f3 | xargs)"
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
VRAM_GB=$(( (VRAM_MIB + 512) / 1024 ))
c_ok "GPU: $GPU_NAME, ${VRAM_GB} GB, driver $DRIVER, ${GPU_COUNT} visible"

if [ -n "${GRPO_PROFILE:-}" ]; then PROFILE="$GRPO_PROFILE"
elif [ "$VRAM_MIB" -ge 72000 ]; then PROFILE=gpu80
elif [ "$VRAM_MIB" -ge 19500 ]; then PROFILE=gpu24
elif [ "$VRAM_MIB" -ge 15000 ]; then PROFILE=gpu16
else die "GPU has ${VRAM_GB} GB; this harness needs >= 16 GB (pick a bigger GPU)"; fi
[ -f "$REPO/configs/profiles/$PROFILE.yaml" ] || die "profile $PROFILE not found in configs/profiles"
c_ok "hardware profile: $PROFILE"

# ---------------------------------------------------------------- 2. driver + disk
DRIVER_MAJOR="${DRIVER%%.*}"
[ "$DRIVER_MAJOR" -ge "$MIN_DRIVER_MAJOR" ] 2>/dev/null \
  || die "driver $DRIVER is too old for CUDA 13 wheels (need >= $MIN_DRIVER_MAJOR). Pick another machine."
c_ok "driver $DRIVER supports CUDA 13 (>= $MIN_DRIVER_MAJOR)"

FREE_GB=$(( $(df -Pk "$ROOT" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
if [ -x "$VENV/bin/python" ]; then NEED=$(( NEED_DISK_GB / 3 )); else NEED=$NEED_DISK_GB; fi
[ "$FREE_GB" -ge "$NEED" ] || die "only ${FREE_GB} GB free at $ROOT; need >= ${NEED} GB"
c_ok "disk: ${FREE_GB} GB free at $ROOT"

if [ "$CHECK_ONLY" = 1 ]; then
  c_ok "machine check passed (nothing installed). Run without --check to install."
  exit 0
fi

# ---------------------------------------------------------------- 3. uv + Python 3.12 (inside ROOT)
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export HF_HOME="${HF_HOME:-$ROOT/hf-cache}"
if [ ! -x "$BOOT/bin/uv" ]; then
  c_info "installing uv"
  command -v python3 >/dev/null || die "python3 not found"
  run python3 -m venv "$BOOT"
  run "$BOOT/bin/pip" install -q --upgrade pip uv
fi
UV="$BOOT/bin/uv"
c_ok "uv $("$UV" --version | awk '{print $2}')"

if [ ! -x "$VENV/bin/python" ]; then
  c_info "creating Python 3.12 environment"
  if ! "$UV" venv --python 3.12 "$VENV" >> "$LOG" 2>&1; then
    c_warn "uv could not fetch Python 3.12; trying a system python3.12"
    command -v python3.12 >/dev/null || die "no Python 3.12 available (uv download failed and python3.12 not installed)"
    run "$UV" venv --python "$(command -v python3.12)" "$VENV"
  fi
fi
PYV="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$PYV" = "3.12" ] || die "venv has Python $PYV, expected 3.12 (delete $VENV and re-run)"
c_ok "Python $PYV at $VENV"

# ---------------------------------------------------------------- 4. packages
LOCK_SHA="$(sha256sum "$REPO/requirements.lock" | cut -c1-16)"
STAMP="$VENV/.grpo-lock-$LOCK_SHA"
if [ ! -f "$STAMP" ]; then
  c_info "installing locked dependencies (~8 GB, 5-15 min; progress in $LOG)"
  run "$UV" pip install --python "$VENV/bin/python" -r "$REPO/requirements.lock"
  touch "$STAMP"
else
  c_ok "dependencies already match requirements.lock"
fi
run "$UV" pip install --python "$VENV/bin/python" --no-deps -e "$REPO"
c_ok "grpo-trainer installed"

# ---------------------------------------------------------------- 5. verify
c_info "checking torch can use the GPU"
TORCH_OUT="$("$VENV/bin/python" -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")' 2>>"$LOG")"
echo "torch: $TORCH_OUT" >> "$LOG"
case "$TORCH_OUT" in
  True\ 13.*) c_ok "torch sees the GPU: $TORCH_OUT" ;;
  *) die "torch cannot use the GPU (got: '$TORCH_OUT'). Driver/CUDA mismatch; send me $LOG" ;;
esac

cat > "$ROOT/grpo-activate.sh" <<EOF
# Source this in every new shell:  source $ROOT/grpo-activate.sh
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export HF_HOME="$HF_HOME"
export GRPO_PROFILE="$PROFILE"
export VLLM_USE_FLASHINFER_SAMPLER=0  # avoid FlashInfer JIT build; see grpo_trainer/cli.py
. "$VENV/bin/activate"
cd "$REPO"
EOF
c_ok "wrote $ROOT/grpo-activate.sh"

c_info "running preflight (profile $PROFILE)"
cd "$REPO" || die "cannot cd to $REPO"
if ! "$VENV/bin/grpo" preflight --config configs/base.yaml --profile "$PROFILE" 2>&1 | tee -a "$LOG"; then
  die "preflight failed (see FAIL lines above)"
fi

echo
c_ok "INSTALL COMPLETE"
echo "Next:"
echo "  source $ROOT/grpo-activate.sh"
echo "  bash scripts/smoke.sh $PROFILE"
