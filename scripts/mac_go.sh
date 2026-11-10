#!/usr/bin/env bash
# Run on your Mac:  bash ~/Downloads/mac_go.sh
# Uploads the newest ~/Downloads/grpo-trainer*.zip to the GridShare box, updates the code in place
# (runs/ results are kept), re-gates the smoke results, and optionally starts the full run detached.
# Asks for the server password once (SSH connection sharing).
#
# Override for a different machine:  GRPO_PORT=30000 GRPO_ALIAS=gridshare-xxxxxxxx bash mac_go.sh
set -uo pipefail

HOST="${GRPO_HOST:-relay.gridshare.in}"
PORT="${GRPO_PORT:-30000}"
ALIAS="${GRPO_ALIAS:-gridshare-9f838f3e}"
USER_AT="root@$HOST"
CTRL="$HOME/.ssh/grpo-ctl-%C"
SSH_OPTS=(-o "HostKeyAlias=$ALIAS" -o CheckHostIP=no -o ControlMaster=auto -o "ControlPath=$CTRL" -o ControlPersist=10m)

ZIP="$(ls -t "$HOME"/Downloads/grpo-trainer*.zip 2>/dev/null | head -1)"
[ -n "$ZIP" ] || { echo "no grpo-trainer*.zip in ~/Downloads"; exit 1; }
unzip -l "$ZIP" 2>/dev/null | grep -q "grpo-trainer/scripts/run_all.sh" \
  || { echo "$ZIP is an old version (no scripts/run_all.sh). Download the newest grpo-trainer.zip first."; exit 1; }
echo "zip:  $ZIP ($(wc -c < "$ZIP" | tr -d ' ') bytes)"
echo "host: $USER_AT port $PORT ($ALIAS)"

ARG=""
printf "Also start the FULL 300-step run now? Runs ~2 h on the RTX A4000 (~Rs 50-60 of GPU time). [y/N] "
read -r ans
case "$ans" in y|Y|yes|YES) ARG="--full" ;; *) echo "OK: update + smoke gate only." ;; esac

echo "== connecting (enter the server password once)"
ssh -p "$PORT" "${SSH_OPTS[@]}" -fN "$USER_AT" || { echo "ssh connection failed"; exit 1; }
cleanup() { ssh -p "$PORT" "${SSH_OPTS[@]}" -O exit "$USER_AT" >/dev/null 2>&1; }
trap cleanup EXIT

echo "== uploading"
scp -P "$PORT" "${SSH_OPTS[@]}" "$ZIP" "$USER_AT:/workspace/grpo-trainer.zip" || { echo "upload failed"; exit 1; }

echo "== updating code on server (runs/ kept) and running"
ssh -p "$PORT" "${SSH_OPTS[@]}" "$USER_AT" "cd /workspace && rm -rf _new \
  && python3 -m zipfile -e grpo-trainer.zip _new && mkdir -p grpo-trainer \
  && cp -a _new/grpo-trainer/. grpo-trainer/ && rm -rf _new \
  && bash grpo-trainer/scripts/run_all.sh $ARG"
rc=$?
[ $rc -eq 0 ] && echo "== done" || echo "== server step exited with code $rc (see output above)"
exit $rc
