#!/usr/bin/env bash
# (Re)starts the decision server from a virtualenv, detached from the shell.
#   FGC_MODELS=gliner-decide,gliner-decide-1b CUDA_VISIBLE_DEVICES=0 ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  kill "$(cat server.pid)"
  while kill -0 "$(cat server.pid)" 2>/dev/null; do sleep 1; done
fi
. .venv/bin/activate
HF_HUB_DISABLE_PROGRESS_BARS=1 setsid nohup python app.py > server.log 2>&1 < /dev/null &
echo $! > server.pid
echo "started pid $(cat server.pid); log: $(pwd)/server.log"
