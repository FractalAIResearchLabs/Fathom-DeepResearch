#!/usr/bin/env bash
# Usage: serving/host_serper.sh <PORT> <WORKERS> <BACKEND>
# Example: serving/host_serper.sh 1211 128 gpt-4.1-mini
# Runs in the foreground (no logs, no PID files). Stop with Ctrl+C.

set -euo pipefail

PORT=${1:-1211}
WORKERS=${2:-128}
BACKEND=${3:-"openai:gpt-4.1-mini"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Optional .env
ENV_FILE="$REPO_ROOT/scripts/.env"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

# Env & paths
export PYTHONPATH="$REPO_ROOT/serving/web_agents"
export MAX_OUTBOUND="${MAX_OUTBOUND:-256}"
export JINA_CACHE_DIR="$REPO_ROOT/.cache/jina_cache"
export SERPER_CACHE_DIR="$REPO_ROOT/.cache/serper_cache"
export QUERY_LLM="$BACKEND"
export SUMMARY_LLM="${SUMMARY_LLM:-$BACKEND}"

mkdir -p "$JINA_CACHE_DIR" "$SERPER_CACHE_DIR"

echo "[host_serper] starting :$PORT | workers=$WORKERS | backend=$BACKEND"
echo "[host_serper] press Ctrl+C to stop."

# IMPORTANT: run in foreground (no setsid, no background, no logging redirection)
exec python3 "$SCRIPT_DIR/sandbox_serper.py" \
  --port "$PORT" \
  --workers "$WORKERS"
