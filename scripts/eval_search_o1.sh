#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"


usage(){ echo "Usage: $0 --model-path PATH --model-port PORT --dataset NAME|/path/file.jsonl"; exit 1; }
MODEL_PATH=""; MODEL_PORT=""; DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-port) MODEL_PORT="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -z "${MODEL_PATH}" || -z "${MODEL_PORT}" || -z "${DATASET}" ]] && usage

load_env
reset_logs
SERVED_NAME="Search-o1"
IFS='|' read -r DATA_ROOT DATASET_NAME DATASET_FILE < <(resolve_dataset_arg "${DATASET}")
OUT_BASE="${REPO_ROOT}/results"
WORKERS="64"; JUDGE_MODEL="gpt-4.1-mini"
CTX="40960"; DTYPE="bfloat16"; TP="2"

echo "▶️ sglang:${MODEL_PORT} ← ${MODEL_PATH}"
nohup python3 -m sglang.launch_server \
  --served-model-name "Qwen3-4B" \
  --model-path "${MODEL_PATH}" \
  --tp "${TP}" \
  --enable-metrics \
  --dtype "${DTYPE}" \
  --host 0.0.0.0 \
  --port "${MODEL_PORT}" \
  --trust-remote-code \
  --disable-radix-cache \
  --disable-cuda-graph \
  --context-length "${CTX}" \
  > "${REPO_ROOT}/logs/sglang_${SERVED_NAME}_${MODEL_PORT}.log" 2>&1 < /dev/null &
echo "Loading Search-o1 model..."  
sleep 120
wait_http "http://0.0.0.0:${MODEL_PORT}/health" || true

set_pythonpath_eval
python3 "${REPO_ROOT}/eval_search.py" \
  --agent "search-o1" \
  --dataset "${DATASET_NAME}" \
  --data-root "${DATA_ROOT}" \
  --agent "search-o1" \
  --model-url "http://0.0.0.0:${MODEL_PORT}" \
  --out-base "${OUT_BASE}" \
  --mode multi --workers "${WORKERS}" \
  --judge-model "${JUDGE_MODEL}" \
  --tokenizer "${MODEL_PATH}"
#   --resume
# scripts/eval_search_o1.sh --model-path /data/home/fractal/shreyas/models/Qwen3-4B --model-port 1255  --dataset upsc_2025
