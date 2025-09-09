#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"

usage(){ echo "Usage: $0 --model-path PATH --model-port PORT --executors CSV_PORTS --dataset NAME|/path/file.jsonl"; exit 1; }
MODEL_PATH=""; MODEL_PORT=""; EXECUTOR_PORTS=""; DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-port) MODEL_PORT="$2"; shift 2 ;;
    --executors)  EXECUTOR_PORTS="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -z "${MODEL_PATH}" || -z "${MODEL_PORT}" || -z "${EXECUTOR_PORTS}" || -z "${DATASET}" ]] && usage

load_env
reset_logs
SERVED_NAME="II-Search-4B"
TOKENIZER="${MODEL_PATH}"
IFS='|' read -r DATA_ROOT DATASET_NAME DATASET_FILE < <(resolve_dataset_arg "${DATASET}")
OUT_BASE="${REPO_ROOT}/results"
WORKERS="64"; JUDGE_MODEL="gpt-4.1-mini"
CTX="131072"; DTYPE="bfloat16"; TP="2"
JSON_OVERRIDE='{"rope_scaling":{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":98304}}'

start_serper "${EXECUTOR_PORTS}" 128
EXEC_URLS="$(ports_to_urls "${EXECUTOR_PORTS}")"

echo "▶️ sglang:${MODEL_PORT} ← ${MODEL_PATH}"
nohup python3 -m sglang.launch_server \
  --served-model-name "${SERVED_NAME}" \
  --model-path "${MODEL_PATH}" \
  --tp "${TP}" \
  --dtype "${DTYPE}" \
  --host 0.0.0.0 \
  --port "${MODEL_PORT}" \
  --trust-remote-code \
  --context-length "${CTX}" \
  --enable-metrics \
  --json-model-override-args "${JSON_OVERRIDE}" \
  > "${REPO_ROOT}/logs/sglang_${SERVED_NAME}_${MODEL_PORT}.log" 2>&1 < /dev/null &
echo "Loading II-Search-4B"
sleep 120
wait_http "http://0.0.0.0:${MODEL_PORT}/health" || true

set_pythonpath_eval
python3 "${REPO_ROOT}/eval_search.py" \
  --agent "recall" \
  --dataset "${DATASET_NAME}" \
  --data-root "${DATA_ROOT}" \
  --tokenizer "${TOKENIZER}" \
  --agent "ii-search" \
  --executors "${EXEC_URLS}" \
  --model-url "http://0.0.0.0:${MODEL_PORT}" \
  --out-base "${OUT_BASE}" \
  --mode multi --workers "${WORKERS}" \
  --search-preset legacy \
  --judge-model "${JUDGE_MODEL}" \
#   --resume


# scripts/eval_ii_search.sh --model-path /data/home/fractal/shreyas/models/benchmarks/II-Search-4B --model-port 1255 --executors 1211,1212 --dataset GPQA-Diamond
