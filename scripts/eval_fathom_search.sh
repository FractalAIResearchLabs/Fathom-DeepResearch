# #!/usr/bin/env bash
# set -euo pipefail
# . "$(cd "$(dirname "$0")" && pwd)/lib.sh"
# export CUDA_VISIBLE_DEVICES=7,8
# usage(){ echo "Usage: $0 --model-path PATH --model-port PORT --executors CSV_PORTS --dataset NAME|/path/file.jsonl"; exit 1; }
# MODEL_PATH=""; MODEL_PORT=""; EXECUTOR_PORTS=""; DATASET=""
# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     --model-path) MODEL_PATH="$2"; shift 2 ;;
#     --model-port) MODEL_PORT="$2"; shift 2 ;;
#     --executors)  EXECUTOR_PORTS="$2"; shift 2 ;;
#     --dataset)    DATASET="$2"; shift 2 ;;
#     *) usage ;;
#   esac
# done
# [[ -z "${MODEL_PATH}" || -z "${MODEL_PORT}" || -z "${EXECUTOR_PORTS}" || -z "${DATASET}" ]] && usage

# load_env
# SERVED_NAME="Fathom-Search-4B"
# TOKENIZER="${MODEL_PATH}"
# IFS='|' read -r DATA_ROOT DATASET_NAME DATASET_FILE < <(resolve_dataset_arg "${DATASET}")
# OUT_BASE="${REPO_ROOT}/results"
# WORKERS="64"; JUDGE_MODEL="gpt-4.1-mini"
# CTX="40960"; DTYPE="bfloat16"; TP="2"; JSON_OVERRIDE='{}'

# start_serper "${EXECUTOR_PORTS}" 128
# EXEC_URLS="$(ports_to_urls "${EXECUTOR_PORTS}")"

# echo "▶️ sglang:${MODEL_PORT} ← ${MODEL_PATH}"
# nohup python3 -m sglang.launch_server \
#   --served-model-name "${SERVED_NAME}" \
#   --model-path "${MODEL_PATH}" \
#   --tp "${TP}" \
#   --dtype "${DTYPE}" \
#   --host 0.0.0.0 \
#   --port "${MODEL_PORT}" \
#   --trust-remote-code \
#   --context-length "${CTX}" \
#   --enable-metrics \
#   --json-model-override-args "${JSON_OVERRIDE}" \
#   > "${REPO_ROOT}/logs/sglang_${SERVED_NAME}_${MODEL_PORT}.log" 2>&1 < /dev/null &
# sleep 5
# wait_http "http://0.0.0.0:${MODEL_PORT}/health" || true

# set_pythonpath_eval
# python3 "${REPO_ROOT}/eval_search.py" \
#   --agent "recall" \
#   --dataset "${DATASET_NAME}" \
#   --data-root "${DATA_ROOT}" \
#   --tokenizer "${TOKENIZER}" \
#   --agent "fathom-search" \
#   --executors "${EXEC_URLS}" \
#   --model-url "http://0.0.0.0:${MODEL_PORT}" \
#   --out-base "${OUT_BASE}" \
#   --mode multi --workers "${WORKERS}" \
#   --search-preset fathom \
#   --judge-model "${JUDGE_MODEL}" \
#   --resume \
#   --name stag3-sft

#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"

# Usage:
# scripts/eval_fathom_search.sh \
#   --model-path /abs/path/to/Fathom-Search-4B \
#   --model-port 1255 \
#   --executors 1211,1212 \
#   --dataset GPQA-Diamond
#
# With a custom query LLM (OpenAI):
#   --query-llm gpt-4.1-mini
#
# With a local query LLM model path (must also pass --query-port):
#   --query-llm /abs/path/to/Qwen2.5-7B-Instruct --query-port 1260
#
# You can also pass a full JSONL path instead of a dataset name:
#   --dataset /abs/path/to/myset.jsonl

#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"

# Usage:
# scripts/eval_fathom_search.sh \
#   --model-path /abs/path/to/Fathom-Search-4B \
#   --model-port 1255 \
#   --executors 1211,1212 \
#   --dataset GPQA-Diamond \
#   --main-gpus 0,1 \
#   --query-llm gpt-4.1-mini
#
# Local query LLM + separate GPU:
#   --query-llm /abs/path/to/Qwen2.5-7B-Instruct --query-port 1260 --query-gpus 2
#
# Or pass a full JSONL file instead of a dataset name:
#   --dataset /abs/path/to/set.jsonl

usage() {
  echo "Usage: $0 --model-path PATH --model-port PORT --executors CSV_PORTS --dataset NAME|/path/file.jsonl [--main-gpus LIST] [--query-llm MODEL_OR_PATH] [--query-port PORT] [--query-gpus LIST]" >&2
  exit 1
}

# ── required ────────────────────────────────────────────────────────────────
MODEL_PATH=""; MODEL_PORT=""; EXECUTOR_PORTS=""; DATASET=""

# ── optional ────────────────────────────────────────────────────────────────
MAIN_GPUS="${MAIN_GPUS:-0,1}"          # GPUs for the main Fathom-Search model
QUERY_LLM="${QUERY_LLM:-gpt-4.1-mini}" # extractor for query_url(); OpenAI by default
QUERY_PORT="${QUERY_PORT:-}"           # only needed if QUERY_LLM is a local model path
QUERY_GPUS="${QUERY_GPUS:-}"           # GPUs for local query LLM (e.g., "2" or "2,3")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-port) MODEL_PORT="$2"; shift 2 ;;
    --executors)  EXECUTOR_PORTS="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    --main-gpus)  MAIN_GPUS="$2"; shift 2 ;;
    --query-llm)  QUERY_LLM="$2"; shift 2 ;;
    --query-port) QUERY_PORT="$2"; shift 2 ;;
    --query-gpus) QUERY_GPUS="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -z "${MODEL_PATH}" || -z "${MODEL_PORT}" || -z "${EXECUTOR_PORTS}" || -z "${DATASET}" ]] && usage

# Load secrets (.env): SERPER_API_KEY, OPENAI_API_KEY (if using gpt-*), JINA_API_KEY (optional)
load_env
reset_logs || true
set_pythonpath_eval

: "${SERPER_API_KEY:?Set SERPER_API_KEY in scripts/.env or env}"
if [[ "${QUERY_LLM}" == gpt-* ]]; then
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY in scripts/.env or env to use OpenAI query LLM}"
fi

SERVED_NAME="Fathom-Search-4B"
TOKENIZER="${MODEL_PATH}"
IFS='|' read -r DATA_ROOT DATASET_NAME DATASET_FILE < <(resolve_dataset_arg "${DATASET}")

REPO_LOGS="${REPO_ROOT}/logs"
OUT_BASE="${REPO_ROOT}/results"
mkdir -p "${REPO_LOGS}" "${OUT_BASE}"

WORKERS="64"
JUDGE_MODEL="gpt-4.1-mini"
CTX="40960"
DTYPE="bfloat16"
TP="2"                    # tensor-parallel for main model
JSON_OVERRIDE='{}'

# ───────────────────────────────────────────────────────────────────────────────
# 0) If QUERY_LLM is a local path, host it with its own GPU(s)
# ───────────────────────────────────────────────────────────────────────────────
if [[ -e "${QUERY_LLM}" ]]; then
  if [[ -z "${QUERY_PORT}" ]]; then
    echo "❌ --query-llm is a local path, please also provide --query-port" >&2
    exit 1
  fi
  # default to a single GPU if none specified
  if [[ -z "${QUERY_GPUS}" ]]; then
    QUERY_GPUS="0"
    echo "ℹ️  --query-gpus not provided; defaulting local query LLM to GPU ${QUERY_GPUS}"
  fi

  echo "▶️ Launching Query LLM on :${QUERY_PORT} ← ${QUERY_LLM} (GPUs: ${QUERY_GPUS})"
  QUERY_LOG="${REPO_LOGS}/sglang_query_llm_${QUERY_PORT}.log"
  CUDA_VISIBLE_DEVICES="${QUERY_GPUS}" nohup python3 -m sglang.launch_server \
    --served-model-name "Query-LLM" \
    --model-path "${QUERY_LLM}" \
    --tp 2 \
    --dtype "${DTYPE}" \
    --host 0.0.0.0 \
    --port "${QUERY_PORT}" \
    --trust-remote-code \
    --context-length 32768 \
    --enable-metrics \
    > "${QUERY_LOG}" 2>&1 < /dev/null &

  echo "Loading Query-LLM"
  sleep 120
  wait_http "http://0.0.0.0:${QUERY_PORT}/health" || true

  # Tools will detect non-gpt-* and call as a vLLM base URL
  export QUERY_LLM="http://0.0.0.0:${QUERY_PORT}"
else
  # Keep OpenAI model name as-is (e.g., gpt-4.1-mini)
  export QUERY_LLM="${QUERY_LLM}"
fi

# ───────────────────────────────────────────────────────────────────────────────
# 1) Start SERPER web executors (tool server)
# ───────────────────────────────────────────────────────────────────────────────
start_serper "${EXECUTOR_PORTS}" 128
EXEC_URLS="$(ports_to_urls "${EXECUTOR_PORTS}")"

# ───────────────────────────────────────────────────────────────────────────────
# 2) Launch main Fathom-Search-4B model on its own GPU set
# ───────────────────────────────────────────────────────────────────────────────
echo "▶️ sglang:${MODEL_PORT} ← ${MODEL_PATH} (GPUs: ${MAIN_GPUS}, TP=${TP})"
MODEL_LOG="${REPO_LOGS}/sglang_${SERVED_NAME}_${MODEL_PORT}.log"
CUDA_VISIBLE_DEVICES="${MAIN_GPUS}" nohup python3 -m sglang.launch_server \
  --served-model-name "${SERVED_NAME}" \
  --model-path "${MODEL_PATH}" \
  --tp 2 \
  --dtype "${DTYPE}" \
  --host 0.0.0.0 \
  --port "${MODEL_PORT}" \
  --trust-remote-code \
  --context-length "${CTX}" \
  --enable-metrics \
  --json-model-override-args "${JSON_OVERRIDE}" \
  > "${MODEL_LOG}" 2>&1 < /dev/null &

echo "Loading Fathom-Search-4B"
sleep 120

# ───────────────────────────────────────────────────────────────────────────────
# 3) Run evaluation
# ───────────────────────────────────────────────────────────────────────────────
python3 "${REPO_ROOT}/eval_search.py" \
  --dataset "${DATASET_NAME}" \
  --data-root "${DATA_ROOT}" \
  --tokenizer "${TOKENIZER}" \
  --agent "fathom-search" \
  --executors "${EXEC_URLS}" \
  --model-url "http://0.0.0.0:${MODEL_PORT}" \
  --out-base "${OUT_BASE}" \
  --mode multi --workers "${WORKERS}" \
  --search-preset fathom \
  --judge-model "${JUDGE_MODEL}" \
  --resume \
  --name "Fathom-Search-4B"






