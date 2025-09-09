#!/usr/bin/env bash
set -euo pipefail

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── PYTHONPATH presets ────────────────────────────────────────────────────────
# Eval harness should see the repo root only.
EVAL_PYTHONPATH="${REPO_ROOT}"
set_pythonpath_eval() { export PYTHONPATH="${EVAL_PYTHONPATH}"; }

# SERPER executors must see web_agents (and serving).
WEBAGENTS_PYTHONPATH="${REPO_ROOT}/serving/web_agents:${REPO_ROOT}/serving:${REPO_ROOT}"

# ── Env loader (reads scripts/.env by default) ───────────────────────────────
load_env() {
  local env_file="${1:-${REPO_ROOT}/scripts/.env}"
  if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    source "${env_file}"
  fi
}

# ── Health check helper ──────────────────────────────────────────────────────
wait_http() {
  local url="$1"; local tries="${2:-60}"; local delay="${3:-1}"
  for ((i=1; i<=tries; i++)); do
    if curl -fsS -m 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay}"
  done
  echo "❌ Timeout waiting for ${url}" >&2
  return 1
}

# ── Start SERPER web executors on CSV ports (1211,1212,...) ──────────────────
start_serper() {
  local ports_csv="$1"; local workers="${2:-128}"
  IFS=',' read -r -a P_ARR <<< "${ports_csv}"
  mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/scripts/.cache"

  : "${JINA_API_KEY:?Set JINA_API_KEY in scripts/.env}"
  : "${SERPER_API_KEY:?Set SERPER_API_KEY in scripts/.env}"
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY in scripts/.env}"
  : "${MAX_OUTBOUND:=256}"
  : "${JINA_CACHE_DIR:=${REPO_ROOT}/scripts/.cache/jina_cache}"
  : "${SERPER_CACHE_DIR:=${REPO_ROOT}/scripts/.cache/serper_cache}"
  export JINA_API_KEY SERPER_API_KEY OPENAI_API_KEY MAX_OUTBOUND JINA_CACHE_DIR SERPER_CACHE_DIR

  for PORT in "${P_ARR[@]}"; do
    local log="${REPO_ROOT}/logs/serper_${PORT}.log"
    echo "▶️ SERPER :${PORT} (workers=${workers}) → ${log}"
    # IMPORTANT: give web-agents PYTHONPATH only to this child process
    nohup env PYTHONPATH="${WEBAGENTS_PYTHONPATH}" \
      python3 "${REPO_ROOT}/serving/sandbox_serper.py" \
      --port "${PORT}" --workers "${workers}" \
      > "${log}" 2>&1 < /dev/null &
    sleep 0.3
  done
}

# ── Ports → URLs helper ──────────────────────────────────────────────────────
ports_to_urls() {
  local ports_csv="$1"; local urls=""
  IFS=',' read -r -a P_ARR <<< "${ports_csv}"
  for P in "${P_ARR[@]}"; do
    local u="http://0.0.0.0:${P}"
    urls="${urls:+${urls},}${u}"
  done
  echo "${urls}"
}

# ── Auto-locate eval_datasets ────────────────────────────────────────────────
resolve_data_root() {
  local c1="${REPO_ROOT}/eval_datasets"
  local c2="${REPO_ROOT}/data/eval_datasets"
  local env_root="${DATA_ROOT:-}"

  if [[ -n "${env_root}" && -d "${env_root}" ]]; then echo "${env_root}"; return 0; fi
  if [[ -d "${c1}" ]]; then echo "${c1}"; return 0; fi
  if [[ -d "${c2}" ]]; then echo "${c2}"; return 0; fi

  echo "❌ Could not locate eval_datasets. Expected ${c1} or ${c2}. You can set DATA_ROOT in scripts/.env" >&2
  return 1
}

# ── Accept dataset *name* or full *.jsonl path ───────────────────────────────
# echoes: "<DATA_ROOT>|<DATASET_NAME>|<DATASET_FILE>"
resolve_dataset_arg() {
  local arg="$1"
  if [[ -f "${arg}" ]]; then
    local file="$(cd "$(dirname "${arg}")" && pwd)/$(basename "${arg}")"
    local dir="$(dirname "${file}")"
    local stem="$(basename "${file}")"; stem="${stem%.jsonl}"
    echo "${dir}|${stem}|${file}"; return 0
  fi
  local root; root="$(resolve_data_root)" || return 1
  local file="${root}/${arg}.jsonl"
  if [[ ! -f "${file}" ]]; then
    echo "❌ Dataset not found: ${file}" >&2
    return 1
  fi
  echo "${root}|${arg}|${file}"
}
# ── Fresh logs each run ───────────────────────────────────────────────────────
reset_logs() {
  local dir="${REPO_ROOT}/logs"
  mkdir -p -- "${dir}"
  rm -rf -- "${dir:?}"/*
}

