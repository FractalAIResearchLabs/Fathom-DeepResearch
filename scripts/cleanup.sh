#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/cleanup_ports.sh 1255 1211 1212 1213 ...
# Kills any processes listening on the given TCP ports (IPv4/IPv6).
# Tries multiple strategies and waits until the port is free.

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 PORT [PORT ...]" >&2
  exit 1
fi

# kill a PID gently then forcefully; also kill its process group
kill_tree() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0

  # Kill the process group first (negative PGID) then the PID itself
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | awk '{print $1}')" || true
  if [[ -n "${pgid:-}" ]]; then
    # TERM group
    kill -TERM -"${pgid}" 2>/dev/null || true
  fi
  kill -TERM "$pid" 2>/dev/null || true
  sleep 0.5

  # If still alive, KILL group, then PID
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -"${pgid}" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

# find listener PIDs for a given port using ss (IPv4/IPv6)
find_pids_by_port() {
  local port="$1"
  # ss prints like: LISTEN 0 4096 *:1254 users:(("python3",pid=1234,fd=7))
  ss -lptn "sport = :${port}" 2>/dev/null \
    | awk -F 'pid=' '/LISTEN/ {split($2,a,","); print a[1]}' \
    | sed 's/[^0-9]//g' \
    | awk 'NF' \
    | sort -u
}

# try to kill known command-lines that include the port number
kill_by_pattern() {
  local port="$1"
  # common patterns in your stack
  pkill -f -TERM "sglang\.launch_server.*(--port[ =]${port}|:${port})" 2>/dev/null || true
  pkill -f -TERM "sandbox_serper\.py.*(--port[ =]${port}|:${port})" 2>/dev/null || true
  pkill -f -TERM "python.*(--port[ =]${port}|:${port})" 2>/dev/null || true
  sleep 0.3
  pkill -f -KILL "sglang\.launch_server.*(--port[ =]${port}|:${port})" 2>/dev/null || true
  pkill -f -KILL "sandbox_serper\.py.*(--port[ =]${port}|:${port})" 2>/dev/null || true
  pkill -f -KILL "python.*(--port[ =]${port}|:${port})" 2>/dev/null || true
}

for PORT in "$@"; do
  echo "🔍 Checking port ${PORT}..."

  # 1) Kill by ss-reported PIDs (listener sockets)
  mapfile -t PIDS < <(find_pids_by_port "${PORT}")
  if (( ${#PIDS[@]} )); then
    echo "⚠️  Found listeners on ${PORT}: ${PIDS[*]}"
    for pid in "${PIDS[@]}"; do
      kill_tree "$pid"
    done
  fi

  # 2) Fallback: lsof and fuser (if installed)
  if command -v lsof >/dev/null 2>&1; then
    mapfile -t LPIDS < <(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
    if (( ${#LPIDS[@]} )); then
      echo "⚠️  lsof found listeners on ${PORT}: ${LPIDS[*]}"
      for pid in "${LPIDS[@]}"; do
        kill_tree "$pid"
      done
    fi
  fi

  if command -v fuser >/dev/null 2>&1; then
    # Try to kill any holders of the port
    fuser -k -TERM -n tcp "${PORT}" 2>/dev/null || true
    sleep 0.3
    fuser -k -KILL -n tcp "${PORT}" 2>/dev/null || true
  fi

  # 3) Kill by commandline pattern (helps with supervisor tree / respawns)
  kill_by_pattern "${PORT}"

  # 4) Wait until free (or give diagnostic)
  freed=0
  for i in {1..20}; do
    sleep 0.2
    if ! ss -lptn "sport = :${PORT}" | grep -q LISTEN; then
      freed=1
      break
    fi
  done

  if [[ "${freed}" -eq 1 ]]; then
    echo "✅ Port ${PORT} is free."
  else
    echo "❌ Port ${PORT} still in use."
    ss -lptn "sport = :${PORT}" || true
    echo "   (If this is a different user, you may need sudo.)"
  fi
done
