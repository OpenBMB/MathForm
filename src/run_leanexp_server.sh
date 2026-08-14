#!/usr/bin/env bash
#
# Start the Lean Explore search server (retrieval backend for the pipeline).
#
# Required environment variables:
#   LEAN_EXPLORE_SERVER   path to lean-explore's scripts/search_server.py
#
# Optional:
#   PY                    python interpreter (default: python3)
#   PORT                  listen port (default: 9000)
#   OUTPUT_DIR            log directory root (default: ./output)
#
# Usage:
#   LEAN_EXPLORE_SERVER=/path/to/lean-explore/scripts/search_server.py \
#   bash run_leanexp_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PY="${PY:-python3}"
PORT="${PORT:-9000}"
LEAN_EXPLORE_SERVER="${LEAN_EXPLORE_SERVER:-}"
SESSION_NAME="lean_explore"

if [[ -z "${LEAN_EXPLORE_SERVER}" ]]; then
  echo "[error] LEAN_EXPLORE_SERVER is required (path to lean-explore's scripts/search_server.py)" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-64}"
export TOKENIZERS_PARALLELISM=false

tmux has-session -t "${SESSION_NAME}" 2>/dev/null || \
  tmux new-session -d -s "${SESSION_NAME}" \
    "\"${PY}\" \"${LEAN_EXPLORE_SERVER}\" \
    --host 0.0.0.0 --port ${PORT} \
    --workers 2 --max-threads 8 --queue-timeout 0.1 \
    --cache-size 2000 --cache-ttl 120 --prewarm \
    --batch-max-size 256 --batch-wait-ms 10 --batch-timeout 30 \
    > \"${LOG_DIR}/lean_explore.log\" 2>&1"

echo "Lean Explore server started in tmux session '${SESSION_NAME}' on port ${PORT}"
echo "Log: ${LOG_DIR}/lean_explore.log"
