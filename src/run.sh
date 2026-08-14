#!/usr/bin/env bash
#
# Data construction pipeline.
# All models (generation / retrieval planner / judge) are called through remote OpenAI-compatible APIs
#
# Prerequisites (must be running, addresses passed via env vars):
#   - Kimina Lean Server        (compilation check, LEAN_SERVER_URL)
#   - Lean Explore search server (retrieval, LEAN_EXPLORE_URL, optional)
#
# Required environment variables:
#   API_URL                   OpenAI-compatible chat/completions endpoint
#   API_KEY                   API key
#   BASE_MODEL_NAME           model for Lean code generation
#   JUDGE_MODEL_NAME          model for semantic consistency check
#
# Optional:
#   RETRIEVAL_PLANNER_NAME    model for retrieval query planning (defaults to BASE_MODEL_NAME)
#
# Usage:
#   API_URL=https://api.example.com/v1/chat/completions API_KEY=sk-... \
#   BASE_MODEL_NAME=gen-model JUDGE_MODEL_NAME=judge-model \
#   bash run.sh <input_jsonl> [output_dir]

set -euo pipefail

INPUT_FILE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/output}"

if [[ -z "${INPUT_FILE}" ]]; then
  echo "Usage: $0 <input_jsonl> [output_dir]" >&2
  exit 1
fi

PY="${PY:-python3}"

# Remote API backend
API_URL="${API_URL:-}"
API_KEY="${API_KEY:-}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-}"
RETRIEVAL_PLANNER_NAME="${RETRIEVAL_PLANNER_NAME:-$BASE_MODEL_NAME}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-}"

# External services (assumed running)
LEAN_SERVER_URL="${LEAN_SERVER_URL:-http://localhost:8000}"
LEAN_EXPLORE_URL="${LEAN_EXPLORE_URL:-}"

# Concurrency and iteration settings
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
MAX_WORKERS="${MAX_WORKERS:-80}"
RETRIEVAL_WORKERS="${RETRIEVAL_WORKERS:-4}"
PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-5000}"
RETRIEVAL_LIMIT="${RETRIEVAL_LIMIT:-15}"
QUERY_TOP_K="${QUERY_TOP_K:-2}"

die() {
  echo "[error] $1" >&2
  exit 1
}

[[ -n "$API_URL" ]]          || die "API_URL is required (OpenAI-compatible chat/completions endpoint)"
[[ -n "$API_KEY" ]]          || die "API_KEY is required"
[[ -n "$BASE_MODEL_NAME" ]]  || die "BASE_MODEL_NAME is required"
[[ -n "$JUDGE_MODEL_NAME" ]] || die "JUDGE_MODEL_NAME is required"

mkdir -p "${OUTPUT_DIR}"

"$PY" "${SCRIPT_DIR}/main.py" \
  --input "${INPUT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-model-url "${API_URL}" \
  --retrieval-planner-url "${API_URL}" \
  --judge-model-url "${API_URL}" \
  --lean-server-url "${LEAN_SERVER_URL}" \
  ${LEAN_EXPLORE_URL:+--lean-explore-url "${LEAN_EXPLORE_URL}"} \
  --max-iterations "${MAX_ITERATIONS}" \
  --pipeline-batch-size "${PIPELINE_BATCH_SIZE}" \
  --retrieval-limit "${RETRIEVAL_LIMIT}" \
  --query-top-k "${QUERY_TOP_K}" \
  --max-workers "${MAX_WORKERS}" \
  --base-model-workers "${MAX_WORKERS}" \
  --retrieval-planner-workers "${MAX_WORKERS}" \
  --judge-model-workers "${MAX_WORKERS}" \
  --retrieval-workers "${RETRIEVAL_WORKERS}" \
  --api-key "${API_KEY}" \
  --base-model-name "${BASE_MODEL_NAME}" \
  --retrieval-planner-name "${RETRIEVAL_PLANNER_NAME}" \
  --judge-model-name "${JUDGE_MODEL_NAME}" \
  --batch-mode
