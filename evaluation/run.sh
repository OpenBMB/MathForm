#!/usr/bin/env bash
#
# End-to-end autoformalization evaluation pipeline.
#
#   Stage 1  Inference : generate Lean candidates with the model under evaluation
#   Stage 2  Judge     : compile-check candidates, then evaluate semantic
#                        consistency with a judge model; report Pass@k
#
# Both stages call a remote OpenAI-compatible API, so this job is CPU-only.
# Completed stages are detected from previous outputs and skipped automatically.
#
# Required environment variables:
#   EVAL_API_BASE_URL    API base URL of the model under evaluation
#   EVAL_API_MODEL       remote model name of the model under evaluation
#   JUDGE_API_BASE_URL   API base URL of the judge model
#   JUDGE_API_MODEL      remote model name of the judge model
#   API_KEY              API key (or set EVAL_API_KEY_ENV / JUDGE_API_KEY_ENV
#                        to read from different variables)
#
# The judge stage additionally requires a running Kimina Lean Server;
# set its address via LEAN_SERVER_HOST / LEAN_SERVER_PORT (default 127.0.0.1:8000).
#
# Example:
#   EVAL_API_BASE_URL=https://api.example.com/v1 EVAL_API_MODEL=my-model \
#   JUDGE_API_BASE_URL=https://api.example.com/v1 JUDGE_API_MODEL=judge-model \
#   API_KEY=sk-... bash run.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="${PY:-python3}"

# --------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# --------------------------------------------------------------------------

# Datasets and sampling
DATASET_PATHS=(${DATASET_PATHS:-
  "$SCRIPT_DIR/benchmarks/formalmath_lite.jsonl"
  "$SCRIPT_DIR/benchmarks/combibench.jsonl"
  "$SCRIPT_DIR/benchmarks/proverbench.jsonl"
  "$SCRIPT_DIR/benchmarks/fate_m.jsonl"
  "$SCRIPT_DIR/benchmarks/fate_h.jsonl"
  "$SCRIPT_DIR/benchmarks/fate_x.jsonl"
})
NUM_SAMPLES="${NUM_SAMPLES:-8}"                  # candidates per problem (k)
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-formalizer}" # inference prompt template
NO_THINK="${NO_THINK:-0}"                        # 1 = skip think/code-block parsing
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}"      # generation cap for inference
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-51200}"    # generation cap for judge

# Model under evaluation (remote API)
EVAL_API_BASE_URL="${EVAL_API_BASE_URL:-}"
EVAL_API_MODEL="${EVAL_API_MODEL:-}"
EVAL_API_KEY_ENV="${EVAL_API_KEY_ENV:-API_KEY}"
EVAL_API_MAX_WORKERS="${EVAL_API_MAX_WORKERS:-64}"

# Judge model (remote API)
JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-}"
JUDGE_API_MODEL="${JUDGE_API_MODEL:-}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-API_KEY}"
JUDGE_API_MAX_WORKERS="${JUDGE_API_MAX_WORKERS:-120}"

# The key is passed to main.py via environment variable, never as a CLI argument
export API_KEY="${API_KEY:-}"

# Output layout
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$OUTPUT_DIR/predictions.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/results.jsonl}"

mkdir -p "$OUTPUT_DIR"

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

die() {
  echo "[error] $1" >&2
  exit 1
}

[[ -n "$EVAL_API_BASE_URL" ]]      || die "EVAL_API_BASE_URL is required (API base URL of the model under evaluation)"
[[ -n "$EVAL_API_MODEL" ]]         || die "EVAL_API_MODEL is required (remote model name of the model under evaluation)"
[[ -n "${!EVAL_API_KEY_ENV:-}" ]]  || die "API key not found in environment variable ${EVAL_API_KEY_ENV}"
[[ -n "$JUDGE_API_BASE_URL" ]]     || die "JUDGE_API_BASE_URL is required (API base URL of the judge model)"
[[ -n "$JUDGE_API_MODEL" ]]        || die "JUDGE_API_MODEL is required (remote model name of the judge model)"
[[ -n "${!JUDGE_API_KEY_ENV:-}" ]] || die "API key not found in environment variable ${JUDGE_API_KEY_ENV}"

# Exit 0 when the stage output is already complete (resume support)
check_infer_complete() { "$PY" "$SCRIPT_DIR/utils.py" infer "$@"; }
check_judge_complete() { "$PY" "$SCRIPT_DIR/utils.py" judge "$@"; }

# --------------------------------------------------------------------------
# Stage 1: Inference (model under evaluation)
# --------------------------------------------------------------------------

if check_infer_complete "$PREDICTIONS_PATH" "$NUM_SAMPLES" "${DATASET_PATHS[@]}"; then
  echo "Complete inference output found, skipping inference stage"
else
  echo "Inference via remote API backend: ${EVAL_API_BASE_URL} (model=${EVAL_API_MODEL})"
  "$PY" "$SCRIPT_DIR/main.py" \
    --mode infer \
    --api_base_url "$EVAL_API_BASE_URL" \
    --api_model "$EVAL_API_MODEL" \
    --api_key_env "$EVAL_API_KEY_ENV" \
    --output_path "$PREDICTIONS_PATH" \
    --dataset_path "${DATASET_PATHS[@]}" \
    --num_samples "$NUM_SAMPLES" \
    --eval_max_tokens "$EVAL_MAX_TOKENS" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --max_workers "$EVAL_API_MAX_WORKERS" \
    $([[ "$NO_THINK" == "1" ]] && echo "--no_think")
fi

# --------------------------------------------------------------------------
# Stage 2: Judge (compilation check + judge model)
# --------------------------------------------------------------------------

if check_judge_complete "$OUTPUT_PATH" "$NUM_SAMPLES" "${DATASET_PATHS[@]}"; then
  echo "Complete judge output found, skipping judge stage"
else
  echo "Judge via remote API backend: ${JUDGE_API_BASE_URL} (model=${JUDGE_API_MODEL})"
  "$PY" "$SCRIPT_DIR/main.py" \
    --mode judge \
    --api_base_url "$JUDGE_API_BASE_URL" \
    --api_model "$JUDGE_API_MODEL" \
    --api_key_env "$JUDGE_API_KEY_ENV" \
    --output_path "$OUTPUT_PATH" \
    --dataset_path "${DATASET_PATHS[@]}" \
    --predictions_path "$PREDICTIONS_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --judge_max_tokens "$JUDGE_MAX_TOKENS" \
    --max_workers "$JUDGE_API_MAX_WORKERS"
fi
