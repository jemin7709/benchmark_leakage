#!/bin/bash
set -o pipefail
set -euo pipefail

GPU_ID="${1:-0}"
NUM_RUNS="${2:-20}"
SEED="${3:-0}"
LIMIT="${4:-0}"

mkdir -p logs
LOG_FILE="logs/execution_random_baseline_air-bench.log"
> "$LOG_FILE"

run_cmd() {
    local label="$1"
    shift

    local tmp_output
    tmp_output=$(mktemp)
    trap "rm -f '$tmp_output'" EXIT INT TERM

    echo "→ $label"
    if "$@" 2>&1 | tee "$tmp_output"; then
        echo "[✓] $label" | tee -a "$LOG_FILE"
    else
        local exit_code=$?
        echo "[✗] $label (Exit Code: $exit_code)" | tee -a "$LOG_FILE"
        echo "Error output (last 200 lines):" >> "$LOG_FILE"
        tail -n 200 "$tmp_output" >> "$LOG_FILE"
        echo "---" >> "$LOG_FILE"
        exit "$exit_code"
    fi

    rm -f "$tmp_output"
    trap - EXIT INT TERM
}

CMD=(uv run src/air-bench/random_baseline.py --num-runs "$NUM_RUNS" --seed "$SEED")
if [[ "$LIMIT" != "0" ]]; then
    CMD+=(--limit "$LIMIT")
fi

run_cmd "AIR-Bench random baseline (runs=$NUM_RUNS seed=$SEED limit=$LIMIT)" bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID ${CMD[*]}"

echo
echo "Log saved to: $LOG_FILE"
