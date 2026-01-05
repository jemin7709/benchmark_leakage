#!/bin/bash
set -o pipefail

LOG_FILE="logs/execution_eval_qwen2_audio.log"
mkdir -p logs
> "$LOG_FILE"

run_cmd() {
    local label="$1"
    shift
    
    local tmp_output=$(mktemp)
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
    fi
    
    rm -f "$tmp_output"
    trap - EXIT INT TERM
}

run_cmd "Eval Qwen2-Audio audio" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/air-bench/foundation_scoring.py --input results/air-bench/qwen2-audio_predictions_foundation_audio.jsonl"

echo
echo "Log saved to: $LOG_FILE"
