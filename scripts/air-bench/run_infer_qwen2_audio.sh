#!/bin/bash
set -o pipefail

LOG_FILE="logs/execution_infer_qwen2_audio.log"
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

run_cmd "Qwen2-Audio with audio" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/air-bench/foundation_infer.py --model qwen2-audio --batch-size 1 --max-per-task 1000"

echo
echo "Log saved to: $LOG_FILE"
