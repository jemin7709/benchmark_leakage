#!/bin/bash
set -o pipefail

LOG_FILE="logs/execution_infer_mmau-pro.log"
> "$LOG_FILE"  # Clear log file

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

run_cmd "Qwen2.5-Omni with audio" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference.py --model qwen2.5-omni --batch-size 100"
run_cmd "Qwen3-Omni-Thinking with audio" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference.py --model qwen3-omni-thinking --batch-size 100"
run_cmd "Qwen3-Omni-Instruct with audio" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference.py --model qwen3-omni-instruction --batch-size 100"
run_cmd "Gemma3n with audio" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference.py --model gemma3n --batch-size 1"

echo
echo "Log saved to: $LOG_FILE"
