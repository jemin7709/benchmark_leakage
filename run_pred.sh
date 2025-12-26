#!/bin/bash

LOG_FILE="execution_pred.log"
> "$LOG_FILE"  # Clear log file

run_cmd() {
    local label="$1"
    shift
    
    local tmp_output=$(mktemp)
    trap "rm -f '$tmp_output'" EXIT INT TERM
    
    echo "→ $label"
    if "$@" 2>&1 | tee "$tmp_output"; then
        echo "[✓] $label"
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

run_cmd "Pred prompt half (Qwen2.5)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/pred_prompt_half.py --model qwen2.5-omni --batch-size 100"
run_cmd "Pred prompt half (Qwen3)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/pred_prompt_half.py --model qwen3-omni --batch-size 100"
run_cmd "Pred prompt half (Gemma3n)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/pred_prompt_half.py --model gemma3n --batch-size 1"
run_cmd "Pred prompt incorrect (Qwen2.5)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/pred_prompt_incorrect.py --model qwen2.5-omni --batch-size 100"
run_cmd "Pred prompt incorrect (Qwen3)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/pred_prompt_incorrect.py --model qwen3-omni --batch-size 100"
run_cmd "Pred prompt incorrect (Gemma3n)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/pred_prompt_incorrect.py --model gemma3n --batch-size 1"

echo
echo "Log saved to: $LOG_FILE"
