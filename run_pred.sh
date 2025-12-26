#!/bin/bash

gpu_num=$1

LOG_FILE="execution_pred.log"
> "$LOG_FILE"  # Clear log file

run_cmd() {
    local label="$1"
    shift
    
    echo "→ $label"
    if output=$("$@" 2>&1); then
        echo "[✓] $label" | tee -a "$LOG_FILE"
    else
        local exit_code=$?
        echo "[✗] $label (Exit Code: $exit_code)" | tee -a "$LOG_FILE"
        echo "Error output (last 200 lines):" | tee -a "$LOG_FILE"
        echo "$output" | tail -n 200 | tee -a "$LOG_FILE"
        echo "---" >> "$LOG_FILE"
    fi
}

run_cmd "Pred prompt half (Qwen2.5)" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/pred_prompt_half.py --model qwen2.5-omni --batch-size 100"
run_cmd "Pred prompt half (Qwen3)" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/pred_prompt_half.py --model qwen3-omni --batch-size 100"
run_cmd "Pred prompt half (Gemma3n)" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/pred_prompt_half.py --model gemma3n --batch-size 1"
run_cmd "Pred prompt incorrect (Qwen2.5)" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/pred_prompt_incorrect.py --model qwen2.5-omni --batch-size 100"
run_cmd "Pred prompt incorrect (Qwen3)" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/pred_prompt_incorrect.py --model qwen3-omni --batch-size 100"
run_cmd "Pred prompt incorrect (Gemma3n)" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/pred_prompt_incorrect.py --model gemma3n --batch-size 1"

echo
echo "Log saved to: $LOG_FILE"
