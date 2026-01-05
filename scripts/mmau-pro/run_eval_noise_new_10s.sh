#!/bin/bash
set -o pipefail

LOG_FILE="logs/execution_eval_noise_new_10s_mmau-pro.log"
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

run_cmd "Eval Gemma3n noise new 10s" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/mmau-pro/evaluation.py results/mmau-pro/gemma3n_predictions_assets-white-noise-179828-10s.parquet --model_output_column model_response"
run_cmd "Eval Qwen2.5-Omni noise new 10s" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/mmau-pro/evaluation.py results/mmau-pro/qwen25-omni_predictions_assets-white-noise-179828-10s.parquet --model_output_column model_response"
run_cmd "Eval Qwen3-Omni-Thinking noise new 10s" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/mmau-pro/evaluation.py results/mmau-pro/qwen3-omni-thinking_predictions_assets-white-noise-179828-10s.parquet --model_output_column model_response"
run_cmd "Eval Qwen3-Omni-Instruct noise new 10s" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/mmau-pro/evaluation.py results/mmau-pro/qwen3-omni-instruction_predictions_assets-white-noise-179828-10s.parquet --model_output_column model_response"

echo
echo "Log saved to: $LOG_FILE"
