#!/bin/bash

LOG_FILE="execution_eval.log"
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

run_cmd "Eval Gemma3n audio" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/evaluation.py results/gemma3n_predictions_audio.parquet --model_output_column model_response"
run_cmd "Eval Gemma3n noise" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/evaluation.py results/gemma3n_predictions_noise.parquet --model_output_column model_response"
run_cmd "Eval Qwen2.5-Omni audio" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/evaluation.py results/qwen25-omni_predictions_audio.parquet --model_output_column model_response"
run_cmd "Eval Qwen2.5-Omni noise" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/evaluation.py results/qwen25-omni_predictions_noise.parquet --model_output_column model_response"
run_cmd "Eval Qwen3-Omni audio" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/evaluation.py results/qwen3-omni_predictions_audio.parquet --model_output_column model_response"
run_cmd "Eval Qwen3-Omni noise" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/evaluation.py results/qwen3-omni_predictions_noise.parquet --model_output_column model_response"

echo
echo "Log saved to: $LOG_FILE"
