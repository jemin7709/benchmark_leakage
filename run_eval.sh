#!/bin/bash

gpu_num=$1

LOG_FILE="execution_eval.log"
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

run_cmd "Eval Gemma3n audio" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num .venv-eval/bin/python src/evaluation.py results/gemma3n_predictions_audio.parquet --model_output_column model_response"
run_cmd "Eval Gemma3n noise" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num .venv-eval/bin/python src/evaluation.py results/gemma3n_predictions_noise.parquet --model_output_column model_response"
run_cmd "Eval Qwen2.5-Omni audio" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num .venv-eval/bin/python src/evaluation.py results/qwen25-omni_predictions_audio.parquet --model_output_column model_response"
run_cmd "Eval Qwen2.5-Omni noise" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num .venv-eval/bin/python src/evaluation.py results/qwen25-omni_predictions_noise.parquet --model_output_column model_response"
run_cmd "Eval Qwen3-Omni audio" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num .venv-eval/bin/python src/evaluation.py results/qwen3-omni_predictions_audio.parquet --model_output_column model_response"
run_cmd "Eval Qwen3-Omni noise" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num .venv-eval/bin/python src/evaluation.py results/qwen3-omni_predictions_noise.parquet --model_output_column model_response"

echo
echo "Log saved to: $LOG_FILE"
