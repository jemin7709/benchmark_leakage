#!/bin/bash

gpu_num=$1

LOG_FILE="execution_infer_noise_new.log"
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

run_cmd "Qwen2.5-Omni with audio" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/inference.py --model qwen2.5-omni --batch-size 100"
run_cmd "Qwen3-Omni with audio" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/inference.py --model qwen3-omni --batch-size 100"
run_cmd "Gemma3n with audio" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/inference.py --model gemma3n --batch-size 1"

run_cmd "Qwen2.5-Omni with noise" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/inference.py --model qwen2.5-omni --noise-path ./assets/white-noise-179828-10s.mp3 --batch-size 100"
run_cmd "Qwen3-Omni with noise" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/inference.py --model qwen3-omni --noise-path ./assets/white-noise-179828-10s.mp3 --batch-size 100"
run_cmd "Gemma3n with noise" bash -c "CUDA_VISIBLE_DEVICES=$gpu_num uv run src/inference.py --model gemma3n --noise-path ./assets/white-noise-179828-10s.mp3 --batch-size 1"

echo
echo "Log saved to: $LOG_FILE"
