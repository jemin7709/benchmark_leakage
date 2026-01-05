#!/bin/bash
set -o pipefail

LOG_FILE="logs/execution_infer_noise_new_3m_air-bench.log"
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

run_cmd "Qwen2.5-Omni with noise new 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/air-bench/foundation_infer.py --model qwen2.5-omni --noise-path ./assets/white-noise-179828-3m.mp3 --batch-size 100 --max-per-task 1000"
run_cmd "Qwen3-Omni-Thinking with noise new 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/air-bench/foundation_infer.py --model qwen3-omni-thinking --noise-path ./assets/white-noise-179828-3m.mp3 --batch-size 100 --max-per-task 1000"
run_cmd "Qwen3-Omni-Instruct with noise new 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/air-bench/foundation_infer.py --model qwen3-omni-instruction --noise-path ./assets/white-noise-179828-3m.mp3 --batch-size 100 --max-per-task 1000"
run_cmd "Gemma3n with noise new 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/air-bench/foundation_infer.py --model gemma3n --noise-path ./assets/white-noise-179828-3m.mp3 --batch-size 100 --max-per-task 1000"

echo
echo "Log saved to: $LOG_FILE"
