#!/bin/bash
set -o pipefail

LOG_FILE="logs/execution_eval_noise_origin_3m_air-bench.log"
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

run_cmd "Eval Gemma3n noise origin 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/air-bench/foundation_scoring.py --input results/air-bench/gemma3n_predictions_foundation_assets-white-noise-358382-3m.jsonl"
run_cmd "Eval Qwen2.5-Omni noise origin 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/air-bench/foundation_scoring.py --input results/air-bench/qwen25-omni_predictions_foundation_assets-white-noise-358382-3m.jsonl"
run_cmd "Eval Qwen3-Omni-Thinking noise origin 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/air-bench/foundation_scoring.py --input results/air-bench/qwen3-omni-thinking_predictions_foundation_assets-white-noise-358382-3m.jsonl"
run_cmd "Eval Qwen3-Omni-Instruct noise origin 3m" bash -c "CUDA_VISIBLE_DEVICES=$1 .venv-eval/bin/python src/air-bench/foundation_scoring.py --input results/air-bench/qwen3-omni-instruction_predictions_foundation_assets-white-noise-358382-3m.jsonl"

echo
echo "Log saved to: $LOG_FILE"
