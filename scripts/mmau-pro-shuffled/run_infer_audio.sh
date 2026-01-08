#!/bin/bash
set -o pipefail

LOG_FILE="logs/execution_infer_audio_shuffled_mmau-pro.log"
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

for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_cmd "Qwen2.5-Omni with audio (seed=$seed)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference_shuffled.py --model qwen2.5-omni --batch-size 100 --shuffle-seed $seed"
    run_cmd "Qwen3-Omni-Thinking with audio (seed=$seed)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference_shuffled.py --model qwen3-omni-thinking --batch-size 100 --shuffle-seed $seed"
    run_cmd "Qwen3-Omni-Instruct with audio (seed=$seed)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference_shuffled.py --model qwen3-omni-instruction --batch-size 100 --shuffle-seed $seed"
    run_cmd "Gemma3n with audio (seed=$seed)" bash -c "CUDA_VISIBLE_DEVICES=$1 uv run src/mmau-pro/inference_shuffled.py --model gemma3n --batch-size 1 --shuffle-seed $seed"
done

echo
echo "Log saved to: $LOG_FILE"
