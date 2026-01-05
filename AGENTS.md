# PROJECT KNOWLEDGE BASE

**Generated:** 2025-12-30T02:05:55Z
**Commit:** 22245eb
**Branch:** main

## OVERVIEW

Python 3.12. UV-managed ML benchmark/eval harness.
Focus: audio→text multimodal models + “noise substitution” leakage checks.

## STRUCTURE

```
benchmark_leakage/
├── src/                 # Python entrypoints + model wrappers
│   ├── mmau-pro/        # MMAU-Pro inference + evaluation
│   │   ├── inference.py # MMAU-Pro inference → parquet in ./results/
│   │   └── evaluation.py# MMAU-Pro eval (LLM judge + NVEmbed + AIF)
│   ├── model/           # Gemma/Qwen wrappers (HF + vLLM)
│   └── air-bench/       # AIR-Bench 2024 inference+scoring
├── scripts/             # Repro scripts with logging + CUDA_VISIBLE_DEVICES
├── assets/              # White-noise audio files
├── Dockerfile           # CUDA image + uv sync + .venv-eval
└── compose.yaml         # Dev container (cache volume, watch sync)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Run MMAU-Pro inference (audio/noise) | `scripts/run_infer*.sh`, `src/mmau-pro/inference.py` | `--noise-path` swaps all audio to one file |
| Run MMAU-Pro evaluation | `scripts/run_eval*.sh`, `src/mmau-pro/evaluation.py` | Uses `.venv-eval/bin/python` |
| Change model backends | `src/model/*.py` | HF + vLLM wrappers per model |

## CONVENTIONS

- **UV**: env setup via `uv sync`; runtime via `uv run ...`.
- **Dual envs**: evaluation runs from `.venv-eval/` (pinned `requirements-eval.txt`).
- **Results**: written under `./results/` (gitignored).
- **Dataset IO**:
  - MMAU-Pro inference reads HF cache directly at `~/.cache/huggingface/hub/.../test.parquet`.

## ANTI-PATTERNS (THIS PROJECT)

- **Assume CI/tests exist**: none found; validation is via scripts under `scripts/`.
- **Assume ripgrep available**: `rg` not installed in this environment.

## COMMANDS

```bash
# MMAU-Pro inference (audio)
bash scripts/run_infer.sh 0

# MMAU-Pro inference (noise variants)
bash scripts/run_infer_noise_new.sh 0
bash scripts/run_infer_noise_origin.sh 0
bash scripts/run_infer_noise_origin_long.sh 0

# MMAU-Pro eval (audio/noise)
bash scripts/run_eval.sh 0
bash scripts/run_eval_noise_new.sh 0
bash scripts/run_eval_noise_origin.sh 0
bash scripts/run_eval_noise_origin_long.sh 0

# Masked-prompt experiments
bash scripts/run_pred.sh 0
```

## NOTES

- `compose.yaml` expects `HF_TOKEN` and mounts a persistent cache volume.