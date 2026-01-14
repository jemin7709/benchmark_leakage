#!/bin/bash
uv sync --frozen
uv venv .venv-eval && \
    uv pip install --python .venv-eval/bin/python -r requirements-eval.txt