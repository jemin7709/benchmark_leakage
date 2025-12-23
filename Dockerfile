FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    libnvtoolsext1 cuda-nvtx-12-9 && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

COPY pyproject.toml uv.lock .python-version requirements-eval.txt ./
RUN uv sync --frozen
RUN uv venv .venv-eval && \
    uv pip install --python .venv-eval/bin/python -r requirements-eval.txt

COPY . .

CMD ["bash", "-c"]