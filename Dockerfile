# syntax=docker/dockerfile:1.7
# GRPO trainer image. torch 2.13 / vLLM 0.32 wheels bundle the CUDA 13 runtime; the devel base supplies
# nvcc for kernels that JIT-compile at first use (e.g. FlashInfer). Host needs an NVIDIA driver that
# supports CUDA 13 and the NVIDIA Container Toolkit.
ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    VENV=/opt/venv

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev git ca-certificates procps \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv "$VENV"
ENV PATH="$VENV/bin:$PATH"

WORKDIR /app

# Dependency layer: exact lock, cached independently of source changes.
COPY requirements.lock .
RUN pip install --upgrade pip==26.3 \
 && pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e . \
 && python -c "import grpo_trainer, trl, vllm, torch; print('trl', trl.__version__, 'vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"

COPY configs ./configs
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

# Non-root runtime user; runs/ and the HF cache are expected to be mounted volumes.
RUN useradd --create-home --uid 1000 trainer \
 && mkdir -p /app/runs /cache/huggingface \
 && chown -R trainer:trainer /app /cache
USER trainer

VOLUME ["/app/runs", "/cache/huggingface"]
ENTRYPOINT ["grpo"]
CMD ["--help"]
