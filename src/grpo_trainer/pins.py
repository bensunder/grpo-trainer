"""Exact versions this harness is verified against. Preflight refuses to run on drift.

Keep in sync with pyproject.toml [project.optional-dependencies].train and requirements.lock.
"""

from __future__ import annotations

PINNED: dict[str, str] = {
    "trl": "1.17.1",
    "vllm": "0.32.0",
    "torch": "2.13.0",
    "transformers": "5.19.0",
    "accelerate": "1.15.0",
    "datasets": "5.1.0",
    "peft": "0.21.2",
    "bitsandbytes": "0.50.2",
    "tensorboard": "2.21.0",
}

# torch 2.13.0 PyPI wheels ship CUDA 13 runtime libraries. The host driver must support CUDA 13.
REQUIRED_CUDA_MAJOR = 13
