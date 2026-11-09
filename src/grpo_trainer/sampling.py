"""Thin, offline vLLM wrapper shared by the profiler and eval (separate processes from training)."""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@contextmanager
def vllm_engine(
    cfg: Config, model: str, *, gpu_memory_utilization: float, max_new_tokens: int
) -> Iterator[object]:
    from vllm import LLM  # heavy import

    max_model_len = cfg.vllm.max_model_length or (max_new_tokens + 1024)
    llm = LLM(
        model=model,
        revision=cfg.model.revision if model == cfg.model.name_or_path else None,
        dtype=cfg.model.dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=cfg.run.seed,
        trust_remote_code=cfg.model.trust_remote_code,
        enable_prefix_caching=True,  # the system prompt is shared by every request
    )
    try:
        yield llm
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


def generate(
    llm: object,
    conversations: list[list[dict[str, str]]],
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> list[list[Sample]]:
    """Return `n` samples per conversation, in input order."""
    from vllm import SamplingParams

    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    outputs = llm.chat(conversations, params, use_tqdm=True)  # type: ignore[attr-defined]
    if len(outputs) != len(conversations):
        raise RuntimeError(f"vLLM returned {len(outputs)} results for {len(conversations)} prompts")
    return [
        [
            Sample(text=o.text, token_ids=tuple(o.token_ids), finish_reason=o.finish_reason)
            for o in req.outputs
        ]
        for req in outputs
    ]
