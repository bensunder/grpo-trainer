"""Difficulty profiler.

GRPO's advantage is (r - mean_group) / std_group. A prompt the base model solves 0/K or K/K times
yields std = 0 -> zero advantage -> zero gradient, but still costs a full rollout. Profiling once,
offline, with the same sampling settings as training, and keeping only prompts in the learnable band
removes that waste before any training step is paid for.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Config
from .data import load_splits, profile_cache_dir, profiling_sample
from .manifest import atomic_write_json
from .rewards.registry import build_scorer, reward_context
from .tasks import Example, dataset_fingerprint, task_for

log = logging.getLogger(__name__)


def profile_key(cfg: Config, sample: list[Example]) -> str:
    """Identity of a profile: anything that changes pass rates changes the key."""
    task = task_for(cfg)
    correctness = next(r for r in cfg.rewards if r.name == "correctness")
    payload: dict[str, Any] = {
        "model": cfg.model.name_or_path,
        "revision": cfg.model.revision,
        "dtype": cfg.model.dtype,
        "sample": dataset_fingerprint(sample),
        "k": cfg.profiler.samples_per_prompt,
        "temperature": cfg.grpo.temperature,
        "top_p": cfg.grpo.top_p,
        "max_new_tokens": cfg.grpo.max_completion_length,
        "system_prompt": task.system_prompt(cfg.task.system_prompt),
        "correctness": correctness.params,
        "seed": cfg.run.seed,
    }
    grading = task.grading_fingerprint()
    if grading is not None:  # only spec-driven tasks; keeps existing GSM8K cache keys unchanged
        payload["grading"] = grading
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def profile_path_for(cfg: Config, sample: list[Example]) -> Path:
    return profile_cache_dir(cfg) / f"{profile_key(cfg, sample)}.jsonl"


def summarize(rates: dict[str, float], k: int, lo: float, hi: float) -> dict[str, Any]:
    n = len(rates)
    hist = Counter(round(r * k) for r in rates.values())
    kept = sum(1 for r in rates.values() if lo < r < hi)
    return {
        "prompts": n,
        "samples_per_prompt": k,
        "always_solved": hist.get(k, 0),
        "never_solved": hist.get(0, 0),
        "kept": kept,
        "kept_fraction": kept / n if n else 0.0,
        "histogram_correct_of_k": {str(i): hist.get(i, 0) for i in range(k + 1)},
        "mean_pass_rate": sum(rates.values()) / n if n else 0.0,
    }


def run_profile(cfg: Config, *, force: bool = False) -> Path:
    from .sampling import generate, vllm_engine

    splits = load_splits(cfg)
    sample = profiling_sample(cfg, splits.train)
    out = profile_path_for(cfg, sample)
    if out.is_file() and not force:
        log.info("profile cache hit: %s (use --force to recompute)", out)
        return out

    task = task_for(cfg)
    system_prompt = task.system_prompt(cfg.task.system_prompt)
    correctness_spec = next(r for r in cfg.rewards if r.name == "correctness")
    scorer = build_scorer(correctness_spec, reward_context(cfg, task))
    k = cfg.profiler.samples_per_prompt

    log.info("profiling %d prompts x %d samples with %s", len(sample), k, cfg.model.name_or_path)
    with vllm_engine(
        cfg,
        cfg.model.name_or_path,
        gpu_memory_utilization=cfg.profiler.gpu_memory_utilization,
        max_new_tokens=cfg.grpo.max_completion_length,
    ) as llm:
        results = generate(
            llm,
            [e.messages(system_prompt) for e in sample],
            n=k,
            temperature=cfg.grpo.temperature,
            top_p=cfg.grpo.top_p,
            max_new_tokens=cfg.grpo.max_completion_length,
            seed=cfg.run.seed,
        )

    rates: dict[str, float] = {}
    stds: dict[str, float] = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for ex, samples in zip(sample, results, strict=True):
            row = {"ground_truth": ex.ground_truth}
            scores = [float(scorer(s.text, completion_ids=s.token_ids, row=row)) for s in samples]
            correct = sum(sc >= 1.0 for sc in scores)
            rate = correct / len(samples)
            rates[ex.id] = rate
            mean = sum(scores) / len(scores)
            std = (sum((sc - mean) ** 2 for sc in scores) / len(scores)) ** 0.5
            stds[ex.id] = std
            rec = {
                "id": ex.id,
                "pass_rate": rate,
                "correct": correct,
                "k": len(samples),
                "score_mean": round(mean, 6),
                "score_std": round(std, 6),
            }
            f.write(json.dumps(rec) + "\n")
    tmp.replace(out)

    summary = summarize(rates, k, cfg.profiler.keep_min_pass_rate, cfg.profiler.keep_max_pass_rate)
    summary["kept_by_score_variance"] = sum(1 for v in stds.values() if v > cfg.profiler.min_score_std)
    summary["selection"] = cfg.profiler.select
    if cfg.profiler.select == "reward_variance":
        summary["kept"] = summary["kept_by_score_variance"]
        summary["kept_fraction"] = summary["kept"] / summary["prompts"] if summary["prompts"] else 0.0
    atomic_write_json(
        out.with_suffix(".summary.json"), {"key": out.stem, "model": cfg.model.name_or_path, **summary}
    )
    log.info(
        "profile: %d prompts, always=%d never=%d kept=%d (%.0f%%) -> %s",
        summary["prompts"],
        summary["always_solved"],
        summary["never_solved"],
        summary["kept"],
        100 * summary["kept_fraction"],
        out,
    )
    return out
