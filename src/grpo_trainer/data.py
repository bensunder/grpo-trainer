"""Assembles the exact example sets each stage consumes. Single source of truth for selection logic."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .errors import DataError
from .tasks import Example, Task, dataset_fingerprint, drop_contaminated, get_task

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Splits:
    train: list[Example]
    held_out: list[Example]
    contaminated_dropped: int


def load_splits(cfg: Config, task: Task | None = None) -> Splits:
    task = task or get_task(cfg.task.name)
    held_out = task.load(cfg.task.eval_split, None)
    train_all = task.load(cfg.task.train_split, None)
    train, dropped = drop_contaminated(train_all, held_out)
    if dropped:
        log.warning("dropped %d train examples that also appear in %s", dropped, cfg.task.eval_split)
    if cfg.task.train_limit is not None:
        train = train[: cfg.task.train_limit]
    return Splits(train=train, held_out=held_out, contaminated_dropped=dropped)


def profiling_sample(cfg: Config, train: list[Example]) -> list[Example]:
    """Deterministic subset of train to profile (seeded shuffle, then cap)."""
    order = list(train)
    random.Random(cfg.run.seed).shuffle(order)
    if cfg.profiler.max_prompts is not None:
        order = order[: cfg.profiler.max_prompts]
    return order


def eval_examples(cfg: Config, held_out: list[Example]) -> list[Example]:
    limit = cfg.eval.limit or cfg.task.eval_limit
    return held_out[:limit] if limit else held_out


# ------------------------------------------------------------------ profile artefact


def profile_cache_dir(cfg: Config) -> Path:
    return Path(cfg.run.output_root) / "_cache" / "profiles"


def read_profile(path: Path) -> dict[str, float]:
    rates: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rates[str(rec["id"])] = float(rec["pass_rate"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise DataError(f"corrupt profile {path} line {line_no}: {e}") from e
    return rates


def select_learnable(cfg: Config, sample: list[Example], rates: dict[str, float]) -> list[Example]:
    lo, hi = cfg.profiler.keep_min_pass_rate, cfg.profiler.keep_max_pass_rate
    missing = [e.id for e in sample if e.id not in rates]
    if missing:
        raise DataError(
            f"profile is missing {len(missing)} sampled prompts (first {missing[:3]}); re-run `grpo profile`"
        )
    kept = [e for e in sample if lo < rates[e.id] < hi]
    if not kept:
        raise DataError(
            f"no prompts in learnable band ({lo}, {hi}); the base model always/never solves this set. "
            "Change model, task, or band."
        )
    return kept


def training_examples(cfg: Config, splits: Splits, profile_path: Path | None) -> list[Example]:
    if not cfg.profiler.enabled:
        return splits.train
    if profile_path is None or not profile_path.is_file():
        raise DataError(
            "profiler.enabled is true but no profile exists for this config; run `grpo profile` first"
        )
    sample = profiling_sample(cfg, splits.train)
    kept = select_learnable(cfg, sample, read_profile(profile_path))
    needed = cfg.prompts_per_step * cfg.grpo.max_steps
    log.info(
        "training on %d/%d profiled prompts (%.0f%%); %d prompt-visits planned (~%.1f epochs)",
        len(kept),
        len(sample),
        100 * len(kept) / len(sample),
        needed,
        needed / len(kept),
    )
    return kept


def fingerprint(examples: list[Example]) -> str:
    return dataset_fingerprint(examples)
