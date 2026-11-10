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
    task = task or get_task(cfg.task.name, spec=cfg.task.spec)
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


def _read_profile_field(path: Path, field: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                values[str(rec["id"])] = float(rec[field])
            except KeyError as e:
                if field != "pass_rate":
                    raise DataError(
                        f"profile {path} has no {field!r} (written by an older version); "
                        "re-run `grpo profile --force`"
                    ) from e
                raise DataError(f"corrupt profile {path} line {line_no}: {e}") from e
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise DataError(f"corrupt profile {path} line {line_no}: {e}") from e
    return values


def read_profile(path: Path) -> dict[str, float]:
    """Per-prompt pass rate (fraction of samples scored exactly 1.0)."""
    return _read_profile_field(path, "pass_rate")


def read_profile_std(path: Path) -> dict[str, float]:
    """Per-prompt standard deviation of the correctness score across samples."""
    return _read_profile_field(path, "score_std")


def select_learnable(
    cfg: Config,
    sample: list[Example],
    rates: dict[str, float],
    stds: dict[str, float] | None = None,
) -> list[Example]:
    p = cfg.profiler
    source = stds if p.select == "reward_variance" else rates
    if source is None:
        raise DataError("profiler.select=reward_variance needs per-prompt score_std from the profile")
    missing = [e.id for e in sample if e.id not in source]
    if missing:
        raise DataError(
            f"profile is missing {len(missing)} sampled prompts (first {missing[:3]}); re-run `grpo profile`"
        )
    if p.select == "reward_variance":
        kept = [e for e in sample if source[e.id] > p.min_score_std]
        band = f"score_std > {p.min_score_std}"
    else:
        lo, hi = p.keep_min_pass_rate, p.keep_max_pass_rate
        kept = [e for e in sample if lo < source[e.id] < hi]
        band = f"pass rate in ({lo}, {hi})"
    if not kept:
        raise DataError(
            f"no prompts in learnable band ({band}); every sampled prompt scores the same on all samples, "
            "so GRPO has no gradient. Change model, task, or selection."
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
    stds = read_profile_std(profile_path) if cfg.profiler.select == "reward_variance" else None
    kept = select_learnable(cfg, sample, read_profile(profile_path), stds)
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
