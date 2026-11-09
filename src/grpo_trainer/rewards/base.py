"""Reward function contract and the fault-isolation wrapper.

TRL calls each reward as `fn(prompts=..., completions=..., completion_ids=..., **dataset_columns,
trainer_state=..., log_extra=..., log_metric=...)` and expects `list[float | None]` of len(completions).
TRL logs each reward under `rewards/<fn.__name__>/mean`, so names must be stable and unique.

A reward that raises would kill a multi-hour run. Every reward is wrapped so that a per-item exception
scores 0.0 for that item, is counted, and is reported via `log_metric` -- never propagated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

log = logging.getLogger(__name__)


class ItemScorer(Protocol):
    """Scores one completion. Receives the per-item slice of any dataset columns."""

    def __call__(self, text: str, *, completion_ids: Sequence[int] | None, row: dict[str, Any]) -> float: ...


RewardFunc = Callable[..., list[float]]

# Keys TRL injects that are not per-row dataset columns.
_NON_ROW_KWARGS = frozenset({"trainer_state", "log_extra", "log_metric", "environments"})


def make_reward(name: str, scorer: ItemScorer) -> RewardFunc:
    from .parsing import completion_to_text

    def reward(
        prompts: list[Any],
        completions: list[Any],
        completion_ids: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        n = len(completions)
        row_cols = {k: v for k, v in kwargs.items() if k not in _NON_ROW_KWARGS and isinstance(v, list)}
        bad_cols = [k for k, v in row_cols.items() if len(v) != n]
        if bad_cols:
            log.error(
                "reward %s: column length mismatch %s (expected %d); scoring batch 0.0", name, bad_cols, n
            )
            return [0.0] * n
        log_metric = kwargs.get("log_metric")
        scores: list[float] = []
        errors = 0
        for i, completion in enumerate(completions):
            row = {k: v[i] for k, v in row_cols.items()}
            ids = completion_ids[i] if completion_ids is not None and i < len(completion_ids) else None
            try:
                score = float(scorer(completion_to_text(completion), completion_ids=ids, row=row))
                if score != score or score in (float("inf"), float("-inf")):  # NaN/inf guard
                    raise ValueError(f"non-finite score {score}")
            except Exception:  # noqa: BLE001 - isolation boundary by design
                errors += 1
                if errors == 1:
                    log.exception("reward %s failed on item %d; scoring 0.0", name, i)
                score = 0.0
            scores.append(score)
        if callable(log_metric) and n:
            try:
                log_metric(f"reward_errors/{name}", errors / n)
            except Exception:  # noqa: BLE001 - metrics must never break training
                log.debug("log_metric unavailable for %s", name, exc_info=True)
        return scores

    reward.__name__ = name
    reward.__qualname__ = name
    return reward
