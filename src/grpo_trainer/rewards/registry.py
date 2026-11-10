"""Name -> reward factory. Adding a reward = one scorer class + one entry here.

`correctness` and `format` are task-owned: when a Task is in the context, the task builds them (a
structured-extraction task grades JSON fields, GSM8K grades a number). Without a task they fall back to
the numeric scorers, which is what GSM8K uses anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import Config, RewardSpec
from ..errors import ConfigError
from .base import ItemScorer, RewardFunc, make_reward
from .scorers import CorrectnessScorer, FormatScorer, LengthPenaltyScorer

if TYPE_CHECKING:
    from ..tasks.base import Task


@dataclass(frozen=True)
class RewardContext:
    """Run-level facts a factory may need (kept explicit; factories never see the whole Config)."""

    max_completion_length: int
    task: Task | None = None


ScorerFactory = Callable[[dict[str, Any], RewardContext], ItemScorer]


def _correctness(params: dict[str, Any], ctx: RewardContext) -> ItemScorer:
    if ctx.task is not None:
        return ctx.task.correctness_scorer(params)
    return CorrectnessScorer(**params)


def _format(params: dict[str, Any], ctx: RewardContext) -> ItemScorer:
    if ctx.task is not None:
        return ctx.task.format_scorer(params)
    return FormatScorer(**params)


def _length(params: dict[str, Any], ctx: RewardContext) -> ItemScorer:
    params = {"max_tokens": ctx.max_completion_length, **params}
    return LengthPenaltyScorer(**params)


REGISTRY: dict[str, ScorerFactory] = {
    "correctness": _correctness,
    "format": _format,
    "length": _length,
}


def build_scorer(spec: RewardSpec, ctx: RewardContext) -> ItemScorer:
    factory = REGISTRY.get(spec.name)
    if factory is None:
        raise ConfigError(f"unknown reward {spec.name!r}; registered: {sorted(REGISTRY)}")
    try:
        return factory(dict(spec.params), ctx)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"bad params for reward {spec.name!r}: {e}") from e


def reward_context(cfg: Config, task: Task | None) -> RewardContext:
    return RewardContext(max_completion_length=cfg.grpo.max_completion_length, task=task)


def build_rewards(cfg: Config, task: Task | None = None) -> tuple[list[RewardFunc], list[float]]:
    """Return TRL-ready reward functions and their weights, in config order."""
    ctx = reward_context(cfg, task)
    funcs: list[RewardFunc] = []
    weights: list[float] = []
    for spec in cfg.rewards:
        funcs.append(make_reward(spec.name, build_scorer(spec, ctx)))
        weights.append(spec.weight)
    return funcs, weights
