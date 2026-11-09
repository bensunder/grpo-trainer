"""Name -> reward factory. Adding a reward = one scorer class + one entry here."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import Config, RewardSpec
from ..errors import ConfigError
from .base import ItemScorer, RewardFunc, make_reward
from .scorers import CorrectnessScorer, FormatScorer, LengthPenaltyScorer


@dataclass(frozen=True)
class RewardContext:
    """Run-level facts a factory may need (kept explicit; factories never see the whole Config)."""

    max_completion_length: int


ScorerFactory = Callable[[dict[str, Any], RewardContext], ItemScorer]


def _correctness(params: dict[str, Any], ctx: RewardContext) -> ItemScorer:
    return CorrectnessScorer(**params)


def _format(params: dict[str, Any], ctx: RewardContext) -> ItemScorer:
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


def build_rewards(cfg: Config) -> tuple[list[RewardFunc], list[float]]:
    """Return TRL-ready reward functions and their weights, in config order."""
    ctx = RewardContext(max_completion_length=cfg.grpo.max_completion_length)
    funcs: list[RewardFunc] = []
    weights: list[float] = []
    for spec in cfg.rewards:
        funcs.append(make_reward(spec.name, build_scorer(spec, ctx)))
        weights.append(spec.weight)
    return funcs, weights
