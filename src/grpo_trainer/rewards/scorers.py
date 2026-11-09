"""The built-in item scorers. Each is a pure function of (text, token ids, row)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .parsing import extract_answer, match_contract, numbers_equal, parse_single_number


@dataclass(frozen=True)
class CorrectnessScorer:
    """1.0 iff the single <answer> block holds exactly one number equal to the ground truth.

    `require_contract=True` additionally demands the full think/answer format; default False keeps the
    correctness signal independent so it can teach the answer even while format is still being learned.
    """

    ground_truth_column: str = "ground_truth"
    require_contract: bool = False

    def __call__(self, text: str, *, completion_ids: Sequence[int] | None, row: dict[str, Any]) -> float:
        gt = row.get(self.ground_truth_column)
        if gt is None:
            raise KeyError(f"dataset column {self.ground_truth_column!r} missing from reward kwargs")
        if self.require_contract and match_contract(text) is None:
            return 0.0
        answer = extract_answer(text)
        if answer is None:
            return 0.0
        return 1.0 if numbers_equal(parse_single_number(answer), str(gt)) else 0.0


@dataclass(frozen=True)
class FormatScorer:
    """1.0 iff output is exactly `<think>REASONING</think><answer>X</answer>`.

    Blocks the empty-think hack: reasoning must have at least `min_think_chars` non-space characters,
    and the answer must be non-empty and at most `max_answer_chars`.
    """

    min_think_chars: int = 20
    max_answer_chars: int = 64

    def __call__(self, text: str, *, completion_ids: Sequence[int] | None, row: dict[str, Any]) -> float:
        parts = match_contract(text)
        if parts is None:
            return 0.0
        think, answer = parts
        if len("".join(think.split())) < self.min_think_chars:
            return 0.0
        a = answer.strip()
        if not a or len(a) > self.max_answer_chars:
            return 0.0
        return 1.0


@dataclass(frozen=True)
class LengthPenaltyScorer:
    """Smooth penalty in [-1, 0] on token count.

    0 up to `soft_fraction * max_tokens`, then linear to -1 at `max_tokens`. Uses token ids when TRL
    provides them; falls back to a chars/4 estimate otherwise. Never rewards brevity below the soft limit,
    so it cannot push the model toward empty answers.
    """

    max_tokens: int
    soft_fraction: float = 0.75

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not 0.0 < self.soft_fraction < 1.0:
            raise ValueError("soft_fraction must be in (0, 1)")

    def __call__(self, text: str, *, completion_ids: Sequence[int] | None, row: dict[str, Any]) -> float:
        n = len(completion_ids) if completion_ids is not None else max(1, len(text) // 4)
        soft = self.soft_fraction * self.max_tokens
        if n <= soft:
            return 0.0
        return -min(1.0, (n - soft) / (self.max_tokens - soft))
