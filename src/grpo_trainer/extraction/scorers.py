"""Reward scorers for structured extraction. Same contract as rewards.scorers: pure (text, ids, row)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .grader import Grader


@dataclass(frozen=True)
class ExtractionCorrectnessScorer:
    """Field-level correctness in [0, 1]; exactly 1.0 iff every field matches.

    Values are coerced leniently ("$1,234.50" counts as 1234.5), so this measures extraction knowledge;
    the strict JSON contract is the format reward's job.
    """

    grader: Grader
    ground_truth_column: str = "ground_truth"
    require_contract: bool = False

    def __call__(self, text: str, *, completion_ids: Sequence[int] | None, row: dict[str, Any]) -> float:
        gt_json = row.get(self.ground_truth_column)
        if gt_json is None:
            raise KeyError(f"dataset column {self.ground_truth_column!r} missing from reward kwargs")
        if self.require_contract and self.grader.contract_answer(text) is None:
            return 0.0
        return self.grader.score_text(text, str(gt_json)).score


@dataclass(frozen=True)
class ExtractionFormatScorer:
    """1.0 iff the completion is a schema-exact API response: contract tags, a bare JSON object (no
    fence, no duplicate keys), exactly the declared keys, every value of its declared JSON type.

    With reasoning enabled, the think block must hold at least `min_think_chars` non-space characters
    (blocks the empty-think hack). Correctness of values is not checked here.
    """

    grader: Grader
    min_think_chars: int = 20

    def __call__(self, text: str, *, completion_ids: Sequence[int] | None, row: dict[str, Any]) -> float:
        if self.grader.contract_errors(text):
            return 0.0
        if self.grader.spec.output.reasoning:
            from ..rewards.parsing import match_contract

            parts = match_contract(text)
            if parts is None or len("".join(parts[0].split())) < self.min_think_chars:
                return 0.0
        return 1.0
