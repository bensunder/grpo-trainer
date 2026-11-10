"""Task contract: where prompts and ground truth come from, and the output contract the model is told."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..errors import DataError

if TYPE_CHECKING:
    from ..rewards.base import ItemScorer

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful problem solver. First reason step by step inside <think> and </think>. "
    "Then give only the final answer as a single number inside <answer> and </answer>, "
    "with no units and no other text. Output nothing after </answer>."
)


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    ground_truth: str

    def messages(self, system_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.question},
        ]

    def to_row(self, system_prompt: str) -> dict[str, Any]:
        """Row shape consumed by GRPOTrainer: `prompt` + extra columns forwarded to rewards."""
        return {"id": self.id, "prompt": self.messages(system_prompt), "ground_truth": self.ground_truth}


def content_hash(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Prediction:
    """A model output parsed for serving: the value a client receives, and whether it met the contract."""

    output: Any
    valid: bool
    errors: tuple[str, ...] = ()


class Task(ABC):
    """A task owns its data AND its grading: what correct means, what the output contract is.

    The defaults implement the numeric single-answer contract (GSM8K). A task with a different answer
    shape (e.g. structured extraction) overrides the grading hooks; eval, profiling, the gate, the report
    and the serving gateway all go through these hooks, never through task-specific code.
    """

    name: str
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @abstractmethod
    def load(self, split: str, limit: int | None = None) -> list[Example]:
        """Load a split. Must be deterministic for a given (split, limit)."""

    def system_prompt(self, override: str | None) -> str:
        return override if override else self.default_system_prompt

    # ------------------------------------------------------------- grading hooks

    def correctness_scorer(self, params: dict[str, Any]) -> ItemScorer:
        """Scorer in [0, 1]; a score of exactly 1.0 means fully correct (used for pass rates)."""
        from ..rewards.scorers import CorrectnessScorer

        return CorrectnessScorer(**params)

    def format_scorer(self, params: dict[str, Any]) -> ItemScorer:
        """1.0 iff the output meets the full output contract."""
        from ..rewards.scorers import FormatScorer

        return FormatScorer(**params)

    def lenient_correct(self, text: str, ground_truth: str) -> bool:
        """Knowledge-only correctness, format ignored. Eval/gate only; never a training reward."""
        from ..rewards.parsing import lenient_correct

        return lenient_correct(text, ground_truth)

    def eval_detail(self, text: str, ground_truth: str) -> dict[str, Any] | None:
        """Optional per-example breakdown stored in eval JSONL (e.g. per-field status). None = none."""
        return None

    def grading_fingerprint(self) -> str | None:
        """Hash of grading rules that live outside the config (e.g. a spec file). None = all in config."""
        return None

    def parse_prediction(self, text: str) -> Prediction:
        """Turn one raw completion into the value a serving client receives."""
        from ..rewards.parsing import extract_answer, format_decimal, match_contract, parse_single_number

        errors: list[str] = []
        if match_contract(text) is None:
            errors.append("output does not match the <think>/<answer> contract")
        answer = extract_answer(text)
        value = parse_single_number(answer) if answer is not None else None
        if value is None:
            errors.append("answer block missing or does not hold exactly one number")
            return Prediction(output=None, valid=False, errors=tuple(errors))
        return Prediction(output=format_decimal(value), valid=not errors, errors=tuple(errors))


def drop_contaminated(train: list[Example], held_out: Iterable[Example]) -> tuple[list[Example], int]:
    """Remove training examples whose question also appears in the held-out set. Returns (clean, dropped)."""
    held = {content_hash(e.question) for e in held_out}
    clean = [e for e in train if content_hash(e.question) not in held]
    return clean, len(train) - len(clean)


def assert_disjoint(train: Iterable[Example], held_out: Iterable[Example]) -> None:
    """Contamination guard: no held-out question may appear (normalised) in the training set."""
    train_hashes = {content_hash(e.question) for e in train}
    leaked = [e.id for e in held_out if content_hash(e.question) in train_hashes]
    if leaked:
        raise DataError(f"{len(leaked)} held-out examples also appear in train (first: {leaked[:5]})")


def dataset_fingerprint(examples: Iterable[Example]) -> str:
    h = hashlib.sha256()
    for e in examples:
        h.update(e.id.encode())
        h.update(b"\x00")
        h.update(content_hash(e.question).encode())
        h.update(b"\x00")
        h.update(e.ground_truth.encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]
