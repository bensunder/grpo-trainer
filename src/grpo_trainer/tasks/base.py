"""Task contract: where prompts and ground truth come from, and the output contract the model is told."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..errors import DataError

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


class Task(ABC):
    name: str
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @abstractmethod
    def load(self, split: str, limit: int | None = None) -> list[Example]:
        """Load a split. Must be deterministic for a given (split, limit)."""

    def system_prompt(self, override: str | None) -> str:
        return override if override else self.default_system_prompt


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
