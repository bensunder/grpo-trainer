"""GSM8K: grade-school math word problems with a single integer answer after `####`."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from ..errors import DataError
from ..rewards.parsing import parse_gsm8k_ground_truth
from .base import Example, Task

log = logging.getLogger(__name__)

HF_DATASET = "openai/gsm8k"
HF_CONFIG = "main"

RowLoader = Callable[[str], Iterable[dict[str, Any]]]


def _hf_loader(split: str) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset  # heavy import kept off the CPU test path

    return load_dataset(HF_DATASET, HF_CONFIG, split=split)


class GSM8KTask(Task):
    name = "gsm8k"

    def __init__(self, loader: RowLoader | None = None) -> None:
        self._loader = loader or _hf_loader

    def load(self, split: str, limit: int | None = None) -> list[Example]:
        try:
            rows = self._loader(split)
        except Exception as e:  # network, auth, bad split
            raise DataError(f"failed to load {HF_DATASET}/{HF_CONFIG} split={split!r}: {e}") from e
        out: list[Example] = []
        skipped = 0
        for idx, row in enumerate(rows):
            question = row.get("question")
            gt = parse_gsm8k_ground_truth(row.get("answer", ""))
            if not isinstance(question, str) or not question.strip() or gt is None:
                skipped += 1
                continue
            out.append(Example(id=f"{split}-{idx}", question=question.strip(), ground_truth=gt))
            if limit is not None and len(out) >= limit:
                break
        if skipped:
            log.warning("gsm8k %s: skipped %d rows with unparseable question/answer", split, skipped)
        if not out:
            raise DataError(f"gsm8k split={split!r} produced no usable examples")
        return out
