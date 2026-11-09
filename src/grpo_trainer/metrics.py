"""Pure evaluation math and the promotion gate. No I/O, no GPU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import GateConfig
from .errors import GateFailedError


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if not 0 <= c <= n:
        raise ValueError(f"c ({c}) must be in [0, n]")
    if n - c < k:
        return 1.0
    # Product form avoids huge binomials.
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@dataclass(frozen=True)
class GateResult:
    passed: bool
    accuracy_delta: float
    format_delta: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "accuracy_delta": self.accuracy_delta,
            "format_delta": self.format_delta,
            "reasons": list(self.reasons),
        }


def evaluate_gate(base: dict[str, Any], candidate: dict[str, Any], gate: GateConfig) -> GateResult:
    """Compare two eval reports produced by `grpo eval` on the same split/limit."""
    for key in ("split", "num_examples", "dataset_fingerprint", "max_completion_length", "system_prompt_sha"):
        if base.get(key) != candidate.get(key):
            raise GateFailedError(
                f"reports are not comparable: {key} differs ({base.get(key)!r} vs {candidate.get(key)!r})"
            )
    acc_delta = float(candidate["greedy"]["accuracy"]) - float(base["greedy"]["accuracy"])
    fmt_delta = float(candidate["greedy"]["format_rate"]) - float(base["greedy"]["format_rate"])
    reasons: list[str] = []
    if acc_delta < gate.min_accuracy_delta:
        reasons.append(f"accuracy delta {acc_delta:+.4f} < required {gate.min_accuracy_delta:+.4f}")
    if fmt_delta < -gate.max_format_regression:
        reasons.append(f"format rate regressed {fmt_delta:+.4f} (allowed -{gate.max_format_regression:.4f})")
    return GateResult(
        passed=not reasons, accuracy_delta=acc_delta, format_delta=fmt_delta, reasons=tuple(reasons)
    )
