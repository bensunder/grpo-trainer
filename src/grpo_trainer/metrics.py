"""Pure evaluation math and the promotion gate. No I/O, no GPU."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .config import GateConfig
from .errors import GateFailedError
from .rewards.parsing import lenient_correct

COMPARABLE_KEYS = (
    "split",
    "num_examples",
    "dataset_fingerprint",
    "grading_fingerprint",  # absent in older reports -> None on both sides -> still comparable
    "max_completion_length",
    "system_prompt_sha",
)

Judge = Callable[[str, str], bool]  # (output text, ground truth) -> correct


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


def mcnemar_exact(only_base: int, only_candidate: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    Under H0 (no difference) each disagreement is a fair coin, so the smaller count ~ Binomial(n, 0.5).
    Questions both models got right (or both wrong) carry no information and are excluded by design.
    """
    if only_base < 0 or only_candidate < 0:
        raise ValueError("counts must be non-negative")
    n = only_base + only_candidate
    if n == 0:
        return 1.0
    k = min(only_base, only_candidate)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff_ci(only_a: int, only_b: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wald CI for the paired accuracy difference acc(b) - acc(a) on the same n questions.

    d = (only_b - only_a) / n,  se = sqrt(only_a + only_b - (only_b - only_a)^2 / n) / n.
    Adequate for reporting at n in the hundreds; the significance claim itself uses the exact test.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    d = (only_b - only_a) / n
    var = only_a + only_b - (only_b - only_a) ** 2 / n
    se = math.sqrt(max(var, 0.0)) / n
    return max(-1.0, d - z * se), min(1.0, d + z * se)


@dataclass(frozen=True)
class PairedResult:
    n: int
    base_accuracy: float
    candidate_accuracy: float
    only_base_correct: int
    only_candidate_correct: int
    p_value: float

    @property
    def delta(self) -> float:
        return self.candidate_accuracy - self.base_accuracy

    @property
    def ci95(self) -> tuple[float, float]:
        return paired_diff_ci(self.only_base_correct, self.only_candidate_correct, self.n)


PairedLenient = PairedResult  # backward-compatible name


def index_rows(rows: Iterable[dict[str, Any]], label: str, judge: Judge) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for r in rows:
        try:
            qid, text, gt = str(r["id"]), r.get("output") or "", r["ground_truth"]
        except KeyError as e:
            raise GateFailedError(f"{label} examples missing field {e}") from None
        if qid in out:
            raise GateFailedError(f"{label} examples contain duplicate id {qid!r}")
        out[qid] = judge(text, str(gt))
    return out


def paired_from_index(b: dict[str, bool], c: dict[str, bool]) -> PairedResult:
    if set(b) != set(c):
        only = sorted(set(b) ^ set(c))[:5]
        raise GateFailedError(f"base and candidate were evaluated on different questions (e.g. {only})")
    if not b:
        raise GateFailedError("no evaluated examples to compare")
    only_base = sum(b[q] and not c[q] for q in b)
    only_cand = sum(c[q] and not b[q] for q in b)
    n = len(b)
    return PairedResult(
        n=n,
        base_accuracy=sum(b.values()) / n,
        candidate_accuracy=sum(c.values()) / n,
        only_base_correct=only_base,
        only_candidate_correct=only_cand,
        p_value=mcnemar_exact(only_base, only_cand),
    )


def paired_compare(
    base_rows: Iterable[dict[str, Any]], cand_rows: Iterable[dict[str, Any]], judge: Judge
) -> PairedResult:
    """Re-score both models' saved outputs with `judge` and pair them by question id."""
    return paired_from_index(index_rows(base_rows, "base", judge), index_rows(cand_rows, "candidate", judge))


def paired_lenient(
    base_rows: Iterable[dict[str, Any]],
    cand_rows: Iterable[dict[str, Any]],
    judge: Judge = lenient_correct,
) -> PairedResult:
    """Lenient (knowledge-only) paired comparison. Defaults to the numeric GSM8K judge.

    Recomputed from `output` + `ground_truth` rather than trusting a stored flag, so it also works on
    reports written before lenient scoring existed.
    """
    return paired_compare(base_rows, cand_rows, judge)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    accuracy_delta: float  # strict: <answer>-tagged answer correct (format + knowledge)
    format_delta: float
    lenient_accuracy_base: float
    lenient_accuracy_candidate: float
    lenient_delta: float  # knowledge only, format ignored
    only_base_correct: int
    only_candidate_correct: int
    p_value: float
    significant: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def evaluate_gate(
    base: dict[str, Any],
    candidate: dict[str, Any],
    gate: GateConfig,
    base_rows: Iterable[dict[str, Any]],
    cand_rows: Iterable[dict[str, Any]],
    judge: Judge = lenient_correct,
) -> GateResult:
    """Compare two eval reports (and their per-question outputs) produced by `grpo eval`.

    `judge` is the task's lenient (knowledge-only) correctness; the numeric default is GSM8K's.
    """
    for key in COMPARABLE_KEYS:
        if base.get(key) != candidate.get(key):
            raise GateFailedError(
                f"reports are not comparable: {key} differs ({base.get(key)!r} vs {candidate.get(key)!r})"
            )
    acc_delta = float(candidate["greedy"]["accuracy"]) - float(base["greedy"]["accuracy"])
    fmt_delta = float(candidate["greedy"]["format_rate"]) - float(base["greedy"]["format_rate"])
    lp = paired_lenient(base_rows, cand_rows, judge)
    if lp.n != int(base["num_examples"]):
        raise GateFailedError(f"examples file has {lp.n} rows but report says {base['num_examples']}")
    significant = lp.p_value <= gate.alpha and lp.only_candidate_correct > lp.only_base_correct

    reasons: list[str] = []
    if acc_delta < gate.min_accuracy_delta:
        reasons.append(f"strict accuracy delta {acc_delta:+.4f} < required {gate.min_accuracy_delta:+.4f}")
    if fmt_delta < -gate.max_format_regression:
        reasons.append(f"format rate regressed {fmt_delta:+.4f} (allowed -{gate.max_format_regression:.4f})")
    if lp.delta < gate.min_lenient_accuracy_delta:
        reasons.append(
            f"lenient (reasoning) accuracy delta {lp.delta:+.4f} "
            f"< required {gate.min_lenient_accuracy_delta:+.4f}"
        )
    if gate.require_significant_gain and not significant:
        reasons.append(
            f"reasoning gain not significant: only-candidate {lp.only_candidate_correct} vs only-base "
            f"{lp.only_base_correct}, p={lp.p_value:.4f} (need p <= {gate.alpha} in candidate's favour)"
        )
    return GateResult(
        passed=not reasons,
        accuracy_delta=acc_delta,
        format_delta=fmt_delta,
        lenient_accuracy_base=lp.base_accuracy,
        lenient_accuracy_candidate=lp.candidate_accuracy,
        lenient_delta=lp.delta,
        only_base_correct=lp.only_base_correct,
        only_candidate_correct=lp.only_candidate_correct,
        p_value=lp.p_value,
        significant=significant,
        reasons=tuple(reasons),
    )
