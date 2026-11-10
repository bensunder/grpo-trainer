"""Document-level grading for one extraction spec.

A completion is parsed into a JSON object (duplicate keys rejected: `{"total": 1, "total": 2}` is answer
spraying), each declared field is coerced and scored against the canonical label, and the field scores
are combined per the spec's scoring mode. Nothing here raises on model output.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..rewards.parsing import extract_answer, match_contract, normalize_text
from .fields import (
    FieldResult,
    LabelError,
    coerce_prediction,
    from_storage,
    normalize_label,
    score_field,
    strict_type_error,
    to_json_value,
    to_storage,
)
from .spec import ExtractionSpec

_ANSWER_ONLY_RE = re.compile(r"\A\s*<answer>(?P<answer>.*?)</answer>\s*\Z", re.DOTALL)
_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*\Z", re.DOTALL | re.IGNORECASE)
_GT_CACHE_SIZE = 8192


class _DuplicateKey(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(k)
        out[k] = v
    return out


def loads_strict(text: str) -> tuple[Any, str | None]:
    """json.loads that rejects duplicate keys and non-finite numbers. Returns (value, error_code)."""
    try:
        return (
            json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant),
            None,
        )
    except _DuplicateKey:
        return None, "duplicate_keys"
    except (ValueError, RecursionError):
        return None, "invalid_json"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


@dataclass(frozen=True)
class ParsedOutput:
    obj: dict[str, Any] | None
    error: str | None  # no_answer | invalid_json | duplicate_keys | not_object
    fenced: bool = False  # answer was wrapped in a ``` fence (tolerated for scoring, not for format)


@dataclass(frozen=True)
class DocScore:
    score: float  # reward value in [0, 1]
    exact: bool  # every field fully correct
    field_score: float  # weighted mean of field scores, regardless of scoring mode
    fields: dict[str, FieldResult] = field(default_factory=dict)


class Grader:
    def __init__(self, spec: ExtractionSpec) -> None:
        self.spec = spec
        self._weights = {name: f.weight for name, f in spec.fields.items()}
        self._total_weight = sum(self._weights.values())
        self._gt_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    # --------------------------------------------------------------- labels

    def normalize_label(self, label: dict[str, Any]) -> dict[str, Any]:
        """Customer label -> canonical dict over every declared field. Raises LabelError."""
        if not isinstance(label, dict):
            raise LabelError(f"label must be an object, got {type(label).__name__}")
        unknown = sorted(set(label) - set(self.spec.fields))
        if unknown:
            raise LabelError(f"label has keys not in the spec: {unknown}")
        return {name: normalize_label(name, f, label.get(name)) for name, f in self.spec.fields.items()}

    def to_ground_truth(self, canonical: dict[str, Any]) -> str:
        """Canonical dict -> the lossless JSON string stored as the `ground_truth` column."""
        stored = {name: to_storage(f, canonical.get(name)) for name, f in self.spec.fields.items()}
        return json.dumps(stored, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def from_ground_truth(self, gt_json: str) -> dict[str, Any]:
        cached = self._gt_cache.get(gt_json)
        if cached is not None:
            self._gt_cache.move_to_end(gt_json)
            return cached
        stored = json.loads(gt_json)
        if not isinstance(stored, dict):
            raise ValueError("ground_truth must be a JSON object")
        gt = {name: from_storage(f, stored.get(name)) for name, f in self.spec.fields.items()}
        self._gt_cache[gt_json] = gt
        if len(self._gt_cache) > _GT_CACHE_SIZE:
            self._gt_cache.popitem(last=False)
        return gt

    def render_ideal(self, canonical: dict[str, Any]) -> dict[str, Any]:
        """The exact JSON object a perfect model would emit for this label."""
        return {name: to_json_value(f, canonical.get(name)) for name, f in self.spec.fields.items()}

    # --------------------------------------------------------------- parsing model output

    def contract_answer(self, text: str) -> str | None:
        """Answer text if the completion matches the full output contract, else None."""
        if self.spec.output.reasoning:
            parts = match_contract(text)
            return None if parts is None else parts[1]
        t = normalize_text(text)
        if t.count("<answer>") != 1 or t.count("</answer>") != 1 or "<think>" in t or "</think>" in t:
            return None
        m = _ANSWER_ONLY_RE.match(t)
        return None if m is None else m.group("answer")

    def parse_answer(self, text: str) -> ParsedOutput:
        """The single <answer> block, parsed as a JSON object (a ``` fence is tolerated)."""
        answer = extract_answer(text)
        if answer is None:
            return ParsedOutput(None, "no_answer")
        return self._parse_object(answer)

    def _parse_object(self, answer: str) -> ParsedOutput:
        body, fenced = answer.strip(), False
        m = _FENCE_RE.match(body)
        if m is not None:
            body, fenced = m.group("body"), True
        value, err = loads_strict(body)
        if err is not None:
            return ParsedOutput(None, err, fenced)
        if not isinstance(value, dict):
            return ParsedOutput(None, "not_object", fenced)
        return ParsedOutput(value, None, fenced)

    def parse_lenient(self, text: str) -> dict[str, Any] | None:
        """The LAST top-level JSON object anywhere in the text, ignoring tags. Eval only (hackable)."""
        text = normalize_text(text)
        decoder = json.JSONDecoder(object_pairs_hook=_no_duplicates)
        last: dict[str, Any] | None = None
        i, n = 0, len(text)
        while i < n:
            j = text.find("{", i)
            if j < 0:
                break
            try:
                value, end = decoder.raw_decode(text, j)
            except (ValueError, RecursionError):
                i = j + 1
                continue
            if isinstance(value, dict):
                last = value
            i = end
        return last

    # --------------------------------------------------------------- scoring

    def score(self, obj: dict[str, Any] | None, gt: dict[str, Any]) -> DocScore:
        if obj is None:
            results = {name: FieldResult(0.0, "invalid_output") for name in self.spec.fields}
            return DocScore(0.0, False, 0.0, results)
        results = {
            name: score_field(f, name in obj, coerce_prediction(f, obj.get(name)), gt.get(name))
            for name, f in self.spec.fields.items()
        }
        exact = all(r.score >= 1.0 for r in results.values())
        if exact:
            return DocScore(1.0, True, 1.0, results)
        field_score = sum(self._weights[n] * r.score for n, r in results.items()) / self._total_weight
        reward = field_score if self.spec.scoring.mode == "weighted_fields" else 0.0
        return DocScore(reward, False, field_score, results)

    def score_text(self, text: str, gt_json: str) -> DocScore:
        return self.score(self.parse_answer(text).obj, self.from_ground_truth(gt_json))

    # --------------------------------------------------------------- contract

    def contract_errors(self, text: str) -> list[str]:
        """Everything that stops a completion from being a valid, schema-exact API response."""
        answer = self.contract_answer(text)
        if answer is None:
            expected = (
                "<think>..</think><answer>{json}</answer>"
                if self.spec.output.reasoning
                else ("<answer>{json}</answer>")
            )
            return [f"output does not match the contract {expected}"]
        parsed = self._parse_object(answer)
        if parsed.obj is None:
            return [f"answer is not a JSON object ({parsed.error})"]
        errors = ["answer is wrapped in a code fence"] if parsed.fenced else []
        return errors + self.schema_errors(parsed.obj)

    def schema_errors(self, obj: dict[str, Any]) -> list[str]:
        errors = []
        missing = sorted(set(self.spec.fields) - set(obj))
        extra = sorted(set(obj) - set(self.spec.fields))
        if missing:
            errors.append(f"missing keys {missing}")
        if extra:
            errors.append(f"unexpected keys {extra}")
        for name, f in self.spec.fields.items():
            if name in obj:
                err = strict_type_error(name, f, obj[name])
                if err:
                    errors.append(err)
        return errors

    def canonical_output(self, obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Best-effort typed output for an API client: declared keys only, values coerced to the
        contract's JSON types; un-coercible values become null and are reported."""
        out: dict[str, Any] = {}
        problems: list[str] = []
        for name, f in self.spec.fields.items():
            value = coerce_prediction(f, obj.get(name))
            if value is not None and _contains_invalid(value):
                problems.append(f"{name}: could not coerce {obj.get(name)!r}")
                value = None
            out[name] = to_json_value(f, value)
        return out, problems


def _contains_invalid(value: Any) -> bool:
    from .fields import INVALID

    if value is INVALID:
        return True
    if isinstance(value, list):
        return any(_contains_invalid(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_invalid(v) for v in value.values())
    return False
