"""Typed field semantics: normalise labels, coerce predictions, check strict types, score matches.

Three deliberately different views of a value:

* `normalize_label`  -- the label as the customer wrote it -> canonical value. Invalid labels raise
                        LabelError (a data bug the customer must fix, reported by `grpo spec check`).
* `coerce_prediction`-- what the model *meant*: "$1,234.50" is the number 1234.5, "03/04/2025" in a
                        declared input format is a date. Drives correctness (knowledge).
* `strict_type_error`-- what the model *should have emitted*: a JSON number, an ISO date, the exact enum
                        value. Drives the format reward and the serving `valid` flag (contract).

Keeping knowledge and contract separate is what lets the proof report say how much of a gain is
"learned the schema" versus "extracts the right values".

Canonical values: string -> str, number -> Decimal, integer -> int, date -> datetime.date,
enum -> the declared value, boolean -> bool, list -> list of canonical items, object_list -> list of
dicts of canonical values. Empty strings and empty lists mean "absent" (None).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any

from ..rewards.parsing import parse_single_number
from .spec import ListField, ObjectListField, ScalarField

INVALID: Any = type("Invalid", (), {"__repr__": lambda self: "INVALID"})()
_ISO_DATE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}

# Spraying guard: a list prediction far longer than the label scores 0 for that field, and also keeps
# alignment cost bounded inside the reward loop.
MAX_EXTRA_ITEMS = 20
MAX_ITEM_RATIO = 4


class LabelError(ValueError):
    """A label value that does not fit its declared field type."""


# ------------------------------------------------------------------ shared helpers


def normalize_string(s: str) -> str:
    """Comparison form: NFKC, case-folded, whitespace collapsed, edge punctuation stripped."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = " ".join(s.split())
    return s.strip(" .,;:")


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _decimal_from_json_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def _decimal_lenient(value: Any) -> Decimal | None:
    d = _decimal_from_json_number(value)
    if d is not None:
        return d
    if isinstance(value, str):
        return parse_single_number(value)  # "$1,234.50" -> 1234.50; "72 or 73" -> None
    return None


def _parse_date(value: str, formats: list[str]) -> date | None:
    s = value.strip()
    if _ISO_DATE.match(s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _enum_lookup(f: ScalarField, value: str) -> str | None:
    s = value.strip()
    if f.case_sensitive:
        return s if s in f.values else None
    folded = {v.casefold(): v for v in f.values}
    return folded.get(s.casefold())


def _bool_lenient(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().casefold()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


# ------------------------------------------------------------------ labels


def normalize_label(name: str, f: ScalarField | ListField | ObjectListField, value: Any) -> Any:
    """Canonicalise one label value. Raises LabelError naming the field on any type violation."""
    if isinstance(f, ListField):
        return _normalize_list_label(name, f, value)
    if isinstance(f, ObjectListField):
        return _normalize_object_list_label(name, f, value)
    return _normalize_scalar_label(name, f, value)


def _normalize_scalar_label(name: str, f: ScalarField, value: Any) -> Any:
    if _is_blank(value):
        if not f.nullable:
            raise LabelError(f"{name}: value is required (nullable: false) but label is empty")
        return None
    t = f.type
    if t == "string":
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise LabelError(f"{name}: expected string, got {type(value).__name__}")
        return str(value).strip()
    if t in ("number", "integer"):
        d = _decimal_lenient(value)
        if d is None:
            raise LabelError(f"{name}: expected {t}, got {value!r}")
        if t == "integer":
            if d != d.to_integral_value():
                raise LabelError(f"{name}: expected integer, got {value!r}")
            return int(d)
        return d
    if t == "date":
        if not isinstance(value, str):
            raise LabelError(f"{name}: expected date string, got {type(value).__name__}")
        parsed = _parse_date(value, f.input_formats)
        if parsed is None:
            raise LabelError(
                f"{name}: {value!r} is not ISO (YYYY-MM-DD) or any declared input_formats {f.input_formats}"
            )
        return parsed
    if t == "enum":
        if not isinstance(value, str):
            raise LabelError(f"{name}: expected one of {f.values}, got {value!r}")
        canon = _enum_lookup(f, value)
        if canon is None:
            raise LabelError(f"{name}: {value!r} is not one of {f.values}")
        return canon
    if t == "boolean":
        b = _bool_lenient(value)
        if b is None:
            raise LabelError(f"{name}: expected boolean, got {value!r}")
        return b
    raise AssertionError(t)


def _normalize_list_label(name: str, f: ListField, value: Any) -> list[Any] | None:
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        raise LabelError(f"{name}: expected a list, got {type(value).__name__}")
    out = [_normalize_scalar_label(f"{name}[{i}]", f.items, v) for i, v in enumerate(value)]
    return out or None


def _normalize_object_list_label(name: str, f: ObjectListField, value: Any) -> list[dict[str, Any]] | None:
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        raise LabelError(f"{name}: expected a list of objects, got {type(value).__name__}")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise LabelError(f"{name}[{i}]: expected an object, got {type(item).__name__}")
        unknown = sorted(set(item) - set(f.fields))
        if unknown:
            raise LabelError(f"{name}[{i}]: unknown keys {unknown}; declared {sorted(f.fields)}")
        out.append(
            {k: _normalize_scalar_label(f"{name}[{i}].{k}", sub, item.get(k)) for k, sub in f.fields.items()}
        )
    return out or None


# ------------------------------------------------------------------ predictions (lenient)


def coerce_prediction(f: ScalarField | ListField | ObjectListField, value: Any) -> Any:
    """Model value -> canonical value, None (absent), or INVALID. Never raises."""
    if isinstance(f, ListField):
        if value is None or value == []:
            return None
        if not isinstance(value, list):
            return INVALID
        return [_coerce_scalar(f.items, v) for v in value]
    if isinstance(f, ObjectListField):
        if value is None or value == []:
            return None
        if not isinstance(value, list):
            return INVALID
        return [
            {k: _coerce_scalar(sub, item.get(k)) for k, sub in f.fields.items()}
            if isinstance(item, dict)
            else INVALID
            for item in value
        ]
    return _coerce_scalar(f, value)


def _coerce_scalar(f: ScalarField, value: Any) -> Any:
    if _is_blank(value):
        return None
    t = f.type
    if t == "string":
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return INVALID
        return str(value).strip()
    if t == "number":
        d = _decimal_lenient(value)
        return INVALID if d is None else d
    if t == "integer":
        d = _decimal_lenient(value)
        if d is None or d != d.to_integral_value():
            return INVALID
        return int(d)
    if t == "date":
        if not isinstance(value, str):
            return INVALID
        parsed = _parse_date(value, f.input_formats)
        return INVALID if parsed is None else parsed
    if t == "enum":
        if not isinstance(value, str):
            return INVALID
        # Case-insensitive lookup even for case-sensitive enums: knowledge, not contract.
        folded = {v.casefold(): v for v in f.values}
        canon = folded.get(value.strip().casefold())
        return INVALID if canon is None else canon
    if t == "boolean":
        b = _bool_lenient(value)
        return INVALID if b is None else b
    raise AssertionError(t)


# ------------------------------------------------------------------ strict contract


def strict_type_error(name: str, f: ScalarField | ListField | ObjectListField, value: Any) -> str | None:
    """None if `value` is exactly what the contract asks for, else a human-readable reason."""
    if isinstance(f, ListField):
        if value is None:
            return None if f.nullable else f"{name}: must be a list, got null"
        if not isinstance(value, list):
            return f"{name}: must be a list"
        for i, v in enumerate(value):
            if v is None:
                return f"{name}[{i}]: list items may not be null"
            err = _strict_scalar(f"{name}[{i}]", f.items, v)
            if err:
                return err
        return None
    if isinstance(f, ObjectListField):
        if value is None:
            return None if f.nullable else f"{name}: must be a list, got null"
        if not isinstance(value, list):
            return f"{name}: must be a list of objects"
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                return f"{name}[{i}]: must be an object"
            if set(item) != set(f.fields):
                missing = sorted(set(f.fields) - set(item))
                extra = sorted(set(item) - set(f.fields))
                return f"{name}[{i}]: keys differ from schema (missing {missing}, unexpected {extra})"
            for k, sub in f.fields.items():
                err = _strict_scalar(f"{name}[{i}].{k}", sub, item[k])
                if err:
                    return err
        return None
    return _strict_scalar(name, f, value)


def _strict_scalar(name: str, f: ScalarField, value: Any) -> str | None:
    if value is None:
        return None if f.nullable else f"{name}: null is not allowed"
    t = f.type
    if t == "string":
        return None if isinstance(value, str) and value.strip() else f"{name}: must be a non-empty string"
    if t == "number":
        if _decimal_from_json_number(value) is None:
            return f"{name}: must be a JSON number (no quotes, symbols or separators)"
        return None
    if t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{name}: must be a JSON integer"
        return None
    if t == "date":
        if not (isinstance(value, str) and _ISO_DATE.match(value)):
            return f"{name}: must be an ISO date string YYYY-MM-DD"
        try:
            date.fromisoformat(value)
        except ValueError:
            return f"{name}: {value!r} is not a valid calendar date"
        return None
    if t == "enum":
        return None if isinstance(value, str) and value in f.values else f"{name}: must be one of {f.values}"
    if t == "boolean":
        return None if isinstance(value, bool) else f"{name}: must be true or false"
    raise AssertionError(t)


# ------------------------------------------------------------------ matching and scoring


def scalar_match(f: ScalarField, pred: Any, gt: Any) -> bool:
    """Both arguments canonical and non-None."""
    if pred is INVALID:
        return False
    t = f.type
    if t == "string":
        if f.match == "exact":
            return pred == gt
        a, b = normalize_string(pred), normalize_string(gt)
        if f.match == "normalized":
            return a == b
        return a == b or SequenceMatcher(None, a, b, autojunk=False).ratio() >= f.threshold
    if t == "number":
        allowed = max(Decimal(str(f.tolerance)), Decimal(str(f.rel_tolerance)) * abs(gt))
        return abs(pred - gt) <= allowed
    return pred == gt  # integer, date, enum, boolean


def _scalar_score(f: ScalarField, pred: Any, gt: Any) -> float:
    """Null-aware: absent == absent is correct; one side absent is wrong."""
    if gt is None:
        return 1.0 if pred is None else 0.0
    if pred is None or pred is INVALID:
        return 0.0
    return 1.0 if scalar_match(f, pred, gt) else 0.0


def _too_many(n_pred: int, n_gt: int) -> bool:
    return n_pred > max(MAX_ITEM_RATIO * n_gt, n_gt + MAX_EXTRA_ITEMS)


def _greedy_f1(pred: list[Any], gt: list[Any], sim) -> float:  # noqa: ANN001 - small private callable
    """Soft F1 over a greedy one-to-one alignment by descending similarity (deterministic tie-break)."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            s = sim(p, g)
            if s > 0.0:
                pairs.append((-s, i, j))
    pairs.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0.0
    for neg_s, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += -neg_s
    if tp == 0.0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(gt)
    return 2 * precision * recall / (precision + recall)


def item_similarity(f: ObjectListField, pred: Any, gt: dict[str, Any]) -> float:
    if not isinstance(pred, dict):
        return 0.0
    total = sum(sub.weight for sub in f.fields.values())
    got = sum(sub.weight * _scalar_score(sub, pred.get(k), gt.get(k)) for k, sub in f.fields.items())
    return got / total


@dataclass(frozen=True)
class FieldResult:
    score: float  # in [0, 1]
    status: str  # correct | partial | wrong | hallucinated | omitted | missing_key | invalid


def score_field(
    f: ScalarField | ListField | ObjectListField, present: bool, pred: Any, gt: Any
) -> FieldResult:
    """`pred` is the output of `coerce_prediction`; `gt` the output of `normalize_label`."""
    if not present:
        return FieldResult(1.0, "correct") if gt is None else FieldResult(0.0, "missing_key")
    if pred is INVALID:
        return FieldResult(0.0, "invalid")
    if gt is None:
        return FieldResult(1.0, "correct") if pred is None else FieldResult(0.0, "hallucinated")
    if pred is None:
        return FieldResult(0.0, "omitted")
    if isinstance(f, ScalarField):
        ok = scalar_match(f, pred, gt)
        return FieldResult(1.0 if ok else 0.0, "correct" if ok else "wrong")
    if _too_many(len(pred), len(gt)):
        return FieldResult(0.0, "wrong")
    if isinstance(f, ListField):
        s = _greedy_f1(pred, gt, lambda p, g: 1.0 if scalar_match(f.items, p, g) else 0.0)
    else:
        s = _greedy_f1(pred, gt, lambda p, g: item_similarity(f, p, g))
    if s >= 1.0 - 1e-12:
        return FieldResult(1.0, "correct")
    return FieldResult(s, "partial" if s > 0.0 else "wrong")


# ------------------------------------------------------------------ JSON round-trip of canonical values


def to_json_value(f: ScalarField | ListField | ObjectListField, value: Any) -> Any:
    """Canonical value -> the JSON value the contract asks the model (and the API) to emit."""
    if value is None:
        return [] if isinstance(f, (ListField, ObjectListField)) and not f.nullable else None
    if isinstance(f, ListField):
        return [_scalar_to_json(f.items, v) for v in value]
    if isinstance(f, ObjectListField):
        return [{k: _scalar_to_json(sub, item.get(k)) for k, sub in f.fields.items()} for item in value]
    return _scalar_to_json(f, value)


def _scalar_to_json(f: ScalarField, value: Any) -> Any:
    if value is None:
        return None
    if f.type == "number":
        d: Decimal = value
        return int(d) if d == d.to_integral_value() and abs(d) < 2**53 else float(d)
    if f.type == "date":
        return value.isoformat()
    return value


def to_storage(f: ScalarField | ListField | ObjectListField, value: Any) -> Any:
    """Canonical value -> lossless JSON for ground-truth storage (numbers as decimal strings)."""
    if value is None:
        return None
    if isinstance(f, ListField):
        return [_scalar_to_storage(f.items, v) for v in value]
    if isinstance(f, ObjectListField):
        return [{k: _scalar_to_storage(sub, item.get(k)) for k, sub in f.fields.items()} for item in value]
    return _scalar_to_storage(f, value)


def _scalar_to_storage(f: ScalarField, value: Any) -> Any:
    if value is None:
        return None
    if f.type == "number":
        return format(value, "f")
    if f.type == "date":
        return value.isoformat()
    return value


def from_storage(f: ScalarField | ListField | ObjectListField, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(f, ListField):
        return [_scalar_from_storage(f.items, v) for v in value]
    if isinstance(f, ObjectListField):
        return [{k: _scalar_from_storage(sub, item.get(k)) for k, sub in f.fields.items()} for item in value]
    return _scalar_from_storage(f, value)


def _scalar_from_storage(f: ScalarField, value: Any) -> Any:
    if value is None:
        return None
    if f.type == "number":
        return Decimal(value)
    if f.type == "date":
        return date.fromisoformat(value)
    return value
