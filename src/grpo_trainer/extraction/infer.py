"""`grpo spec init`: draft a spec from labelled JSONL so onboarding starts from 80%, not a blank file.

Types are inferred from label values (booleans, integers, numbers, dates in common formats, low-cardinality
enums, lists of scalars, lists of objects). The draft is a starting point to review, never a final
contract: every inferred matcher is a default, and the output says so. `grpo spec check` must pass on
the reviewed spec before training.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..errors import DataError
from ..rewards.parsing import parse_single_number

# Common label date formats, tried in order. Ambiguous numeric formats are only adopted when every
# observed value parses under exactly one of them.
DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y/%m/%d",
)
ENUM_MAX_VALUES = 12
ENUM_MIN_SAMPLES = 20
ENUM_MAX_DISTINCT_RATIO = 0.2
_IDENTIFIER_RE = re.compile(r"[A-Za-z]")


def _is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return (
        isinstance(v, str)
        and parse_single_number(v) is not None
        and not _IDENTIFIER_RE.search(v.replace("USD", "").replace("EUR", "").replace("GBP", ""))
    )


def _is_integer(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return v.is_integer()
    return False


def _date_formats(values: list[str]) -> list[str] | None:
    """Formats that together parse every value, or None if the values aren't dates."""
    used: list[str] = []
    for v in values:
        s = v.strip()
        matches = [fmt for fmt in DATE_FORMATS if _parses(s, fmt)]
        if not matches:
            return None
        if not any(m in used for m in matches):
            used.append(matches[0])
    if "%m/%d/%Y" in used and "%d/%m/%Y" in used:
        used.remove("%d/%m/%Y")  # prefer one convention; flagged for review below
    return used


def _parses(s: str, fmt: str) -> bool:
    try:
        datetime.strptime(s, fmt)
    except ValueError:
        return False
    return True


def _infer_scalar(values: list[Any], notes: list[str], path: str) -> dict[str, Any]:
    present = [v for v in values if v is not None and not (isinstance(v, str) and not v.strip())]
    nullable = len(present) < len(values)
    out: dict[str, Any] = {}
    if not present:
        notes.append(f"{path}: no non-null values seen; typed as string")
        out["type"] = "string"
    elif all(isinstance(v, bool) for v in present):
        out["type"] = "boolean"
    elif all(_is_integer(v) for v in present):
        out["type"] = "integer"
    elif all(_is_number(v) for v in present):
        out["type"] = "number"
        out["tolerance"] = 0.01
        notes.append(f"{path}: number with tolerance 0.01 -- set to 0 for exact quantities")
    elif all(isinstance(v, str) for v in present) and (fmts := _date_formats(present)) is not None:
        out["type"] = "date"
        non_iso = [f for f in fmts if f != "%Y-%m-%d"]
        if non_iso:
            out["input_formats"] = non_iso
        if any(f in ("%m/%d/%Y", "%d/%m/%Y") for f in non_iso):
            notes.append(f"{path}: numeric dates -- confirm month/day order ({non_iso})")
    else:
        strings = [str(v).strip() for v in present]
        counts = Counter(strings)
        if (
            len(strings) >= ENUM_MIN_SAMPLES
            and len(counts) <= ENUM_MAX_VALUES
            and len(counts) / len(strings) <= ENUM_MAX_DISTINCT_RATIO
        ):
            out["type"] = "enum"
            out["values"] = sorted(counts)
            notes.append(
                f"{path}: enum inferred from {len(counts)} distinct values -- add any missing values"
            )
        else:
            out["type"] = "string"
            out["match"] = "normalized"
    out["nullable"] = nullable
    return out


def infer_spec(
    rows: list[dict[str, Any]], *, name: str, label_field: str, input_field: str, split_paths: dict[str, str]
) -> tuple[dict[str, Any], list[str]]:
    labels = []
    for i, r in enumerate(rows, 1):
        lab = r.get(label_field)
        if not isinstance(lab, dict):
            raise DataError(f"row {i}: {label_field!r} is not an object")
        if not isinstance(r.get(input_field), str):
            raise DataError(f"row {i}: {input_field!r} is not a string")
        labels.append(lab)
    if not labels:
        raise DataError("no rows to infer from")
    keys: list[str] = []
    for lab in labels:
        keys += [k for k in lab if k not in keys]
    notes: list[str] = []
    fields: dict[str, Any] = {}
    for key in keys:
        values = [lab.get(key) for lab in labels]
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, list) for v in present):
            items = [x for v in present for x in v]
            if items and all(isinstance(x, dict) for x in items):
                sub_keys: list[str] = []
                for it in items:
                    sub_keys += [k for k in it if k not in sub_keys]
                sub = {}
                for sk in sub_keys:
                    s = _infer_scalar([it.get(sk) for it in items], notes, f"{key}[].{sk}")
                    if s["type"] == "string":
                        s["match"] = "fuzzy"
                        s["threshold"] = 0.85
                    sub[sk] = s
                fields[key] = {"type": "object_list", "fields": sub}
            else:
                item = _infer_scalar(items, notes, f"{key}[]")
                item.pop("nullable", None)
                fields[key] = {"type": "list", "items": item}
            continue
        if present and any(isinstance(v, (list, dict)) for v in present):
            raise DataError(f"field {key!r} mixes nested and scalar values; fix the labels first")
        fields[key] = _infer_scalar(values, notes, key)
    spec = {
        "spec_version": 1,
        "name": name,
        "description": f"Draft inferred from {len(rows)} labelled rows. REVIEW every field before training.",
        "instructions": "Extract the fields below from the document.",
        "fields": fields,
        "output": {"reasoning": True},
        "scoring": {"mode": "weighted_fields"},
        "data": {"splits": split_paths, "input_field": input_field, "label_field": label_field},
    }
    return spec, notes


def init_spec(
    data_path: str | Path,
    out_path: str | Path,
    *,
    name: str,
    input_field: str = "text",
    label_field: str = "label",
    eval_path: str | Path | None = None,
    force: bool = False,
) -> list[str]:
    data_path, out_path = Path(data_path), Path(out_path)
    if out_path.exists() and not force:
        raise DataError(f"{out_path} exists; pass --force to overwrite")
    rows = []
    with data_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DataError(f"{data_path}:{i}: {e}") from e
    out_dir = out_path.resolve().parent

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(out_dir))
        except ValueError:
            return str(p.resolve())

    splits = {"train": rel(data_path)}
    if eval_path is not None:
        splits["eval"] = rel(Path(eval_path))
    spec, notes = infer_spec(
        rows, name=name, label_field=label_field, input_field=input_field, split_paths=splits
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Draft extraction spec generated by `grpo spec init`. Review, then run `grpo spec check`.\n"
    header += "".join(f"# REVIEW: {n}\n" for n in notes)
    out_path.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return notes
