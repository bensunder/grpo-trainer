"""Structured extraction: documents in, a typed JSON object out, graded field by field per a spec file.

Data: one JSONL file per split, each line `{"<input_field>": "...", "<label_field>": {...}}` (plus an
optional id). Labels are validated and canonicalised at load time; the canonical label is stored as a
lossless JSON string in the `ground_truth` column so it travels through TRL unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..errors import ConfigError, DataError
from ..extraction.fields import LabelError
from ..extraction.grader import Grader
from ..extraction.prompt import build_system_prompt
from ..extraction.scorers import ExtractionCorrectnessScorer, ExtractionFormatScorer
from ..extraction.spec import ExtractionSpec, load_spec
from .base import Example, Prediction, Task

log = logging.getLogger(__name__)

_MAX_REPORTED_ERRORS = 20


def _is_blank_id(value: Any) -> bool:
    return value is None or isinstance(value, (dict, list)) or not str(value).strip()


class ExtractionTask(Task):
    name = "extraction"

    def __init__(self, spec: str | Path | ExtractionSpec) -> None:
        self.spec = spec if isinstance(spec, ExtractionSpec) else load_spec(spec)
        self.grader = Grader(self.spec)
        self.default_system_prompt = build_system_prompt(self.spec)

    # ------------------------------------------------------------- data

    def load(self, split: str, limit: int | None = None) -> list[Example]:
        if self.spec.data is None:
            raise ConfigError(f"spec {self.spec.name!r} has no data section")
        data = self.spec.data
        path = self.spec.split_path(split)
        if not path.is_file():
            raise DataError(f"split {split!r}: file not found: {path}")
        examples: list[Example] = []
        errors: list[str] = []
        n_errors = too_long = 0
        seen: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                where = f"{path.name}:{line_no}"
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise LabelError("line is not a JSON object")
                    text = row.get(data.input_field)
                    if not isinstance(text, str) or not text.strip():
                        raise LabelError(f"missing or empty input field {data.input_field!r}")
                    if data.label_field not in row:
                        raise LabelError(f"missing label field {data.label_field!r}")
                    if data.id_field:
                        if data.id_field not in row or _is_blank_id(row[data.id_field]):
                            raise LabelError(f"missing id field {data.id_field!r}")
                        ex_id = str(row[data.id_field]).strip()
                    else:
                        ex_id = f"{split}-{line_no}"
                    canonical = self.grader.normalize_label(row[data.label_field])
                except (json.JSONDecodeError, LabelError) as e:
                    n_errors += 1
                    if len(errors) < _MAX_REPORTED_ERRORS:
                        errors.append(f"{where}: {e}")
                    continue
                if ex_id in seen:
                    raise DataError(f"{where}: duplicate id {ex_id!r} in split {split!r}")
                seen.add(ex_id)
                text = text.strip()
                if len(text) > data.max_input_chars:
                    too_long += 1
                    continue
                examples.append(
                    Example(id=ex_id, question=text, ground_truth=self.grader.to_ground_truth(canonical))
                )
                if limit is not None and len(examples) >= limit:
                    break
        if n_errors:
            detail = "\n  ".join(errors) + (
                f"\n  ... and {n_errors - len(errors)} more" if n_errors > len(errors) else ""
            )
            if data.on_invalid_label == "error":
                raise DataError(
                    f"split {split!r}: {n_errors} invalid rows (run `grpo spec check`):\n  {detail}"
                )
            log.warning("split %s: skipped %d invalid rows:\n  %s", split, n_errors, detail)
        if too_long:
            log.warning(
                "split %s: dropped %d inputs longer than max_input_chars=%d",
                split,
                too_long,
                data.max_input_chars,
            )
        if not examples:
            raise DataError(f"split {split!r} ({path}) produced no usable examples")
        return examples

    # ------------------------------------------------------------- grading hooks

    def correctness_scorer(self, params: dict[str, Any]) -> ExtractionCorrectnessScorer:
        return ExtractionCorrectnessScorer(self.grader, **params)

    def format_scorer(self, params: dict[str, Any]) -> ExtractionFormatScorer:
        return ExtractionFormatScorer(self.grader, **params)

    def lenient_correct(self, text: str, ground_truth: str) -> bool:
        obj = self.grader.parse_lenient(text)
        return obj is not None and self.grader.score(obj, self.grader.from_ground_truth(ground_truth)).exact

    def eval_detail(self, text: str, ground_truth: str) -> dict[str, Any]:
        parsed = self.grader.parse_answer(text)
        doc = self.grader.score(parsed.obj, self.grader.from_ground_truth(ground_truth))
        return {
            "parse_error": parsed.error,
            "field_score": round(doc.field_score, 6),
            "fields": {
                name: {"score": round(r.score, 6), "status": r.status} for name, r in doc.fields.items()
            },
        }

    def grading_fingerprint(self) -> str:
        return self.spec.fingerprint()

    def parse_prediction(self, text: str) -> Prediction:
        errors = self.grader.contract_errors(text)
        parsed = self.grader.parse_answer(text)
        if parsed.obj is None:
            return Prediction(
                output=None, valid=False, errors=tuple(errors or [f"no JSON object ({parsed.error})"])
            )
        output, coercion_problems = self.grader.canonical_output(parsed.obj)
        all_errors = tuple(errors + [p for p in coercion_problems if p not in errors])
        return Prediction(output=output, valid=not errors, errors=all_errors)

    def ideal_output(self, ground_truth: str) -> dict[str, Any]:
        """What a perfect model returns for this label (smoke tests, oracle checks)."""
        return self.grader.render_ideal(self.grader.from_ground_truth(ground_truth))
