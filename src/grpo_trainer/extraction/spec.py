"""Extraction spec: the customer-facing contract that compiles into data loading, prompts and rewards.

One YAML file per use case. It declares the fields to extract, how each is matched against the label,
and where the labelled data lives. Everything is validated up front (unknown keys, options that don't
apply to a field type, unparseable date formats, duplicate enum values), so a bad spec fails in
`grpo spec check` -- not three hours into a GPU run.

    spec_version: 1
    name: invoice-extraction
    instructions: Extract header fields and line items from supplier invoices.
    fields:
      invoice_number: {type: string, match: normalized, nullable: false}
      invoice_date:   {type: date, input_formats: ["%m/%d/%Y"]}
      total_amount:   {type: number, tolerance: 0.01, weight: 2}
      currency:       {type: enum, values: [USD, EUR, GBP]}
      line_items:
        type: object_list
        fields:
          description: {type: string, match: fuzzy, threshold: 0.85}
          amount:      {type: number, tolerance: 0.01}
    data:
      splits: {train: data/train.jsonl, eval: data/eval.jsonl}
      input_field: text
      label_field: label
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from ..errors import ConfigError

_FIELD_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"
ScalarType = Literal["string", "number", "integer", "date", "enum", "boolean"]

# Options each scalar type accepts beyond the common ones. Setting an option that doesn't apply to the
# type is an error: `tolerance` on a string field is a spec bug, not something to ignore silently.
_COMMON = {"type", "description", "nullable", "weight"}
_TYPE_OPTIONS: dict[str, set[str]] = {
    "string": {"match", "threshold"},
    "number": {"tolerance", "rel_tolerance"},
    "integer": set(),
    "date": {"input_formats"},
    "enum": {"values", "case_sensitive"},
    "boolean": set(),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScalarField(_Strict):
    type: ScalarType
    description: str = ""
    nullable: bool = True
    weight: float = Field(default=1.0, gt=0.0)
    # string
    match: Literal["exact", "normalized", "fuzzy"] = "normalized"
    threshold: float = Field(default=0.9, gt=0.0, le=1.0)  # fuzzy similarity cut-off
    # number
    tolerance: float = Field(default=0.0, ge=0.0)  # absolute
    rel_tolerance: float = Field(default=0.0, ge=0.0)  # relative to |label|
    # date: strptime formats accepted in labels (and, leniently, in predictions). ISO is always accepted.
    input_formats: list[str] = Field(default_factory=list)
    # enum
    values: list[str] = Field(default_factory=list)
    case_sensitive: bool = False

    @model_validator(mode="after")
    def _options_apply(self) -> ScalarField:
        allowed = _COMMON | _TYPE_OPTIONS[self.type]
        stray = sorted(self.model_fields_set - allowed)
        if stray:
            raise ValueError(f"option(s) {stray} do not apply to type {self.type!r}")
        if self.type == "enum":
            if not self.values:
                raise ValueError("enum requires a non-empty `values` list")
            keys = [v if self.case_sensitive else v.casefold() for v in self.values]
            if len(keys) != len(set(keys)):
                raise ValueError(
                    f"enum values must be unique{'' if self.case_sensitive else ' (case-insensitive)'}"
                )
            if any(not v.strip() for v in self.values):
                raise ValueError("enum values must be non-empty strings")
        if self.type == "date":
            for fmt in self.input_formats:
                _check_date_format(fmt)
        return self


class ListField(_Strict):
    """Unordered list of scalars, scored by F1 (e.g. tags, SKUs, cited clause numbers)."""

    type: Literal["list"]
    description: str = ""
    nullable: bool = True  # null and [] are equivalent for scoring
    weight: float = Field(default=1.0, gt=0.0)
    items: ScalarField

    @field_validator("items")
    @classmethod
    def _items_not_nullable(cls, v: ScalarField) -> ScalarField:
        if v.nullable and "nullable" in v.model_fields_set:
            raise ValueError("list items cannot be nullable; omit the item instead")
        return v.model_copy(update={"nullable": False})


class ObjectListField(_Strict):
    """Unordered list of records (e.g. invoice line items), aligned greedily and scored by soft F1."""

    type: Literal["object_list"]
    description: str = ""
    nullable: bool = True
    weight: float = Field(default=1.0, gt=0.0)
    fields: dict[str, ScalarField] = Field(min_length=1)

    @field_validator("fields")
    @classmethod
    def _names(cls, v: dict[str, ScalarField]) -> dict[str, ScalarField]:
        _check_names(v)
        return v


FieldSpec = Annotated[ScalarField | ListField | ObjectListField, Field(discriminator="type")]


class DataSpec(_Strict):
    format: Literal["jsonl"] = "jsonl"
    splits: dict[str, str] = Field(min_length=1)  # split name -> path, relative to the spec file
    input_field: str = "text"
    label_field: str = "label"
    id_field: str | None = None  # None -> ids are "<split>-<line number>"
    # Longer inputs are dropped (and counted) rather than crashing vLLM on an over-length prompt.
    max_input_chars: int = Field(default=12_000, ge=100)
    on_invalid_label: Literal["error", "skip"] = "error"


class OutputSpec(_Strict):
    # True: <think>..</think><answer>{json}</answer>. False: <answer>{json}</answer> only -- fewer
    # output tokens at serving time, at the cost of no visible reasoning.
    reasoning: bool = True


class ScoringSpec(_Strict):
    # weighted_fields: reward = weighted mean of field scores (partial credit, denser GRPO signal).
    # all_or_nothing: reward 1 only when every field is right.
    mode: Literal["weighted_fields", "all_or_nothing"] = "weighted_fields"


class ExtractionSpec(_Strict):
    spec_version: Literal[1] = 1
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    instructions: str = ""  # domain guidance placed in the system prompt
    fields: dict[str, FieldSpec] = Field(min_length=1)
    output: OutputSpec = Field(default_factory=OutputSpec)
    scoring: ScoringSpec = Field(default_factory=ScoringSpec)
    data: DataSpec | None = None

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @field_validator("fields")
    @classmethod
    def _names(cls, v: dict[str, Any]) -> dict[str, Any]:
        _check_names(v)
        return v

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def split_path(self, split: str) -> Path:
        if self.data is None:
            raise ConfigError(f"spec {self.name!r} has no `data` section; cannot load split {split!r}")
        rel = self.data.splits.get(split)
        if rel is None:
            raise ConfigError(
                f"spec {self.name!r} has no split {split!r}; declared: {sorted(self.data.splits)}"
            )
        p = Path(rel)
        return p if p.is_absolute() else (self._base_dir / p)

    def fingerprint(self) -> str:
        """Hash of everything that defines grading and the prompt. Excludes data paths and prose."""
        payload = self.model_dump(mode="json", exclude={"data", "description"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _check_names(fields: dict[str, Any]) -> None:
    import re

    bad = [k for k in fields if not re.match(_FIELD_NAME, k)]
    if bad:
        raise ValueError(f"field names must be identifiers ({_FIELD_NAME}): {bad}")


def _check_date_format(fmt: str) -> None:
    if "%" not in fmt:
        raise ValueError(f"date input format {fmt!r} has no strptime directives")
    try:
        rendered = date(2024, 12, 31).strftime(fmt)
        datetime.strptime(rendered, fmt)
    except ValueError as e:
        raise ValueError(f"date input format {fmt!r} does not round-trip: {e}") from None


def load_spec(path: str | Path) -> ExtractionSpec:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"extraction spec not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in spec {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"top level of spec {path} must be a mapping")
    try:
        spec = ExtractionSpec.model_validate(raw)
    except ValidationError as e:
        lines = [f"  {'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()]
        raise ConfigError(f"invalid extraction spec {path}:\n" + "\n".join(lines)) from None
    spec._base_dir = path.resolve().parent
    return spec
