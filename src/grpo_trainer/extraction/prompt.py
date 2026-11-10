"""System prompt compiled from the spec. Deterministic: same spec -> byte-identical prompt.

The same text is used for profiling, training, eval, the frontier baseline and the serving gateway, so
there is no train/serve prompt skew. Its sha256 is recorded in every eval report and in the bundle.
"""

from __future__ import annotations

import json

from .spec import ExtractionSpec, ListField, ObjectListField, ScalarField


def _scalar_shape(f: ScalarField) -> str:
    t = f.type
    if t == "string":
        shape = "string"
    elif t == "number":
        shape = "number (a JSON number: no currency symbols, no thousands separators, no quotes)"
    elif t == "integer":
        shape = "integer"
    elif t == "date":
        shape = 'date as a "YYYY-MM-DD" string'
    elif t == "enum":
        shape = "one of " + ", ".join(json.dumps(v) for v in f.values)
    else:
        shape = "true or false"
    return shape + (", or null if not present" if f.nullable else "; always required")


def _field_line(name: str, f: ScalarField | ListField | ObjectListField) -> str:
    if isinstance(f, ListField):
        shape = f"array of {_scalar_shape(f.items).split(', or null')[0]}; [] if none"
    elif isinstance(f, ObjectListField):
        sub = "; ".join(f"{k}: {_scalar_shape(s)}" for k, s in f.fields.items())
        shape = f"array of objects with exactly these keys ({sub}); [] if none"
    else:
        shape = _scalar_shape(f)
    desc = f" -- {f.description.strip()}" if f.description.strip() else ""
    return f"- {name}: {shape}{desc}"


def build_system_prompt(spec: ExtractionSpec) -> str:
    lines = ["You extract structured data from documents accurately and never guess."]
    if spec.instructions.strip():
        lines += ["", spec.instructions.strip()]
    lines += ["", "Return one JSON object with exactly these keys, in this order:"]
    lines += [_field_line(name, f) for name, f in spec.fields.items()]
    lines += [
        "",
        "Use null (or [] for arrays) when a value does not appear in the document. "
        "Copy identifiers exactly as written. Do not add keys.",
        "",
    ]
    if spec.output.reasoning:
        lines.append(
            "First reason briefly inside <think> and </think>: locate each value and resolve any ambiguity. "
            "Then output only the JSON object inside <answer> and </answer>, with no code fences. "
            "Output nothing after </answer>."
        )
    else:
        lines.append(
            "Output only the JSON object inside <answer> and </answer>, "
            "with no code fences and no other text."
        )
    return "\n".join(lines)
