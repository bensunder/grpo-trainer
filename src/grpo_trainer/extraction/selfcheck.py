"""`grpo spec check`: prove a spec + dataset is trainable before any GPU time is spent.

1. Every label in every split is validated and canonicalised (all errors listed, not just the first).
2. Splits are checked for duplicate ids and for inputs shared between splits (contamination).
3. Per-field statistics flag fields that carry no learning signal (always null, constant).
4. Self-tests run the compiled reward on synthetic completions for every example:
   - oracle: the perfect answer must score correctness 1.0, format 1.0 and lenient-exact;
   - mutations: corrupting each field (wrong value, dropped value, invented value, dropped list item)
     must lower the score -- a mutation that still scores 1.0 means the matcher is too loose;
   - attacks: empty, garbage, non-object JSON, duplicate keys, extra keys and fenced output must not
     earn credit they shouldn't.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..tasks.base import content_hash
from .fields import LabelError, scalar_match
from .grader import Grader
from .prompt import build_system_prompt
from .scorers import ExtractionCorrectnessScorer, ExtractionFormatScorer
from .spec import ExtractionSpec, ListField, ObjectListField, ScalarField, load_spec

ORACLE_THINK = "Located every requested field in the document and checked each value against the text."
MAX_LISTED = 15
MAX_SELFTEST_EXAMPLES = 2000
CHARS_PER_TOKEN = 3.5  # rough, for the prompt-length estimate only


@dataclass
class SplitReport:
    name: str
    path: str
    rows: int = 0
    valid: int = 0
    invalid: int = 0
    too_long: int = 0
    duplicate_ids: int = 0
    errors: list[str] = field(default_factory=list)
    max_input_chars: int = 0
    mean_input_chars: float = 0.0


@dataclass
class CheckReport:
    spec_name: str
    spec_fingerprint: str
    system_prompt_chars: int
    splits: list[SplitReport] = field(default_factory=list)
    contamination: dict[str, int] = field(default_factory=dict)
    field_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    selftest: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "system_prompt_chars": self.system_prompt_chars,
            "splits": [s.__dict__ for s in self.splits],
            "contamination": self.contamination,
            "field_stats": self.field_stats,
            "selftest": self.selftest,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ------------------------------------------------------------------ loading (collect, don't raise)


def _scan_split(
    spec: ExtractionSpec, grader: Grader, split: str
) -> tuple[SplitReport, list[tuple[str, str, dict[str, Any]]]]:
    assert spec.data is not None
    data = spec.data
    path = spec.split_path(split)
    rep = SplitReport(name=split, path=str(path))
    rows: list[tuple[str, str, dict[str, Any]]] = []
    if not path.is_file():
        rep.errors.append(f"file not found: {path}")
        return rep, rows
    seen: set[str] = set()
    lengths: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rep.rows += 1
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
                    if data.id_field not in row or not str(row[data.id_field]).strip():
                        raise LabelError(f"missing id field {data.id_field!r}")
                    ex_id = str(row[data.id_field]).strip()
                else:
                    ex_id = f"{split}-{line_no}"
                canonical = grader.normalize_label(row[data.label_field])
            except (json.JSONDecodeError, LabelError) as e:
                rep.invalid += 1
                if len(rep.errors) < MAX_LISTED:
                    rep.errors.append(f"{where}: {e}")
                continue
            if ex_id in seen:
                rep.duplicate_ids += 1
                if len(rep.errors) < MAX_LISTED:
                    rep.errors.append(f"{where}: duplicate id {ex_id!r}")
                continue
            seen.add(ex_id)
            text = text.strip()
            lengths.append(len(text))
            if len(text) > data.max_input_chars:
                rep.too_long += 1
                continue
            rep.valid += 1
            rows.append((ex_id, text, canonical))
    if lengths:
        rep.max_input_chars = max(lengths)
        rep.mean_input_chars = sum(lengths) / len(lengths)
    return rep, rows


# ------------------------------------------------------------------ field statistics


def _hashable(v: Any) -> Any:
    return json.dumps(v, sort_keys=True, default=str)


def _field_stats(spec: ExtractionSpec, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    n = len(rows)
    for name, f in spec.fields.items():
        values = [r.get(name) for r in rows]
        present = [v for v in values if v is not None]
        st: dict[str, Any] = {"type": f.type, "null_rate": (n - len(present)) / n if n else 0.0}
        if isinstance(f, (ListField, ObjectListField)):
            lens = [len(v) for v in present]
            st["mean_items"] = sum(lens) / len(lens) if lens else 0.0
            st["max_items"] = max(lens) if lens else 0
        else:
            counts = Counter(_hashable(v) for v in present)
            st["distinct"] = len(counts)
            if f.type == "enum":
                used = {json.loads(k) for k in counts}
                st["unused_enum_values"] = [v for v in f.values if v not in used]
        stats[name] = st
    return stats


# ------------------------------------------------------------------ self-tests


def _wrap(spec: ExtractionSpec, answer: str) -> str:
    if spec.output.reasoning:
        return f"<think>{ORACLE_THINK}</think>\n<answer>{answer}</answer>"
    return f"<answer>{answer}</answer>"


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _wrong_scalar(f: ScalarField, gt: Any, rng: random.Random, pool: list[Any] | None = None) -> Any:
    """A JSON value that is wrong for `gt` by a margin no reasonable matcher should accept.

    Strings: another real value from the same column when one exists (the realistic confusion: the
    wrong vendor, the wrong line description). Otherwise a suffix, which exact/normalized must reject.
    """
    t = f.type
    if t == "string":
        others = [v for v in (pool or []) if v != gt and not scalar_match(f, v, gt)]
        if others:
            return rng.choice(others)
        return f"{gt} ZZ{rng.randint(10, 99)}"
    if t == "number":
        # Wrong by 50% + 1 in absolute terms, independent of the configured tolerance: a tolerance
        # that still accepts this is too loose, which is exactly what the check must surface.
        return float(gt + max(Decimal("1"), abs(gt)) * Decimal("0.5") + 1)
    if t == "integer":
        return int(gt) + 1 + int(abs(gt) * 0.5)
    if t == "date":
        return (gt + timedelta(days=397)).isoformat()
    if t == "enum":
        others = [v for v in f.values if v != gt]
        return others[0] if others else None
    if t == "boolean":
        return not gt
    raise AssertionError(t)


def _plausible_scalar(f: ScalarField) -> Any:
    return {
        "string": "UNKNOWN-VALUE",
        "number": 123.45,
        "integer": 7,
        "date": "2001-02-03",
        "enum": f.values[0] if f.values else "x",
        "boolean": True,
    }[f.type]


def _string_pools(spec: ExtractionSpec, rows: list[dict[str, Any]], cap: int = 200) -> dict[str, list[str]]:
    """Distinct observed values per string field path ("vendor", "line_items.description")."""
    pools: dict[str, set[str]] = {}
    for r in rows:
        for name, f in spec.fields.items():
            if isinstance(f, ScalarField) and f.type == "string" and r.get(name) is not None:
                pools.setdefault(name, set()).add(r[name])
            elif isinstance(f, ObjectListField):
                for item in r.get(name) or []:
                    for k, sub in f.fields.items():
                        if sub.type == "string" and item.get(k) is not None:
                            pools.setdefault(f"{name}.{k}", set()).add(item[k])
    return {k: sorted(v)[:cap] for k, v in pools.items()}


def _mutations(
    spec: ExtractionSpec,
    ideal: dict[str, Any],
    gt: dict[str, Any],
    rng: random.Random,
    pools: dict[str, list[str]] | None = None,
):
    """Yield (field, kind, mutated_object). Each mutation must reduce that field's score."""
    pools = pools or {}
    for name, f in spec.fields.items():
        g = gt.get(name)
        if isinstance(f, ScalarField):
            if g is None:
                yield name, "invented", {**ideal, name: _plausible_scalar(f)}
            else:
                yield name, "dropped", {**ideal, name: None}
                wrong = _wrong_scalar(f, g, rng, pools.get(name))
                if wrong is not None:
                    yield name, "wrong_value", {**ideal, name: wrong}
        else:
            items = ideal.get(name) or []
            if not items:
                filler = (
                    [_plausible_scalar(f.items)]
                    if isinstance(f, ListField)
                    else [{k: _plausible_scalar(s) for k, s in f.fields.items()}]
                )
                yield name, "invented", {**ideal, name: filler}
            else:
                yield name, "dropped", {**ideal, name: []}
                if len(items) > 1:
                    yield name, "dropped_item", {**ideal, name: items[:-1]}
                if isinstance(f, ObjectListField):
                    sub_name, sub = next(iter(f.fields.items()))
                    first = dict(items[0])
                    sub_gt = (g or [{}])[0].get(sub_name)
                    if sub_gt is not None:
                        wrong = _wrong_scalar(sub, sub_gt, rng, pools.get(f"{name}.{sub_name}"))
                        if wrong is not None:
                            first[sub_name] = wrong
                            yield name, "wrong_item_value", {**ideal, name: [first, *items[1:]]}


def _selftest(spec: ExtractionSpec, grader: Grader, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    from ..tasks.extraction import ExtractionTask

    correctness = ExtractionCorrectnessScorer(grader)
    fmt = ExtractionFormatScorer(grader)
    task = ExtractionTask(spec)
    rng = random.Random(seed)
    sample = rows if len(rows) <= MAX_SELFTEST_EXAMPLES else rng.sample(rows, MAX_SELFTEST_EXAMPLES)
    pools = _string_pools(spec, rows)

    def score(text: str, gt_json: str) -> tuple[float, float]:
        row = {"ground_truth": gt_json}
        return correctness(text, completion_ids=None, row=row), fmt(text, completion_ids=None, row=row)

    oracle_fail: list[str] = []
    mutation_total: Counter[str] = Counter()
    mutation_escaped: Counter[str] = Counter()
    escaped_examples: dict[str, str] = {}
    out_of_bounds = 0
    for i, gt in enumerate(sample):
        gt_json = grader.to_ground_truth(gt)
        ideal = grader.render_ideal(gt)
        text = _wrap(spec, _dump(ideal))
        c, fm = score(text, gt_json)
        lenient = task.lenient_correct(text, gt_json)
        if (c != 1.0 or fm != 1.0 or not lenient) and len(oracle_fail) < MAX_LISTED:
            oracle_fail.append(f"example #{i}: correctness={c} format={fm} lenient={lenient}")
        for name, kind, mutated in _mutations(spec, ideal, gt, rng, pools):
            key = f"{name}:{kind}"
            mutation_total[key] += 1
            mc, _ = score(_wrap(spec, _dump(mutated)), gt_json)
            if not 0.0 <= mc <= 1.0:
                out_of_bounds += 1
            if mc >= 1.0:
                mutation_escaped[key] += 1
                escaped_examples.setdefault(key, _dump(mutated.get(name))[:120])

    attacks: dict[str, dict[str, float]] = {}
    attack_fail: list[str] = []
    if sample:
        gt = sample[0]
        gt_json = grader.to_ground_truth(gt)
        ideal = grader.render_ideal(gt)
        first_key = next(iter(ideal))
        body = _dump(ideal)
        cases = {
            "empty": ("", 0.0, 0.0),
            "garbage": ("I cannot help with that.", 0.0, 0.0),
            "json_array": (_wrap(spec, _dump([ideal])), 0.0, 0.0),
            "duplicate_key": (
                _wrap(spec, body[:-1] + f", {json.dumps(first_key)}: null}}" if len(ideal) else body),
                0.0,
                0.0,
            ),
            "extra_key": (_wrap(spec, _dump({**ideal, "zz_extra": 1})), None, 0.0),
            "code_fence": (_wrap(spec, f"```json\n{body}\n```"), None, 0.0),
            "no_tags": (body, 0.0, 0.0),
        }
        for label, (text, want_c_max, want_f_max) in cases.items():
            c, fm = score(text, gt_json)
            attacks[label] = {"correctness": c, "format": fm}
            if want_c_max is not None and c > want_c_max:
                attack_fail.append(f"{label}: correctness {c} (expected <= {want_c_max})")
            if fm > want_f_max:
                attack_fail.append(f"{label}: format {fm} (expected <= {want_f_max})")

    mutations = {
        key: {"tested": mutation_total[key], "still_scored_1": mutation_escaped[key]}
        for key in sorted(mutation_total)
    }
    return {
        "examples": len(sample),
        "oracle_failures": oracle_fail,
        "mutations": mutations,
        "escaped_examples": escaped_examples,
        "attacks": attacks,
        "attack_failures": attack_fail,
        "out_of_bounds_scores": out_of_bounds,
    }


# ------------------------------------------------------------------ entrypoint


def check_spec(path: str | Path, *, eval_split: str | None = None, seed: int = 0) -> CheckReport:
    spec = load_spec(path)
    grader = Grader(spec)
    prompt = build_system_prompt(spec)
    rep = CheckReport(
        spec_name=spec.name, spec_fingerprint=spec.fingerprint(), system_prompt_chars=len(prompt)
    )
    if spec.data is None:
        rep.errors.append("spec has no data section; nothing to check")
        return rep

    by_split: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for split in spec.data.splits:
        srep, rows = _scan_split(spec, grader, split)
        rep.splits.append(srep)
        by_split[split] = rows
        if srep.invalid and spec.data.on_invalid_label == "error":
            rep.errors.append(
                f"{split}: {srep.invalid} invalid rows (on_invalid_label=error will refuse to load)"
            )
        elif srep.invalid:
            rep.warnings.append(f"{split}: {srep.invalid} invalid rows will be skipped")
        if srep.duplicate_ids:
            rep.errors.append(f"{split}: {srep.duplicate_ids} duplicate ids")
        if srep.too_long:
            rep.warnings.append(f"{split}: {srep.too_long} inputs exceed max_input_chars and will be dropped")
        if not srep.valid and not any("file not found" in e for e in srep.errors):
            rep.errors.append(f"{split}: no usable examples")
        elif not srep.valid:
            rep.errors.append(f"{split}: {srep.errors[0]}")

    names = list(by_split)
    held = eval_split if eval_split in by_split else next((s for s in names if s in ("eval", "test")), None)
    if held is not None:
        held_hashes = {content_hash(text) for _, text, _ in by_split[held]}
        for split in names:
            if split == held:
                continue
            overlap = sum(1 for _, text, _ in by_split[split] if content_hash(text) in held_hashes)
            rep.contamination[f"{split}->{held}"] = overlap
            if overlap:
                rep.warnings.append(
                    f"{overlap} {split} inputs also appear in {held}; "
                    "they are dropped from training at load time"
                )
    else:
        rep.warnings.append("no eval/test split declared; contamination not checked")

    train_rows = [c for s in names if s != held for _, _, c in by_split[s]] or [
        c for s in names for _, _, c in by_split[s]
    ]
    if train_rows:
        rep.field_stats = _field_stats(spec, train_rows)
        for name, st in rep.field_stats.items():
            if st["null_rate"] == 1.0:
                rep.warnings.append(f"field {name}: always null in training labels (no signal)")
            elif st.get("distinct") == 1 and st["null_rate"] == 0.0:
                rep.warnings.append(f"field {name}: constant in training labels (no signal)")
            if st.get("unused_enum_values"):
                rep.warnings.append(
                    f"field {name}: enum values never used in training {st['unused_enum_values']}"
                )

    all_rows = [c for s in names for _, _, c in by_split[s]]
    if all_rows:
        st = _selftest(spec, grader, all_rows, seed)
        rep.selftest = st
        if st["oracle_failures"]:
            rep.errors.append(
                f"oracle self-test failed on {len(st['oracle_failures'])}+ examples (see selftest)"
            )
        if st["attack_failures"]:
            rep.errors.append("reward attack self-test failed: " + "; ".join(st["attack_failures"]))
        if st["out_of_bounds_scores"]:
            rep.errors.append(f"{st['out_of_bounds_scores']} rewards outside [0, 1]")
        for key, m in st["mutations"].items():
            if m["still_scored_1"]:
                rate = m["still_scored_1"] / m["tested"]
                rep.warnings.append(
                    f"mutation {key} still scored 1.0 on {m['still_scored_1']}/{m['tested']} ({rate:.0%}): "
                    "the matcher may be too loose (fuzzy threshold / tolerance)"
                )

    max_chars = max((s.max_input_chars for s in rep.splits), default=0)
    est = (rep.system_prompt_chars + max_chars) / CHARS_PER_TOKEN
    rep.selftest["max_prompt_tokens_estimate"] = round(est)
    return rep


def format_report(rep: CheckReport) -> str:
    lines = [
        f"spec {rep.spec_name} (fingerprint {rep.spec_fingerprint}), "
        f"system prompt {rep.system_prompt_chars} chars"
    ]
    for s in rep.splits:
        lines.append(
            f"  split {s.name}: {s.valid}/{s.rows} usable, {s.invalid} invalid, "
            f"{s.duplicate_ids} duplicate ids, "
            f"{s.too_long} too long; input chars mean {s.mean_input_chars:.0f} max {s.max_input_chars}"
        )
        lines += [f"    ! {e}" for e in s.errors]
    for pair, n in rep.contamination.items():
        lines.append(f"  contamination {pair}: {n}")
    if rep.field_stats:
        lines.append("  fields:")
        for name, st in rep.field_stats.items():
            extra = (
                f"distinct {st['distinct']}"
                if "distinct" in st
                else f"items mean {st['mean_items']:.1f} max {st['max_items']}"
            )
            lines.append(f"    {name:<24} {st['type']:<12} null {st['null_rate']:.0%}  {extra}")
    st = rep.selftest
    if st.get("examples"):
        escaped = sum(m["still_scored_1"] for m in st["mutations"].values())
        tested = sum(m["tested"] for m in st["mutations"].values())
        lines.append(
            f"  self-test on {st['examples']} examples: oracle failures {len(st['oracle_failures'])}, "
            f"mutations caught {tested - escaped}/{tested}, attack failures {len(st['attack_failures'])}"
        )
    if "max_prompt_tokens_estimate" in st:
        lines.append(
            f"  longest prompt ~{st['max_prompt_tokens_estimate']} tokens "
            "(estimate; set vllm.max_model_length above prompt + completion)"
        )
    lines += [f"  WARN {w}" for w in rep.warnings]
    lines += [f"  FAIL {e}" for e in rep.errors]
    lines.append("spec check: " + ("OK" if rep.ok else "FAILED"))
    return "\n".join(lines)
