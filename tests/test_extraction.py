"""Structured extraction: spec validation, field semantics, grading, task loading, self-check, infer."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from grpo_trainer.errors import ConfigError, DataError
from grpo_trainer.extraction.fields import (
    INVALID,
    LabelError,
    coerce_prediction,
    normalize_label,
    score_field,
    strict_type_error,
)
from grpo_trainer.extraction.grader import Grader, loads_strict
from grpo_trainer.extraction.infer import infer_spec, init_spec
from grpo_trainer.extraction.prompt import build_system_prompt
from grpo_trainer.extraction.selfcheck import check_spec
from grpo_trainer.extraction.spec import ExtractionSpec, ScalarField, load_spec
from grpo_trainer.synth.invoices import generate as synth_generate
from grpo_trainer.tasks import get_task
from grpo_trainer.tasks.extraction import ExtractionTask

ROOT = Path(__file__).resolve().parents[1]

SPEC = {
    "name": "t",
    "fields": {
        "inv": {"type": "string", "nullable": False},
        "date": {"type": "date", "input_formats": ["%m/%d/%Y"]},
        "total": {"type": "number", "tolerance": 0.01, "weight": 2},
        "cur": {"type": "enum", "values": ["USD", "EUR"]},
        "paid": {"type": "boolean"},
        "tags": {"type": "list", "items": {"type": "string"}},
        "lines": {
            "type": "object_list",
            "fields": {
                "desc": {"type": "string", "match": "fuzzy", "threshold": 0.8},
                "amt": {"type": "number", "tolerance": 0.01},
            },
        },
    },
    "data": {"splits": {"train": "train.jsonl", "eval": "eval.jsonl"}, "id_field": "id"},
}
LABEL = {
    "inv": "INV-1",
    "date": "03/14/2025",
    "total": "$1,234.50",
    "cur": "usd",
    "paid": False,
    "tags": ["a", "b"],
    "lines": [{"desc": "Widget large", "amt": 1000}, {"desc": "Bolt", "amt": 234.5}],
}
IDEAL = {
    "inv": "INV-1",
    "date": "2025-03-14",
    "total": 1234.5,
    "cur": "USD",
    "paid": False,
    "tags": ["a", "b"],
    "lines": [{"desc": "Widget large", "amt": 1000}, {"desc": "Bolt", "amt": 234.5}],
}


def wrap(obj, think=True) -> str:
    body = json.dumps(obj)
    return (
        f"<think>found every field in the text carefully</think><answer>{body}</answer>"
        if think
        else (f"<answer>{body}</answer>")
    )


@pytest.fixture
def spec() -> ExtractionSpec:
    return ExtractionSpec.model_validate(SPEC)


@pytest.fixture
def grader(spec) -> Grader:
    return Grader(spec)


@pytest.fixture
def gt_json(grader) -> str:
    return grader.to_ground_truth(grader.normalize_label(LABEL))


# ------------------------------------------------------------------ spec validation


@pytest.mark.parametrize(
    "field, msg",
    [
        ({"type": "string", "tolerance": 0.1}, "do not apply"),
        ({"type": "enum"}, "non-empty"),
        ({"type": "enum", "values": ["A", "a"]}, "unique"),
        ({"type": "date", "input_formats": ["YYYY"]}, "strptime"),
        ({"type": "number", "match": "fuzzy"}, "do not apply"),
        ({"type": "list", "items": {"type": "string", "nullable": True}}, "cannot be nullable"),
    ],
)
def test_spec_rejects_bad_fields(field, msg):
    with pytest.raises(Exception, match=msg):
        ExtractionSpec.model_validate({"name": "x", "fields": {"f": field}})


def test_spec_rejects_bad_field_names_and_unknown_keys():
    with pytest.raises(Exception, match="identifiers"):
        ExtractionSpec.model_validate({"name": "x", "fields": {"bad-name": {"type": "string"}}})
    with pytest.raises(Exception, match="Extra inputs"):
        ExtractionSpec.model_validate({"name": "x", "fields": {"f": {"type": "string"}}, "bogus": 1})


def test_load_spec_resolves_paths_relative_to_spec(tmp_path):
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "s.yaml"
    p.write_text(yaml.safe_dump(SPEC))
    s = load_spec(p)
    assert s.split_path("train") == tmp_path / "sub" / "train.jsonl"
    with pytest.raises(ConfigError, match="no split"):
        s.split_path("nope")
    with pytest.raises(ConfigError, match="not found"):
        load_spec(tmp_path / "missing.yaml")


def test_fingerprint_ignores_data_and_prose_but_not_matchers(spec):
    moved = ExtractionSpec.model_validate(
        {**SPEC, "description": "x", "data": {"splits": {"train": "other"}}}
    )
    assert moved.fingerprint() == spec.fingerprint()
    looser = json.loads(json.dumps(SPEC))
    looser["fields"]["total"]["tolerance"] = 1.0
    assert ExtractionSpec.model_validate(looser).fingerprint() != spec.fingerprint()


# ------------------------------------------------------------------ field semantics


def test_label_normalisation(spec):
    f = spec.fields
    assert normalize_label("date", f["date"], "03/14/2025").isoformat() == "2025-03-14"
    assert normalize_label("total", f["total"], "$1,234.50") == Decimal("1234.50")
    assert normalize_label("cur", f["cur"], "usd") == "USD"
    assert normalize_label("tags", f["tags"], []) is None
    assert normalize_label("total", f["total"], "") is None
    for name, bad in [
        ("date", "14/03/2025"),
        ("cur", "JPY"),
        ("total", "12 or 13"),
        ("inv", None),
        ("paid", "maybe"),
    ]:
        with pytest.raises(LabelError, match=name):
            normalize_label(name, f[name], bad)


def test_prediction_coercion_is_lenient_strict_check_is_not(spec):
    f = spec.fields
    assert coerce_prediction(f["total"], "USD 1,234.50") == Decimal("1234.50")
    assert strict_type_error("total", f["total"], "1234.50") is not None
    assert strict_type_error("total", f["total"], 1234.5) is None
    assert strict_type_error("total", f["total"], True) is not None  # bool is not a number
    assert coerce_prediction(f["cur"], "eur") == "EUR"
    assert strict_type_error("cur", f["cur"], "eur") is not None
    assert coerce_prediction(f["date"], "03/14/2025").isoformat() == "2025-03-14"
    assert strict_type_error("date", f["date"], "03/14/2025") is not None
    assert strict_type_error("date", f["date"], "2025-02-30") is not None
    assert coerce_prediction(f["total"], {"x": 1}) is INVALID
    assert strict_type_error("inv", f["inv"], None) is not None  # nullable: false


def test_null_semantics(spec):
    f = spec.fields["total"]
    assert score_field(f, True, None, None).status == "correct"
    assert score_field(f, False, None, None).status == "correct"
    assert score_field(f, True, Decimal(5), None).status == "hallucinated"
    assert score_field(f, True, None, Decimal(5)).status == "omitted"
    assert score_field(f, False, None, Decimal(5)).status == "missing_key"
    assert score_field(f, True, INVALID, Decimal(5)).status == "invalid"
    assert score_field(f, True, Decimal("5.009"), Decimal(5)).score == 1.0
    assert score_field(f, True, Decimal("5.02"), Decimal(5)).score == 0.0


def test_string_match_modes():
    exact = ScalarField(type="string", match="exact")
    norm = ScalarField(type="string", match="normalized")
    fuzzy = ScalarField(type="string", match="fuzzy", threshold=0.9)
    gt = "Acme Corp."
    assert score_field(exact, True, "acme corp", gt).score == 0.0
    assert score_field(norm, True, "  ACME   corp ", gt).score == 1.0
    assert score_field(fuzzy, True, "Acme Corpp", gt).score == 1.0
    assert score_field(fuzzy, True, "Apex Industries", gt).score == 0.0


def test_object_list_partial_credit_and_spraying_guard(spec):
    f = spec.fields["lines"]
    gt = normalize_label("lines", f, LABEL["lines"])
    pred = coerce_prediction(f, [{"desc": "Widget large", "amt": 1000}])
    r = score_field(f, True, pred, gt)
    assert r.status == "partial" and r.score == pytest.approx(2 / 3)  # P=1, R=0.5
    reordered = coerce_prediction(f, list(reversed(IDEAL["lines"])))
    assert score_field(f, True, reordered, gt).score == 1.0  # order-insensitive
    sprayed = coerce_prediction(f, IDEAL["lines"] * 30)
    assert score_field(f, True, sprayed, gt).score == 0.0


# ------------------------------------------------------------------ grading


def test_oracle_scores_one(grader, gt_json):
    doc = grader.score_text(wrap(IDEAL), gt_json)
    assert doc.exact and doc.score == 1.0
    assert grader.contract_errors(wrap(IDEAL)) == []


def test_weighted_partial_credit(grader, gt_json):
    wrong_total = {**IDEAL, "total": 9}
    doc = grader.score_text(wrap(wrong_total), gt_json)
    # total has weight 2 of a total weight 8 -> 0.75
    assert not doc.exact and doc.score == pytest.approx(0.75)
    assert doc.fields["total"].status == "wrong"


def test_all_or_nothing_mode(gt_json):
    g = Grader(ExtractionSpec.model_validate({**SPEC, "scoring": {"mode": "all_or_nothing"}}))
    doc = g.score_text(wrap({**IDEAL, "total": 9}), gt_json)
    assert doc.score == 0.0 and doc.field_score == pytest.approx(0.75)


@pytest.mark.parametrize(
    "text, error",
    [
        ("no answer here", "no_answer"),
        ("<answer>{not json}</answer>", "invalid_json"),
        ('<answer>[{"inv": 1}]</answer>', "not_object"),
        ('<answer>{"inv": "a", "inv": "b"}</answer>', "duplicate_keys"),
        ('<answer>{"x": NaN}</answer>', "invalid_json"),
    ],
)
def test_parse_errors_score_zero(grader, gt_json, text, error):
    assert grader.parse_answer(text).error == error
    assert grader.score_text(text, gt_json).score == 0.0


def test_fence_tolerated_for_score_not_contract(grader, gt_json):
    fenced = f"```json\n{json.dumps(IDEAL)}\n```"
    text = f"<think>found every field in the text carefully</think><answer>{fenced}</answer>"
    assert grader.score_text(text, gt_json).exact
    assert any("fence" in e for e in grader.contract_errors(text))


def test_lenient_finds_last_object(grader, gt_json):
    task = ExtractionTask(grader.spec)
    text = f'draft {{"inv": "nope"}} final answer: {json.dumps(IDEAL)} done'
    assert task.lenient_correct(text, gt_json)
    assert not task.lenient_correct("nothing", gt_json)


def test_reasoning_off_contract():
    g = Grader(ExtractionSpec.model_validate({**SPEC, "output": {"reasoning": False}}))
    assert g.contract_answer(wrap(IDEAL, think=False)) is not None
    assert g.contract_answer(wrap(IDEAL, think=True)) is None


def test_ground_truth_roundtrip_is_lossless(grader):
    canonical = grader.normalize_label(LABEL)
    assert grader.from_ground_truth(grader.to_ground_truth(canonical)) == canonical


def test_loads_strict():
    assert loads_strict('{"a": 1}') == ({"a": 1}, None)
    assert loads_strict('{"a": 1, "a": 2}')[1] == "duplicate_keys"
    assert loads_strict("Infinity")[1] == "invalid_json"


# ------------------------------------------------------------------ scorers and task


def test_scorers_and_prediction(spec, gt_json):
    task = ExtractionTask(spec)
    corr = task.correctness_scorer({})
    fmt = task.format_scorer({"min_think_chars": 10})
    row = {"ground_truth": gt_json}
    assert corr(wrap(IDEAL), completion_ids=None, row=row) == 1.0
    assert fmt(wrap(IDEAL), completion_ids=None, row=row) == 1.0
    lenient_types = {**IDEAL, "total": "$1,234.50"}
    assert corr(wrap(lenient_types), completion_ids=None, row=row) == 1.0  # knowledge
    assert fmt(wrap(lenient_types), completion_ids=None, row=row) == 0.0  # contract
    empty_think = f"<think></think><answer>{json.dumps(IDEAL)}</answer>"
    assert fmt(empty_think, completion_ids=None, row=row) == 0.0
    with pytest.raises(KeyError):
        corr(wrap(IDEAL), completion_ids=None, row={})

    pred = task.parse_prediction(wrap(lenient_types))
    assert not pred.valid and pred.output["total"] == 1234.5
    assert task.parse_prediction(wrap(IDEAL)).valid
    assert task.parse_prediction("garbage").output is None
    detail = task.eval_detail(wrap({**IDEAL, "cur": None}), gt_json)
    assert detail["fields"]["cur"]["status"] == "omitted"


def test_system_prompt_is_deterministic_and_lists_every_field(spec):
    p = build_system_prompt(spec)
    assert p == build_system_prompt(ExtractionSpec.model_validate(SPEC))
    for name in SPEC["fields"]:
        assert f"- {name}:" in p
    assert "<think>" in p and '"USD"' in p


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_task_load_validates_and_canonicalises(tmp_path):
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(SPEC))
    _write_jsonl(tmp_path / "train.jsonl", [{"id": "a", "text": "doc a", "label": LABEL}])
    task = get_task("extraction", spec=str(tmp_path / "s.yaml"))
    [ex] = task.load("train")
    assert ex.id == "a" and json.loads(ex.ground_truth)["total"] == "1234.5"

    _write_jsonl(tmp_path / "train.jsonl", [{"id": "a", "text": "doc", "label": {**LABEL, "cur": "JPY"}}])
    with pytest.raises(DataError, match="cur"):
        task.load("train")
    _write_jsonl(tmp_path / "train.jsonl", [{"id": "a", "text": "d", "label": LABEL}] * 2)
    with pytest.raises(DataError, match="duplicate id"):
        task.load("train")


def test_get_task_spec_rules():
    with pytest.raises(ConfigError, match="requires task.spec"):
        get_task("extraction")
    with pytest.raises(ConfigError, match="only valid"):
        get_task("gsm8k", spec="x.yaml")


def test_config_requires_spec_for_extraction(config_factory):
    with pytest.raises(ConfigError, match="task.spec"):
        config_factory("task.name=extraction")


# ------------------------------------------------------------------ self-check, infer, synthetic data


@pytest.fixture
def synth_spec(tmp_path) -> Path:
    synth_generate(tmp_path / "data", n_train=60, n_eval=20, seed=3)
    spec = yaml.safe_load((ROOT / "specs" / "invoice.yaml").read_text())
    spec["data"]["splits"] = {"train": "data/train.jsonl", "eval": "data/eval.jsonl"}
    p = tmp_path / "invoice.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


def test_synthetic_invoices_pass_the_shipped_spec(synth_spec):
    rep = check_spec(synth_spec)
    assert rep.ok, rep.errors
    st = rep.selftest
    assert st["oracle_failures"] == [] and st["attack_failures"] == []
    assert all(m["still_scored_1"] == 0 for m in st["mutations"].values())
    assert rep.contamination == {"train->eval": 0}


def test_synthetic_generation_is_deterministic(tmp_path):
    a = synth_generate(tmp_path / "a", n_train=5, n_eval=2, seed=11)
    b = synth_generate(tmp_path / "b", n_train=5, n_eval=2, seed=11)
    assert Path(a["train"]).read_text() == Path(b["train"]).read_text()


def test_check_reports_bad_labels_and_loose_matchers(tmp_path):
    loose = json.loads(json.dumps(SPEC))
    loose["fields"]["total"]["tolerance"] = 1e9  # accepts anything
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(loose))
    _write_jsonl(
        tmp_path / "train.jsonl", [{"id": str(i), "text": f"doc {i}", "label": LABEL} for i in range(5)]
    )
    _write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {"id": "e1", "text": "doc 0", "label": LABEL},
            {"id": "e2", "text": "x", "label": {**LABEL, "cur": "JPY"}},
        ],
    )
    rep = check_spec(tmp_path / "s.yaml")
    assert not rep.ok and any("invalid rows" in e for e in rep.errors)
    assert rep.contamination["train->eval"] == 1
    assert any("total:wrong_value" in w for w in rep.warnings)


def test_infer_spec_types(tmp_path):
    rows = [
        {
            "text": f"doc {i}",
            "label": {
                "num": f"${i},000.50",
                "qty": i,
                "when": f"0{1 + i % 9}/2{i % 9}/2024",
                "kind": ["A", "B"][i % 2],
                "flag": bool(i % 2),
                "ref": None if i % 3 else f"R{i}",
                "items": [{"sku": f"S{i}", "price": 1.5}],
            },
        }
        for i in range(30)
    ]
    spec, notes = infer_spec(
        rows, name="d", label_field="label", input_field="text", split_paths={"train": "x"}
    )
    f = spec["fields"]
    assert f["num"]["type"] == "number" and f["qty"]["type"] == "integer"
    assert f["when"]["type"] == "date" and f["when"]["input_formats"] == ["%m/%d/%Y"]
    assert f["kind"] == {"type": "enum", "values": ["A", "B"], "nullable": False}
    assert f["flag"]["type"] == "boolean" and f["ref"]["nullable"] is True
    assert f["items"]["type"] == "object_list" and f["items"]["fields"]["price"]["type"] == "number"
    assert notes
    ExtractionSpec.model_validate(spec)  # the draft is itself a valid spec

    _write_jsonl(tmp_path / "d.jsonl", rows)
    init_spec(tmp_path / "d.jsonl", tmp_path / "out" / "spec.yaml", name="d")
    assert load_spec(tmp_path / "out" / "spec.yaml").split_path("train") == (tmp_path / "d.jsonl").resolve()
    with pytest.raises(DataError, match="exists"):
        init_spec(tmp_path / "d.jsonl", tmp_path / "out" / "spec.yaml", name="d")
