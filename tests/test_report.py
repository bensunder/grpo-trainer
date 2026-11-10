"""Proof report: re-scoring, paired claims, economics, comparability guards, HTML rendering."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml

from grpo_trainer.baseline import HttpResponse, run_baseline
from grpo_trainer.config import load_config
from grpo_trainer.errors import DataError
from grpo_trainer.evaluate import Scorer, aggregate_detail, held_out_examples, system_prompt_sha
from grpo_trainer.manifest import atomic_write_json
from grpo_trainer.report import build_summary, run_report
from grpo_trainer.synth.invoices import generate as synth_generate
from grpo_trainer.tasks import dataset_fingerprint, task_for

ROOT = Path(__file__).resolve().parents[1]
N_EVAL = 40


@pytest.fixture
def run_cfg(tmp_path):
    synth_generate(tmp_path / "data", n_train=10, n_eval=N_EVAL, seed=9)
    spec = yaml.safe_load((ROOT / "specs" / "invoice.yaml").read_text())
    spec["data"]["splits"] = {"train": "data/train.jsonl", "eval": "data/eval.jsonl"}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    prov = {
        "name": "frontier",
        "kind": "openai",
        "model": "big-model",
        "api_key_env": "K",
        "input_usd_per_mtok": 2.0,
        "output_usd_per_mtok": 8.0,
        "requests_per_minute": 100000,
    }

    def make(*overrides: str):
        return load_config(
            ROOT / "configs" / "extraction.yaml",
            "gpu16",
            [
                f"task.spec={tmp_path / 'spec.yaml'}",
                f"run.output_root={tmp_path / 'runs'}",
                f"eval.limit={N_EVAL}",
                f"baseline.providers={json.dumps([prov])}",
                *overrides,
            ],
        )

    return make


def oracle(task, ex) -> str:
    return (
        "<think>read the invoice and located every field</think><answer>"
        + json.dumps(task.ideal_output(ex.ground_truth))
        + "</answer>"
    )


def corrupt(task, ex) -> str:
    obj = task.ideal_output(ex.ground_truth)
    obj["total_amount"] = (obj["total_amount"] or 0) + 100
    return "<think>read the invoice and located every field</think><answer>" + json.dumps(obj) + "</answer>"


def write_eval(cfg, label: str, wrong_ids: set[str], *, eph: float | None = 3600.0) -> None:
    task = task_for(cfg)
    examples = held_out_examples(cfg, task)
    scorer = Scorer(cfg, task)
    rows = []
    for ex in examples:
        text = corrupt(task, ex) if ex.id in wrong_ids else oracle(task, ex)
        rec = scorer.row(ex, text, None)
        rec.update({"truncated": False, "tokens": 300, "output": text})
        rows.append(rec)
    d = cfg.run_dir / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{label}.examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    rep = {
        "label": label,
        "model": f"/models/{label}",
        "task": cfg.task.name,
        "split": cfg.task.eval_split,
        "num_examples": n,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "grading_fingerprint": task.grading_fingerprint(),
        "max_completion_length": 1024,
        "system_prompt_sha": system_prompt_sha(task.system_prompt(None)),
        "greedy": {
            "accuracy": sum(r["correct"] for r in rows) / n,
            "lenient_accuracy": 0.0,
            "format_rate": sum(r["format"] for r in rows) / n,
            "truncation_rate": 0.0,
            "mean_tokens": 300.0,
        },
        "pass@4": None,
        "throughput": {"examples_per_hour": eph, "device": "TEST GPU"},
        "fields": aggregate_detail(rows),
    }
    atomic_write_json(d / f"{label}.json", rep)


class WrongForSome:
    """Fake frontier API: oracle answers, except a corrupted total for the listed documents."""

    def __init__(self, cfg, wrong_ids):
        task = task_for(cfg)
        self.by_q = {
            ex.question: (corrupt if ex.id in wrong_ids else oracle)(task, ex)
            for ex in held_out_examples(cfg, task)
        }

    def post(self, url, headers, payload, timeout):
        user = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        body = {
            "choices": [{"message": {"content": self.by_q[user]}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 400},
        }
        return HttpResponse(200, {}, json.dumps(body).encode())


def ids(cfg):
    return [e.id for e in held_out_examples(cfg, task_for(cfg))]


def test_summary_claims_follow_the_statistics(run_cfg):
    cfg = run_cfg("economics.gpu_hourly_usd=0.5", "economics.monthly_volume=200000")
    all_ids = ids(cfg)
    write_eval(cfg, "trained", set(all_ids[:2]))  # 38/40
    write_eval(cfg, "base", set(all_ids[:24]))  # 16/40
    run_baseline(
        cfg, "frontier", transport=WrongForSome(cfg, set(all_ids[1:4])), env={"K": "x"}, sleep=lambda s: None
    )
    atomic_write_json(cfg.run_dir / "eval" / "gate.json", {"passed": True, "reasons": []})

    s = build_summary(cfg)
    by = {m["label"]: m for m in s["models"]}
    assert by["trained"]["strict_accuracy"] == pytest.approx(38 / 40)
    assert by["base"]["strict_accuracy"] == pytest.approx(16 / 40)
    assert by["baseline-frontier"]["strict_accuracy"] == pytest.approx(37 / 40)
    assert by["trained"]["fields"]["total_amount"]["accuracy"] == pytest.approx(38 / 40)

    vs_base = next(c for c in s["comparisons"] if c["other"] == "base" and c["metric"] == "strict accuracy")
    assert vs_base["only_trained_correct"] == 22 and vs_base["only_other_correct"] == 0
    assert vs_base["significant"] and "significantly more accurate" in vs_base["statement"]
    vs_api = next(
        c for c in s["comparisons"] if c["other"] == "baseline-frontier" and c["metric"] == "strict accuracy"
    )
    assert not vs_api["significant"] and "No statistically significant difference" in vs_api["statement"]
    assert vs_api["ci95"][0] < 0 < vs_api["ci95"][1]
    assert s["headline"][0].startswith("The trained model passed")

    eco = s["economics"]
    assert eco["self_hosted_usd_per_1k"] == pytest.approx(0.5 / (3600 * 0.6) * 1000)
    assert eco["replicas_needed"] == 1  # 3600 * 0.6 * 730 = 1.58M/month capacity
    fr = eco["frontier"][0]
    assert fr["usd_per_1k"] == pytest.approx((1200 * 2 + 400 * 8) / 1e6 * 1000)
    assert fr["monthly_savings_usd"] == pytest.approx(fr["monthly_usd"] - 0.5 * 730)


def test_report_html_is_well_formed_and_escaped(run_cfg):
    cfg = run_cfg("report.include_examples=2", "report.customer=<script>x</script>")
    all_ids = ids(cfg)
    write_eval(cfg, "trained", {all_ids[0]})
    write_eval(cfg, "base", set(all_ids[1:10]))  # disagree on id 0 (trained wrong) and 1..9 (base wrong)
    path = run_report(cfg)
    text = path.read_text()
    assert "<script>x" not in text and "&lt;script&gt;" in text
    assert "Per-field accuracy" in text and "Paired comparisons" in text and "not been run" in text

    class Balance(HTMLParser):
        void = {"meta", "br", "link"}

        def __init__(self):
            super().__init__()
            self.stack = []

        def handle_starttag(self, tag, attrs):
            if tag not in self.void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            assert self.stack and self.stack[-1] == tag, (tag, self.stack[-5:])
            self.stack.pop()

    p = Balance()
    p.feed(text)
    assert p.stack == []
    summary = json.loads((path.parent / "proof-report.json").read_text())
    assert summary["examples"] and summary["examples"][0]["trained_correct"] is False  # trained-wrong first


def test_report_refuses_incomparable_runs(run_cfg):
    cfg = run_cfg()
    write_eval(cfg, "trained", set())
    write_eval(cfg, "base", set())
    rep_path = cfg.run_dir / "eval" / "base.json"
    rep = json.loads(rep_path.read_text())
    rep["grading_fingerprint"] = "different"
    rep_path.write_text(json.dumps(rep))
    with pytest.raises(DataError, match="grading_fingerprint"):
        build_summary(cfg)
    rep_path.unlink()
    with pytest.raises(DataError, match="grpo eval --label base"):
        build_summary(cfg)
