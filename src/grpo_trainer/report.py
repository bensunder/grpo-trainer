"""Proof report: the document a buyer signs off on.

Everything is recomputed from the saved per-question outputs with the task's grader -- stored flags are
not trusted -- and every model is compared to the trained model on the same questions with an exact
paired test. Claims are generated from the statistics, never written by hand:

  * "significantly more/less accurate" only when the exact McNemar p-value is at or below alpha;
  * otherwise "no statistically significant difference", with the 95% CI of the difference.

Outputs `<run>/report/proof-report.html` (self-contained, print to PDF from any browser) and
`proof-report.json` (the same numbers, machine-readable).
"""

from __future__ import annotations

import html
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config
from .errors import DataError
from .evaluate import Scorer, aggregate_detail
from .manifest import atomic_write_json
from .metrics import PairedResult, paired_from_index
from .tasks import Example, Task, task_for

log = logging.getLogger(__name__)

HOURS_PER_MONTH = 730
SAME_RUN_KEYS = ("split", "grading_fingerprint", "system_prompt_sha", "task")


@dataclass
class ModelRun:
    label: str  # trained | base | baseline-<name>
    kind: str  # trained | base | baseline
    name: str  # model id / provider:model
    report: dict[str, Any]
    rows: list[dict[str, Any]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataError(f"missing {path}")
    out = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DataError(f"corrupt {path} line {n}: {e}") from None
    return out


def _load_run(eval_dir: Path, label: str, kind: str) -> ModelRun:
    rep_path = eval_dir / f"{label}.json"
    if not rep_path.is_file():
        raise DataError(f"missing {rep_path}; run `grpo eval --label {label}` first")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(eval_dir / f"{label}.examples.jsonl")
    if kind == "baseline":
        prov = rep.get("provider") or {}
        name = f"{prov.get('name', label)} ({prov.get('model', rep.get('model'))})"
    else:
        name = str(rep.get("model"))
    return ModelRun(label, kind, name, rep, rows)


def collect_runs(cfg: Config) -> list[ModelRun]:
    eval_dir = cfg.run_dir / "eval"
    runs = [_load_run(eval_dir, "trained", "trained"), _load_run(eval_dir, "base", "base")]
    for p in sorted(eval_dir.glob("baseline-*.json")):
        runs.append(_load_run(eval_dir, p.stem, "baseline"))
    ref = runs[0].report
    for r in runs[1:]:
        for key in SAME_RUN_KEYS:
            if r.report.get(key) != ref.get(key):
                raise DataError(
                    f"{r.label} is not comparable with trained: {key} differs "
                    f"({r.report.get(key)!r} vs {ref.get(key)!r}); re-run it with the current config"
                )
    return runs


# ------------------------------------------------------------------ re-scoring


@dataclass
class Scored:
    correct: dict[str, bool]
    lenient: dict[str, bool]
    format: dict[str, bool]
    gt: dict[str, str]
    details: list[dict[str, Any]]


def rescore(run: ModelRun, scorer: Scorer, task: Task) -> Scored:
    correct, lenient, fmt, gt = {}, {}, {}, {}
    details = []
    for r in run.rows:
        qid, text, g = str(r["id"]), r.get("output") or "", str(r["ground_truth"])
        if qid in correct:
            raise DataError(f"{run.label}: duplicate id {qid!r}")
        ex = Example(qid, "", g)
        correct[qid] = scorer.is_correct(ex, text)
        lenient[qid] = task.lenient_correct(text, g)
        fmt[qid] = scorer.format(text, completion_ids=None, row={"ground_truth": g}) >= 1.0
        gt[qid] = g
        d = task.eval_detail(text, g)
        details.append({**r, "detail": d} if d is not None else r)
    return Scored(correct, lenient, fmt, gt, details)


def _align(a: Scored, b: Scored, label_b: str) -> list[str]:
    ids = sorted(set(a.correct) & set(b.correct))
    if not ids:
        raise DataError(f"trained and {label_b} share no evaluated questions")
    bad = [q for q in ids if a.gt[q] != b.gt[q]]
    if bad:
        raise DataError(f"{label_b}: ground truth differs from trained for {len(bad)} ids (e.g. {bad[:3]})")
    return ids


def _paired(a: dict[str, bool], b: dict[str, bool], ids: list[str]) -> PairedResult:
    # base = other model, candidate = trained, so positive deltas favour the trained model
    return paired_from_index({q: b[q] for q in ids}, {q: a[q] for q in ids})


def _statement(pr: PairedResult, other: str, metric: str, alpha: float) -> str:
    lo, hi = pr.ci95
    ci = f"95% CI of the difference {100 * lo:+.1f} to {100 * hi:+.1f} pts"
    d = f"{100 * pr.delta:+.1f} pts"
    stats = f"({d}, p={pr.p_value:.3g}; {ci})"
    if pr.p_value <= alpha and pr.delta > 0:
        return f"The trained model is significantly more accurate than {other} on {metric} {stats}."
    if pr.p_value <= alpha and pr.delta < 0:
        return f"The trained model is significantly less accurate than {other} on {metric} {stats}."
    return (
        f"No statistically significant difference from {other} on {metric} ({d}, p={pr.p_value:.3g}; {ci})."
    )


# ------------------------------------------------------------------ economics


def economics(cfg: Config, runs: list[ModelRun]) -> dict[str, Any]:
    e = cfg.economics
    trained = runs[0].report
    tp = trained.get("throughput") or {}
    eph = tp.get("examples_per_hour")
    out: dict[str, Any] = {
        "serving_gpu": e.serving_gpu,
        "gpu_hourly_usd": e.gpu_hourly_usd,
        "replicas": e.replicas,
        "monthly_volume": e.monthly_volume,
        "target_utilization": e.target_utilization,
        "measured_examples_per_hour": eph,
        "measured_on": tp.get("device"),
        "frontier": [
            {
                "label": r.label,
                "name": r.name,
                "usd_per_1k": (r.report.get("cost") or {}).get("usd_per_1k"),
                "latency_p50_ms": (r.report.get("latency_ms") or {}).get("p50"),
                "latency_p95_ms": (r.report.get("latency_ms") or {}).get("p95"),
            }
            for r in runs
            if r.kind == "baseline"
        ],
        "caveats": [
            "Self-hosted throughput is offline batched generation measured during eval on the eval host; "
            "online serving with uneven traffic is lower. Treat it as an upper bound.",
            "Frontier cost uses the per-token prices in the config and the token usage each API reported.",
        ],
    }
    if eph and e.gpu_hourly_usd:
        effective = eph * e.target_utilization
        out["self_hosted_usd_per_1k"] = e.gpu_hourly_usd / effective * 1000
        out["capacity_per_month"] = effective * HOURS_PER_MONTH * e.replicas
        out["self_hosted_monthly_usd"] = e.gpu_hourly_usd * HOURS_PER_MONTH * e.replicas
        if e.monthly_volume:
            out["replicas_needed"] = max(1, math.ceil(e.monthly_volume / (effective * HOURS_PER_MONTH)))
            for f in out["frontier"]:
                if f["usd_per_1k"] is not None:
                    f["monthly_usd"] = f["usd_per_1k"] * e.monthly_volume / 1000
                    f["monthly_savings_usd"] = f["monthly_usd"] - out["self_hosted_monthly_usd"]
    else:
        out["caveats"].append(
            "Set economics.gpu_hourly_usd (and monthly_volume) to compute self-hosted cost."
        )
    return out


# ------------------------------------------------------------------ assemble


def build_summary(cfg: Config) -> dict[str, Any]:
    task = task_for(cfg)
    scorer = Scorer(cfg, task)
    runs = collect_runs(cfg)
    alpha = cfg.eval.gate.alpha
    scored = {r.label: rescore(r, scorer, task) for r in runs}
    t = scored["trained"]

    models = []
    for r in runs:
        s = scored[r.label]
        n = len(s.correct)
        g = r.report.get("greedy") or {}
        detail = aggregate_detail(s.details)
        models.append(
            {
                "label": r.label,
                "kind": r.kind,
                "name": r.name,
                "n": n,
                "strict_accuracy": sum(s.correct.values()) / n,
                "lenient_accuracy": sum(s.lenient.values()) / n,
                "format_rate": sum(s.format.values()) / n,
                "mean_field_score": detail["mean_field_score"] if detail else None,
                "truncation_rate": g.get("truncation_rate"),
                "mean_output_tokens": g.get("mean_tokens"),
                "pass_at_k": next((v for k, v in r.report.items() if k.startswith("pass@")), None),
                "usd_per_1k": (r.report.get("cost") or {}).get("usd_per_1k"),
                "error_rate": r.report.get("error_rate"),
                "fields": detail["fields"] if detail else None,
                "parse_errors": detail["parse_errors"] if detail else None,
            }
        )

    comparisons = []
    for r in runs[1:]:
        s = scored[r.label]
        ids = _align(t, s, r.label)
        for metric, attr in (("strict accuracy", "correct"), ("knowledge-only accuracy", "lenient")):
            pr = _paired(getattr(t, attr), getattr(s, attr), ids)
            comparisons.append(
                {
                    "other": r.label,
                    "other_name": r.name,
                    "metric": metric,
                    "n": pr.n,
                    "full_overlap": pr.n == len(t.correct),
                    "trained_accuracy": pr.candidate_accuracy,
                    "other_accuracy": pr.base_accuracy,
                    "delta": pr.delta,
                    "only_trained_correct": pr.only_candidate_correct,
                    "only_other_correct": pr.only_base_correct,
                    "p_value": pr.p_value,
                    "ci95": list(pr.ci95),
                    "significant": pr.p_value <= alpha,
                    "statement": _statement(
                        pr, r.name if r.kind == "baseline" else "the base model", metric, alpha
                    ),
                }
            )

    gate_path = cfg.run_dir / "eval" / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
    manifest_path = cfg.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    complete_path = cfg.run_dir / "final" / "COMPLETE"
    complete = json.loads(complete_path.read_text()) if complete_path.is_file() else {}

    headline = [c["statement"] for c in comparisons if c["metric"] == "strict accuracy"]
    if gate is None:
        headline.insert(0, "The promotion gate has not been run for this model.")
    elif gate.get("passed"):
        headline.insert(0, "The trained model passed the promotion gate.")
    else:
        headline.insert(
            0, "The trained model did NOT pass the promotion gate: " + "; ".join(gate.get("reasons", []))
        )

    task_info: dict[str, Any] = {
        "name": cfg.task.name,
        "split": cfg.task.eval_split,
        "grading_fingerprint": task.grading_fingerprint(),
    }
    spec = getattr(task, "spec", None)
    if spec is not None:
        task_info.update(
            {
                "spec_name": spec.name,
                "spec_file": cfg.task.spec,
                "fields": {k: {"type": f.type, "weight": f.weight} for k, f in spec.fields.items()},
            }
        )

    examples = _worked_examples(cfg, runs, scored) if cfg.report.include_examples else []
    return {
        "schema_version": 1,
        "title": cfg.report.title or f"{cfg.task.name}: trained model proof report",
        "customer": cfg.report.customer,
        "generated_at": datetime.now(UTC).isoformat(),
        "alpha": alpha,
        "task": task_info,
        "headline": headline,
        "gate": gate,
        "models": models,
        "comparisons": comparisons,
        "economics": economics(cfg, runs),
        "lineage": {
            "run": cfg.run.name,
            "base_model": cfg.model.name_or_path,
            "base_model_revision": cfg.model.revision,
            "adapter": cfg.adapter.mode,
            "trained_model_path": runs[0].report.get("model"),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "git_revision": manifest.get("git_revision"),
            "packages": manifest.get("packages"),
            "train_examples": manifest.get("train_examples"),
            "contaminated_dropped": manifest.get("contaminated_dropped"),
            "global_step": complete.get("global_step"),
            "train_runtime_s": complete.get("train_runtime_s"),
            "eval_examples": len(t.correct),
            "eval_dataset_fingerprint": runs[0].report.get("dataset_fingerprint"),
            "system_prompt_sha": runs[0].report.get("system_prompt_sha"),
        },
        "examples": examples,
    }


def _worked_examples(cfg: Config, runs: list[ModelRun], scored: dict[str, Scored]) -> list[dict[str, Any]]:
    """Questions where the trained model and the strongest other model disagree, trained-wrong first."""
    task = task_for(cfg)
    by_id = {e.id: e for e in task.load(cfg.task.eval_split, None)}
    t = scored["trained"]
    others = [r for r in runs[1:]]
    if not others:
        return []
    best = max(others, key=lambda r: sum(scored[r.label].correct.values()))
    o = scored[best.label]
    ids = [q for q in sorted(t.correct) if q in o.correct and t.correct[q] != o.correct[q]]
    ids.sort(key=lambda q: t.correct[q])  # trained-wrong first: the honest view
    outputs = {r.label: {str(x["id"]): x.get("output") or "" for x in r.rows} for r in (runs[0], best)}
    out = []
    for q in ids[: cfg.report.include_examples]:
        ex = by_id.get(q)
        out.append(
            {
                "id": q,
                "input_excerpt": (ex.question[: cfg.report.example_input_chars] if ex else ""),
                "trained_correct": t.correct[q],
                "other": best.label,
                "other_correct": o.correct[q],
                "trained_output": outputs["trained"].get(q, "")[-1500:],
                "other_output": outputs[best.label].get(q, "")[-1500:],
            }
        )
    return out


# ------------------------------------------------------------------ HTML


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _usd(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"${v:,.{digits}f}"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:,.0f}"


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><b>{value}</b><span>{_e(label)}</span></div>'


def _thead(labels: list[str], text_cols: int = 1) -> str:
    cells = "".join(
        f"<th>{_e(lb)}</th>" if i < text_cols else f'<th class="num">{_e(lb)}</th>'
        for i, lb in enumerate(labels)
    )
    return f"<thead><tr>{cells}</tr></thead><tbody>"


def _cells(*values: Any) -> str:
    """Numeric cells; values are pre-formatted, pre-escaped strings or numbers."""
    return "".join(f'<td class="num">{v}</td>' for v in values)


def _bar(v: float | None) -> str:
    if v is None:
        return ""
    w = max(0.0, min(1.0, v)) * 100
    return f'<span class="bar" aria-hidden="true"><span style="width:{w:.1f}%"></span></span>'


CSS = """
:root{--ink:#16181d;--muted:#5b6170;--line:#d9dce3;--soft:#f4f5f8;--accent:#1f5eff;--good:#167a3f;--bad:#b42318}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 24px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-bottom:6px;
border-bottom:1px solid var(--line)}h3{font-size:15px;margin:20px 0 8px}
.meta{color:var(--muted);font-size:13px;margin-bottom:20px}
.verdict{background:var(--soft);border-left:4px solid var(--accent);padding:14px 18px;border-radius:6px}
.verdict li{margin:4px 0}.pass{color:var(--good);font-weight:600}.fail{color:var(--bad);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 4px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.trained td{font-weight:600}caption{text-align:left;color:var(--muted);font-size:12.5px;padding:4px 0}
.bar{display:block;width:100%;max-width:80px;margin:4px 0 0 auto;height:7px;background:var(--soft);
border-radius:4px;overflow:hidden}.bar span{display:block;height:100%;background:var(--accent)}
.note{color:var(--muted);font-size:12.5px}
.kv th{text-transform:none;letter-spacing:0;font-size:13px;color:var(--ink);width:34%}
code,pre{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--soft);padding:10px;border-radius:6px;white-space:pre-wrap;word-break:break-word;
margin:4px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:10px 0}
.kpi{border:1px solid var(--line);border-radius:8px;padding:12px}.kpi b{display:block;font-size:22px}
.kpi span{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}
@media print{main{padding:0;max-width:none}h2{break-after:avoid}table,.kpi,pre{break-inside:avoid}
.bar span{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def render_html(s: dict[str, Any]) -> str:
    gate = s.get("gate")
    gate_html = (
        '<span class="pass">PASSED</span>'
        if gate and gate.get("passed")
        else '<span class="fail">FAILED</span>'
        if gate
        else "<span>not run</span>"
    )
    trained = next(m for m in s["models"] if m["kind"] == "trained")
    eco = s["economics"]
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(s['title'])}</title><style>{CSS}</style></head><body><main>",
        f"<h1>{_e(s['title'])}</h1>",
        f'<div class="meta">{_e(s.get("customer") or "")}{" · " if s.get("customer") else ""}'
        f"Generated {_e(s['generated_at'][:19].replace('T', ' '))} UTC"
        f" · task {_e(s['task'].get('spec_name') or s['task']['name'])}"
        f" · grading {_e(s['task'].get('grading_fingerprint') or 'built-in')} · gate {gate_html}</div>",
        '<section class="verdict" aria-label="Verdict"><ul>',
        *[f"<li>{_e(h)}</li>" for h in s["headline"]],
        "</ul></section>",
        '<div class="grid">',
        _kpi(_pct(trained["strict_accuracy"]), "trained: fully correct documents"),
        _kpi(_pct(trained["format_rate"]), "trained: schema-valid responses"),
    ]
    if trained["mean_field_score"] is not None:
        parts.append(_kpi(_pct(trained["mean_field_score"]), "trained: mean field score"))
    if eco.get("self_hosted_usd_per_1k") is not None:
        parts.append(_kpi(_usd(eco["self_hosted_usd_per_1k"]), "self-hosted cost per 1,000 requests"))
    parts.append("</div>")

    # accuracy
    parts += [
        "<h2>Accuracy on the held-out set</h2>",
        '<div class="scroll"><table><caption>Every number recomputed from saved outputs with the same '
        "grader. Strict = every field correct and inside the output contract; knowledge-only = values "
        "right, format ignored; schema-valid = exact JSON contract.</caption>",
        _thead(
            [
                "Model",
                "n",
                "Strict",
                "Knowledge-only",
                "Schema-valid",
                "Mean field score",
                "Truncated",
                "Output tokens",
                "Cost / 1k",
            ]
        ),
    ]
    for m in s["models"]:
        cost = (
            m["usd_per_1k"]
            if m["kind"] == "baseline"
            else eco.get("self_hosted_usd_per_1k")
            if m["kind"] == "trained"
            else None
        )
        parts.append(
            f'<tr class="{_e(m["kind"])}"><td>{_e(m["label"])}<br><span class="note">{_e(m["name"])}</span>'
            "</td>"
            + _cells(
                m["n"],
                _pct(m["strict_accuracy"]) + _bar(m["strict_accuracy"]),
                _pct(m["lenient_accuracy"]),
                _pct(m["format_rate"]),
                _pct(m["mean_field_score"]),
                _pct(m["truncation_rate"]),
                _num(m["mean_output_tokens"]),
                _usd(cost),
            )
            + "</tr>"
        )
    parts.append("</tbody></table></div>")

    # statistics
    parts += [
        "<h2>Paired comparisons</h2>",
        f'<p class="note">Same questions, exact two-sided McNemar test, significance at alpha = {s["alpha"]}.'
        " Only questions where the two models disagree carry information.</p>",
        '<div class="scroll"><table>',
        _thead(
            [
                "Trained vs",
                "Metric",
                "n",
                "Trained",
                "Other",
                "Only trained right",
                "Only other right",
                "p",
                "95% CI (pts)",
            ],
            text_cols=2,
        ),
    ]
    for c in s["comparisons"]:
        lo, hi = c["ci95"]
        partial = "" if c["full_overlap"] else ' <span class="note">(subset)</span>'
        parts.append(
            f"<tr><td>{_e(c['other'])}{partial}</td><td>{_e(c['metric'])}</td>"
            + _cells(
                c["n"],
                _pct(c["trained_accuracy"]),
                _pct(c["other_accuracy"]),
                c["only_trained_correct"],
                c["only_other_correct"],
                f"{c['p_value']:.3g}",
                f"{100 * lo:+.1f} to {100 * hi:+.1f}",
            )
            + "</tr>"
        )
    parts.append("</tbody></table></div>")

    # fields
    with_fields = [m for m in s["models"] if m.get("fields")]
    if with_fields:
        names = list(with_fields[0]["fields"])
        parts += [
            "<h2>Per-field accuracy</h2>",
            '<div class="scroll"><table><thead><tr><th>Field</th>',
            *[f'<th class="num">{_e(m["label"])}</th>' for m in with_fields],
            "</tr></thead><tbody>",
        ]
        for f in names:
            parts.append(f"<tr><td><code>{_e(f)}</code></td>")
            for m in with_fields:
                fs = (m["fields"] or {}).get(f)
                parts.append(f'<td class="num">{_pct(fs["accuracy"]) if fs else "n/a"}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table></div>")

        statuses = ["wrong", "partial", "omitted", "hallucinated", "missing_key", "invalid", "invalid_output"]
        parts += [
            "<h3>Error taxonomy (field errors, all fields)</h3>",
            '<p class="note">omitted = value present in the document but returned as null; '
            "hallucinated = value invented where the document has none; "
            "invalid_output = no parseable JSON object at all.</p>",
            '<div class="scroll"><table><thead><tr><th>Model</th>',
            *[f'<th class="num">{_e(st)}</th>' for st in statuses],
            "</tr></thead><tbody>",
        ]
        for m in with_fields:
            totals = {st: 0 for st in statuses}
            for fs in (m["fields"] or {}).values():
                for st, n in fs["statuses"].items():
                    if st in totals:
                        totals[st] += n
            parts.append(
                f"<tr><td>{_e(m['label'])}</td>"
                + "".join(f'<td class="num">{totals[st]}</td>' for st in statuses)
                + "</tr>"
            )
        parts.append("</tbody></table></div>")

    # economics
    parts.append("<h2>Cost model</h2>")
    rows = []
    if eco.get("measured_examples_per_hour"):
        rows.append(
            (
                "Trained model throughput (measured)",
                f"{eco['measured_examples_per_hour']:,.0f} requests/hour on "
                f"{_e(eco.get('measured_on') or 'eval host')}",
            )
        )
    if eco.get("gpu_hourly_usd"):
        rows.append(
            (
                "Serving GPU",
                f"{_e(eco['serving_gpu'])} at {_usd(eco['gpu_hourly_usd'], 3)}/hour x {eco['replicas']} "
                f"replica(s), {_pct(eco['target_utilization'])} target utilisation",
            )
        )
    if eco.get("self_hosted_usd_per_1k") is not None:
        rows.append(("Self-hosted cost per 1,000 requests", _usd(eco["self_hosted_usd_per_1k"], 3)))
        rows.append(("Self-hosted monthly (dedicated, 24/7)", _usd(eco["self_hosted_monthly_usd"])))
        rows.append(("Monthly capacity", f"{eco['capacity_per_month']:,.0f} requests"))
    if eco.get("replicas_needed"):
        rows.append(("Replicas needed for stated volume", str(eco["replicas_needed"])))
    for f in eco["frontier"]:
        val = f"{_usd(f['usd_per_1k'], 3)} per 1,000 requests"
        if f.get("latency_p50_ms") is not None:
            val += f"; latency p50 {f['latency_p50_ms']:.0f} ms, p95 {f['latency_p95_ms']:.0f} ms"
        if f.get("monthly_usd") is not None:
            val += (
                f"; {_usd(f['monthly_usd'])}/month at {eco['monthly_volume']:,} requests "
                f"(savings {_usd(f['monthly_savings_usd'])}/month)"
            )
        rows.append((f"Frontier: {_e(f['name'])}", val))
    if rows:
        parts.append(
            '<table class="kv"><tbody>'
            + "".join(f'<tr><th scope="row">{k}</th><td>{v}</td></tr>' for k, v in rows)
            + "</tbody></table>"
        )
    parts += [f'<p class="note">{_e(c)}</p>' for c in eco["caveats"]]

    # examples
    if s.get("examples"):
        parts.append("<h2>Disagreements (worked examples)</h2>")
        for ex in s["examples"]:
            parts += [
                f"<h3>{_e(ex['id'])}: trained {'right' if ex['trained_correct'] else 'wrong'}, "
                f"{_e(ex['other'])} {'right' if ex['other_correct'] else 'wrong'}</h3>",
                f"<pre>{_e(ex['input_excerpt'])}</pre>",
                f'<p class="note">trained output</p><pre>{_e(ex["trained_output"])}</pre>',
                f'<p class="note">{_e(ex["other"])} output</p><pre>{_e(ex["other_output"])}</pre>',
            ]

    # lineage
    lin = s["lineage"]
    keys = [
        ("Run", "run"),
        ("Base model", "base_model"),
        ("Base revision", "base_model_revision"),
        ("Adapter", "adapter"),
        ("Trained model", "trained_model_path"),
        ("Training steps", "global_step"),
        ("Training examples", "train_examples"),
        ("Train/eval overlap dropped", "contaminated_dropped"),
        ("Config fingerprint", "config_fingerprint"),
        ("Training data fingerprint", "dataset_fingerprint"),
        ("Eval set fingerprint", "eval_dataset_fingerprint"),
        ("System prompt sha", "system_prompt_sha"),
        ("Code revision", "git_revision"),
    ]
    parts.append('<h2>Lineage</h2><table class="kv"><tbody>')
    parts += [f'<tr><th scope="row">{k}</th><td><code>{_e(lin.get(v))}</code></td></tr>' for k, v in keys]
    if s["task"].get("spec_name"):
        parts.append(
            f'<tr><th scope="row">Extraction spec</th><td><code>{_e(s["task"]["spec_name"])} '
            f"({_e(s['task']['grading_fingerprint'])})</code></td></tr>"
        )
    pk = lin.get("packages") or {}
    if pk:
        parts.append(
            '<tr><th scope="row">Pinned stack</th><td><code>'
            + _e(", ".join(f"{k} {v}" for k, v in pk.items()))
            + "</code></td></tr>"
        )
    parts.append("</tbody></table></main></body></html>")
    return "\n".join(parts)


def run_report(cfg: Config) -> Path:
    summary = build_summary(cfg)
    out_dir = cfg.run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "proof-report.json", summary)
    html_path = out_dir / "proof-report.html"
    tmp = html_path.with_suffix(".html.tmp")
    tmp.write_text(render_html(summary), encoding="utf-8")
    tmp.replace(html_path)
    log.info("proof report -> %s", html_path)
    return html_path
