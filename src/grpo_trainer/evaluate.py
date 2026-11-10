"""Held-out evaluation. Run once for the base model and once for the trained model, then `grpo gate`.

Each eval runs in its own process so vLLM memory is fully released between models. Scoring goes through
the task's grading hooks, so the same code evaluates GSM8K numbers and extraction JSON.

Per-example rows (`<label>.examples.jsonl`) are the source of truth for the gate and the proof report:
both re-score from `output` + `ground_truth` rather than trusting stored flags.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config
from .data import eval_examples
from .manifest import atomic_write_json
from .metrics import pass_at_k
from .rewards.registry import build_scorer, reward_context
from .tasks import Example, Task, dataset_fingerprint, task_for

log = logging.getLogger(__name__)


def held_out_examples(cfg: Config, task: Task) -> list[Example]:
    """The evaluated set. Only the eval split is loaded: contamination filtering touches train only."""
    return eval_examples(cfg, task.load(cfg.task.eval_split, None))


def system_prompt_sha(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


class Scorer:
    """Scores one output exactly as eval does; shared by eval, the frontier baseline and the report."""

    def __init__(self, cfg: Config, task: Task) -> None:
        self.task = task
        ctx = reward_context(cfg, task)
        self.correctness = build_scorer(next(r for r in cfg.rewards if r.name == "correctness"), ctx)
        fmt = next((r for r in cfg.rewards if r.name == "format"), None)
        self.format = task.format_scorer(dict(fmt.params) if fmt else {})

    def row(self, ex: Example, text: str, token_ids: tuple[int, ...] | None) -> dict[str, Any]:
        gt_row = {"ground_truth": ex.ground_truth}
        rec: dict[str, Any] = {
            "id": ex.id,
            "ground_truth": ex.ground_truth,
            "correct": self.correctness(text, completion_ids=token_ids, row=gt_row) >= 1.0,
            "lenient_correct": self.task.lenient_correct(text, ex.ground_truth),
            "format": self.format(text, completion_ids=token_ids, row=gt_row) >= 1.0,
        }
        detail = self.task.eval_detail(text, ex.ground_truth)
        if detail is not None:
            rec["detail"] = detail
        return rec

    def is_correct(self, ex: Example, text: str) -> bool:
        return self.correctness(text, completion_ids=None, row={"ground_truth": ex.ground_truth}) >= 1.0


def aggregate_detail(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Per-field accuracy / mean score / status counts across examples (tasks with eval_detail only)."""
    with_detail = [r["detail"] for r in rows if isinstance(r.get("detail"), dict) and "fields" in r["detail"]]
    if not with_detail:
        return None
    n = len(with_detail)
    exact: dict[str, int] = defaultdict(int)
    score_sum: dict[str, float] = defaultdict(float)
    statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    parse_errors: dict[str, int] = defaultdict(int)
    for d in with_detail:
        if d.get("parse_error"):
            parse_errors[d["parse_error"]] += 1
        for name, fr in d["fields"].items():
            score_sum[name] += float(fr["score"])
            exact[name] += float(fr["score"]) >= 1.0
            statuses[name][fr["status"]] += 1
    return {
        "mean_field_score": sum(float(d["field_score"]) for d in with_detail) / n,
        "parse_errors": dict(parse_errors),
        "fields": {
            name: {
                "accuracy": exact[name] / n,
                "mean_score": score_sum[name] / n,
                "statuses": dict(statuses[name]),
            }
            for name in score_sum
        },
    }


def device_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - informational only
        return None


def run_eval(cfg: Config, model: str, label: str, out_dir: Path | None = None) -> Path:
    from .sampling import generate, vllm_engine

    task = task_for(cfg)
    examples = held_out_examples(cfg, task)
    system_prompt = task.system_prompt(cfg.task.system_prompt)
    scorer = Scorer(cfg, task)
    max_new = cfg.eval_max_completion_length
    k = cfg.eval.pass_k
    convs = [e.messages(system_prompt) for e in examples]

    log.info("eval %s (%s): %d examples, greedy + pass@%d", label, model, len(examples), k)
    with vllm_engine(
        cfg, model, gpu_memory_utilization=cfg.eval.gpu_memory_utilization, max_new_tokens=max_new
    ) as llm:
        t0 = time.monotonic()
        greedy = generate(
            llm, convs, n=1, temperature=0.0, top_p=1.0, max_new_tokens=max_new, seed=cfg.run.seed
        )
        greedy_seconds = time.monotonic() - t0
        sampled = (
            generate(
                llm,
                convs,
                n=k,
                temperature=cfg.eval.pass_k_temperature,
                top_p=1.0,
                max_new_tokens=max_new,
                seed=cfg.run.seed,
            )
            if k > 1
            else None
        )

    out_dir = out_dir or (cfg.run_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = out_dir / f"{label}.examples.jsonl"
    rows: list[dict[str, Any]] = []
    n_trunc = total_tokens = 0
    pass_k_sum = 0.0
    with per_example_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            g = greedy[i][0]
            rec = scorer.row(ex, g.text, g.token_ids)
            rec.update({"truncated": g.truncated, "tokens": len(g.token_ids), "output": g.text})
            n_trunc += g.truncated
            total_tokens += len(g.token_ids)
            if sampled is not None:
                gt_row = {"ground_truth": ex.ground_truth}
                c = sum(
                    scorer.correctness(s.text, completion_ids=s.token_ids, row=gt_row) >= 1.0
                    for s in sampled[i]
                )
                pass_k_sum += pass_at_k(len(sampled[i]), c, k)
                rec["pass_k_correct"] = c
            rows.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    report: dict[str, Any] = {
        "label": label,
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
        "task": cfg.task.name,
        "split": cfg.task.eval_split,
        "num_examples": n,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "grading_fingerprint": task.grading_fingerprint(),
        "max_completion_length": max_new,
        "system_prompt_sha": system_prompt_sha(system_prompt),
        "greedy": {
            "accuracy": sum(r["correct"] for r in rows) / n,
            "lenient_accuracy": sum(r["lenient_correct"] for r in rows) / n,
            "format_rate": sum(r["format"] for r in rows) / n,
            "truncation_rate": n_trunc / n,
            "mean_tokens": total_tokens / n,
        },
        f"pass@{k}": (pass_k_sum / n) if sampled is not None else None,
        "pass_k_temperature": cfg.eval.pass_k_temperature if sampled is not None else None,
        # Offline batched greedy generation on this host: an upper bound for serving throughput.
        "throughput": {
            "greedy_seconds": greedy_seconds,
            "examples_per_hour": (n / greedy_seconds * 3600) if greedy_seconds > 0 else None,
            "output_tokens_per_second": (total_tokens / greedy_seconds) if greedy_seconds > 0 else None,
            "device": device_name(),
            "mode": "offline_batch",
        },
        "examples_file": str(per_example_path),
    }
    detail = aggregate_detail(rows)
    if detail is not None:
        report["fields"] = detail
    report_path = out_dir / f"{label}.json"
    atomic_write_json(report_path, report)
    log.info(
        "eval %s: accuracy %.4f format %.4f truncated %.4f -> %s",
        label,
        report["greedy"]["accuracy"],
        report["greedy"]["format_rate"],
        report["greedy"]["truncation_rate"],
        report_path,
    )
    return report_path
