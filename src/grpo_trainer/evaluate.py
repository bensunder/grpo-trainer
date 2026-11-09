"""Held-out evaluation. Run once for the base model and once for the trained model, then `grpo gate`.

Each eval runs in its own process so vLLM memory is fully released between models.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config
from .data import eval_examples, load_splits
from .manifest import atomic_write_json
from .metrics import pass_at_k
from .rewards.registry import RewardContext, build_scorer
from .rewards.scorers import CorrectnessScorer, FormatScorer
from .tasks import dataset_fingerprint, get_task

log = logging.getLogger(__name__)


def _format_scorer(cfg: Config) -> FormatScorer:
    spec = next((r for r in cfg.rewards if r.name == "format"), None)
    return FormatScorer(**spec.params) if spec else FormatScorer()


def run_eval(cfg: Config, model: str, label: str, out_dir: Path | None = None) -> Path:
    from .sampling import generate, vllm_engine

    splits = load_splits(cfg)
    examples = eval_examples(cfg, splits.held_out)
    task = get_task(cfg.task.name)
    system_prompt = task.system_prompt(cfg.task.system_prompt)
    correctness_spec = next(r for r in cfg.rewards if r.name == "correctness")
    correctness = build_scorer(correctness_spec, RewardContext(cfg.grpo.max_completion_length))
    assert isinstance(correctness, CorrectnessScorer)
    fmt = _format_scorer(cfg)
    max_new = cfg.eval_max_completion_length
    k = cfg.eval.pass_k
    convs = [e.messages(system_prompt) for e in examples]

    log.info("eval %s (%s): %d examples, greedy + pass@%d", label, model, len(examples), k)
    with vllm_engine(
        cfg, model, gpu_memory_utilization=cfg.eval.gpu_memory_utilization, max_new_tokens=max_new
    ) as llm:
        greedy = generate(
            llm, convs, n=1, temperature=0.0, top_p=1.0, max_new_tokens=max_new, seed=cfg.run.seed
        )
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
    n_correct = n_format = n_trunc = total_tokens = 0
    pass_k_sum = 0.0
    with per_example_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            row = {"ground_truth": ex.ground_truth}
            g = greedy[i][0]
            ok = correctness(g.text, completion_ids=g.token_ids, row=row) >= 1.0
            fm = fmt(g.text, completion_ids=g.token_ids, row=row) >= 1.0
            n_correct += ok
            n_format += fm
            n_trunc += g.truncated
            total_tokens += len(g.token_ids)
            rec: dict[str, Any] = {
                "id": ex.id,
                "ground_truth": ex.ground_truth,
                "correct": ok,
                "format": fm,
                "truncated": g.truncated,
                "tokens": len(g.token_ids),
                "output": g.text,
            }
            if sampled is not None:
                c = sum(correctness(s.text, completion_ids=s.token_ids, row=row) >= 1.0 for s in sampled[i])
                pk = pass_at_k(len(sampled[i]), c, k)
                pass_k_sum += pk
                rec["pass_k_correct"] = c
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    report = {
        "label": label,
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
        "task": cfg.task.name,
        "split": cfg.task.eval_split,
        "num_examples": n,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "max_completion_length": max_new,
        "system_prompt_sha": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16],
        "greedy": {
            "accuracy": n_correct / n,
            "format_rate": n_format / n,
            "truncation_rate": n_trunc / n,
            "mean_tokens": total_tokens / n,
        },
        f"pass@{k}": (pass_k_sum / n) if sampled is not None else None,
        "pass_k_temperature": cfg.eval.pass_k_temperature if sampled is not None else None,
        "examples_file": str(per_example_path),
    }
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
