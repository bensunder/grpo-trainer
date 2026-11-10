"""Profiler and eval stage logic with the vLLM engine replaced by a deterministic fake."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from grpo_trainer import data as data_mod
from grpo_trainer import evaluate as eval_mod
from grpo_trainer import profiler as prof_mod
from grpo_trainer import sampling
from grpo_trainer.cli import main
from grpo_trainer.sampling import Sample
from grpo_trainer.tasks import GSM8KTask

from .conftest import CONFIGS

TRAIN = [{"question": f"Q{i}?", "answer": f"#### {i}"} for i in range(6)]
TEST = [{"question": f"T{i}?", "answer": f"#### {i}"} for i in range(4)]


def good(n):
    return Sample(
        f"<think>careful reasoning about the problem</think><answer>{n}</answer>", (1,) * 10, "stop"
    )


def bad():
    return Sample("I don't know", (1,) * 768, "length")


@pytest.fixture
def fake_env(monkeypatch):
    task = GSM8KTask(loader=lambda split: {"train": TRAIN, "test": TEST}[split])
    monkeypatch.setattr(data_mod, "get_task", lambda name, **_: task)
    monkeypatch.setattr(prof_mod, "task_for", lambda cfg: task)
    monkeypatch.setattr(eval_mod, "task_for", lambda cfg: task)
    calls = []

    @contextmanager
    def fake_engine(cfg, model, **kw):
        calls.append(model)
        yield model

    def fake_generate(llm, conversations, *, n, temperature, **kw):
        out = []
        for conv in conversations:
            i = int(conv[-1]["content"][1:-1])  # "Q3?" -> 3
            if llm == "trained":
                out.append([good(i)] * n)
            else:
                # base: question i solved in (i mod 3) of n samples; greedy correct for even i
                k = (i % 3) * n // 2 if temperature > 0 else (n if i % 2 == 0 else 0)
                out.append([good(i)] * min(k, n) + [bad()] * (n - min(k, n)))
        return out

    monkeypatch.setattr(sampling, "vllm_engine", fake_engine)
    monkeypatch.setattr(sampling, "generate", fake_generate)
    return calls


def test_profile_writes_rates_summary_and_caches(fake_env, config_factory, tmp_path):
    cfg = config_factory(f"run.output_root={tmp_path}", "profiler.samples_per_prompt=4")
    path = prof_mod.run_profile(cfg)
    rates = data_mod.read_profile(path)
    # i % 3 == 0 -> 0/4, i % 3 == 1 -> 2/4, i % 3 == 2 -> 4/4
    assert rates == {f"train-{i}": [0.0, 0.5, 1.0][i % 3] for i in range(6)}
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert (summary["never_solved"], summary["always_solved"], summary["kept"]) == (2, 2, 2)

    assert prof_mod.run_profile(cfg) == path  # cache hit, engine not started again
    assert len(fake_env) == 1

    splits = data_mod.load_splits(cfg)
    kept = data_mod.training_examples(cfg, splits, path)
    assert sorted(e.id for e in kept) == ["train-1", "train-4"]


def test_eval_reports_and_gate_end_to_end(fake_env, tmp_path):
    common = [
        "--config",
        str(CONFIGS / "base.yaml"),
        "--profile",
        "gpu24",
        "--set",
        f"run.output_root={tmp_path}",
        "--set",
        "run.name=e",
        "--set",
        "eval.pass_k=4",
    ]
    assert main(["eval", *common, "--model", "base", "--label", "base"]) == 0
    assert main(["eval", *common, "--model", "trained", "--label", "trained"]) == 0

    base = json.loads((tmp_path / "e" / "eval" / "base.json").read_text())
    trained = json.loads((tmp_path / "e" / "eval" / "trained.json").read_text())
    assert base["num_examples"] == 4
    assert base["greedy"]["accuracy"] == pytest.approx(0.5)  # T0, T2 correct
    assert base["greedy"]["truncation_rate"] == pytest.approx(0.5)
    assert trained["greedy"]["accuracy"] == 1.0 and trained["greedy"]["format_rate"] == 1.0
    assert trained["pass@4"] == 1.0
    lines = (tmp_path / "e" / "eval" / "base.examples.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["correct"] is True

    assert main(["gate", *common]) == 0
    gate = json.loads((tmp_path / "e" / "eval" / "gate.json").read_text())
    assert gate["passed"] and gate["accuracy_delta"] == pytest.approx(0.5)
