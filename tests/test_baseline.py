"""Frontier baseline: request shapes, retries, guards, cache resume, eval-compatible report. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from grpo_trainer.baseline import (
    CircuitBreaker,
    HttpResponse,
    ProviderClient,
    RateLimiter,
    RequestFailed,
    TransportError,
    baseline_paths,
    percentile,
    run_baseline,
)
from grpo_trainer.config import BaselineProvider, load_config
from grpo_trainer.errors import BudgetError, ConfigError, ProviderError
from grpo_trainer.synth.invoices import generate as synth_generate

ROOT = Path(__file__).resolve().parents[1]


def provider(**kw) -> dict:
    base = {
        "name": "fake",
        "kind": "openai",
        "model": "m",
        "api_key_env": "FAKE_KEY",
        "input_usd_per_mtok": 1.0,
        "output_usd_per_mtok": 4.0,
        "max_retries": 2,
        "requests_per_minute": 100000,
    }
    return {**base, **kw}


@pytest.fixture
def ext_cfg(tmp_path):
    synth_generate(tmp_path / "data", n_train=20, n_eval=6, seed=5)
    spec = yaml.safe_load((ROOT / "specs" / "invoice.yaml").read_text())
    spec["data"]["splits"] = {"train": "data/train.jsonl", "eval": "data/eval.jsonl"}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))

    def make(*overrides: str, providers: list[dict] | None = None):
        base = [
            f"task.spec={tmp_path / 'spec.yaml'}",
            f"run.output_root={tmp_path / 'runs'}",
            "eval.limit=6",
            f"baseline.providers={json.dumps(providers or [provider()])}",
        ]
        return load_config(ROOT / "configs" / "extraction.yaml", "gpu16", base + list(overrides))

    return make


class FakeTransport:
    """Answers each request with the oracle output for the matching document, unless scripted."""

    def __init__(self, cfg, script=None, kind="openai"):
        from grpo_trainer.evaluate import held_out_examples
        from grpo_trainer.tasks import task_for

        task = task_for(cfg)
        self.answers = {
            e.question: "<think>located all fields in the invoice text</think><answer>"
            + json.dumps(task.ideal_output(e.ground_truth))
            + "</answer>"
            for e in held_out_examples(cfg, task)
        }
        self.script = list(script or [])
        self.kind = kind
        self.calls: list[tuple[str, dict, dict]] = []

    def post(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), payload))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if isinstance(step, HttpResponse):
                return step
        user = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        text = self.answers[user]
        if self.kind == "anthropic":
            body = {
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 200},
            }
        else:
            body = {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
            }
        return HttpResponse(200, {}, json.dumps(body).encode())


ENV = {"FAKE_KEY": "sk-test"}


def test_end_to_end_report_is_eval_compatible(ext_cfg):
    cfg = ext_cfg()
    t = FakeTransport(cfg)
    path = run_baseline(cfg, "fake", transport=t, env=ENV, sleep=lambda s: None)
    rep = json.loads(Path(path).read_text())
    assert rep["greedy"]["accuracy"] == 1.0 and rep["greedy"]["format_rate"] == 1.0
    assert rep["num_examples"] == 6 and rep["error_rate"] == 0.0
    assert rep["cost"]["total_usd"] == pytest.approx(6 * (1000 * 1 + 200 * 4) / 1e6)
    assert rep["grading_fingerprint"] and rep["system_prompt_sha"]
    rows = [json.loads(line) for line in Path(rep["examples_file"]).read_text().splitlines()]
    assert len(rows) == 6 and all(r["correct"] and r["detail"]["field_score"] == 1.0 for r in rows)
    url, headers, payload = t.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["authorization"] == "Bearer sk-test"
    assert payload["temperature"] == 0.0 and payload["max_completion_tokens"] == 1024
    assert payload["messages"][0]["role"] == "system"
    assert "sk-test" not in Path(baseline_paths(cfg, "fake")[2]).read_text()  # key never cached


def test_cache_makes_rerun_free(ext_cfg):
    cfg = ext_cfg()
    run_baseline(cfg, "fake", transport=FakeTransport(cfg), env=ENV, sleep=lambda s: None)
    t2 = FakeTransport(cfg)
    run_baseline(cfg, "fake", transport=t2, env={}, sleep=lambda s: None)  # no key needed: all cached
    assert t2.calls == []
    assert run_baseline(cfg, "fake", dry_run=True)["cached"] == 6


def test_retries_honour_retry_after_then_succeed(ext_cfg):
    cfg = ext_cfg("eval.limit=1")
    slept: list[float] = []
    t = FakeTransport(
        cfg, script=[HttpResponse(429, {"Retry-After": "7"}, b"slow down"), TransportError("reset")]
    )
    run_baseline(cfg, "fake", transport=t, env=ENV, sleep=slept.append)
    assert len(t.calls) == 3
    assert slept[0] == 7.0 and 0.5 <= slept[1] <= 2.0  # Retry-After, then jittered backoff


def test_auth_failure_aborts_immediately(ext_cfg):
    cfg = ext_cfg()
    t = FakeTransport(cfg, script=[HttpResponse(401, {}, b"bad key")] * 10)
    with pytest.raises(ProviderError, match="401"):
        run_baseline(cfg, "fake", transport=t, env=ENV, sleep=lambda s: None)


def test_error_rate_guard_and_circuit_breaker(ext_cfg):
    cfg = ext_cfg("baseline.circuit_breaker_failures=100", providers=[provider(max_concurrency=1)])
    t = FakeTransport(cfg, script=[HttpResponse(400, {}, b"content filter")])
    with pytest.raises(ProviderError, match="max_error_rate"):
        run_baseline(cfg, "fake", transport=t, env=ENV, sleep=lambda s: None)
    # the 5 successes were cached: tolerate the failure and only the missing one is re-sent
    cfg2 = ext_cfg("baseline.max_error_rate=0.5", providers=[provider(max_concurrency=1)])
    t2 = FakeTransport(cfg2)
    run_baseline(cfg2, "fake", transport=t2, env=ENV, sleep=lambda s: None)
    assert len(t2.calls) == 1

    cfg3 = ext_cfg(
        "run.name=breaker", "baseline.circuit_breaker_failures=2", providers=[provider(max_concurrency=1)]
    )
    t3 = FakeTransport(cfg3, script=[HttpResponse(500, {}, b"x")] * 50)
    with pytest.raises(ProviderError, match="consecutive failures"):
        run_baseline(cfg3, "fake", transport=t3, env=ENV, sleep=lambda s: None)
    assert len(t3.calls) == 6  # 2 requests x 3 attempts, then the breaker stops the run


def test_budget_refusal_and_dry_run(ext_cfg):
    cfg = ext_cfg("baseline.max_cost_usd=0.0001")
    est = run_baseline(cfg, "fake", dry_run=True)
    assert est["requests"] == 6 and est["usd_worst_case"] > 0.0001
    with pytest.raises(BudgetError, match="exceeds"):
        run_baseline(cfg, "fake", transport=FakeTransport(cfg), env=ENV)


def test_missing_key_and_unknown_provider(ext_cfg):
    cfg = ext_cfg()
    with pytest.raises(ProviderError, match="FAKE_KEY"):
        run_baseline(cfg, "fake", transport=FakeTransport(cfg), env={})
    with pytest.raises(ConfigError, match="no baseline provider"):
        run_baseline(cfg, "nope")


def test_anthropic_and_azure_request_shapes(ext_cfg):
    cfg = ext_cfg(providers=[provider(kind="anthropic", name="claude")])
    t = FakeTransport(cfg, kind="anthropic")
    rep = json.loads(
        Path(run_baseline(cfg, "claude", transport=t, env=ENV, sleep=lambda s: None)).read_text()
    )
    url, headers, payload = t.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-test" and headers["anthropic-version"]
    assert "system" in payload and all(m["role"] != "system" for m in payload["messages"])
    assert payload["max_tokens"] == 1024 and rep["greedy"]["accuracy"] == 1.0

    az = BaselineProvider(
        **provider(kind="azure_openai", base_url="https://r.openai.azure.com/", api_version="v1x")
    )
    url, headers, payload = ProviderClient(az, "k", None).build_request(
        [{"role": "user", "content": "x"}], 50, 1
    )
    assert url == "https://r.openai.azure.com/openai/deployments/m/chat/completions?api-version=v1x"
    assert headers == {"api-key": "k"}


def test_provider_config_validation():
    with pytest.raises(ValueError, match="azure_openai requires"):
        BaselineProvider(**provider(kind="azure_openai"))
    with pytest.raises(ValueError, match="extra_body"):
        BaselineProvider(**provider(extra_body={"model": "other"}))
    with pytest.raises(ValueError, match="max_tokens_param"):
        BaselineProvider(**provider(kind="anthropic", max_tokens_param="max_tokens"))


def test_client_gives_up_and_parses_truncation():
    p = BaselineProvider(**provider(max_retries=1))

    class Always:
        calls = 0

        def post(self, *a):
            Always.calls += 1
            return HttpResponse(503, {}, b"busy")

    with pytest.raises(RequestFailed, match="gave up after 2"):
        ProviderClient(p, "k", Always(), sleep=lambda s: None).complete(
            [{"role": "user", "content": "x"}], 10, 1
        )
    body = json.dumps({"choices": [{"message": {"content": "a"}, "finish_reason": "length"}]}).encode()
    assert ProviderClient(p, "k", None).parse_response(body)[2] is True


def test_primitives():
    t = [0.0]
    slept: list[float] = []
    rl = RateLimiter(60, clock=lambda: t[0], sleep=slept.append)
    rl.acquire()
    rl.acquire()
    assert slept == [1.0]
    cb = CircuitBreaker(2)
    cb.failure()
    cb.success()
    cb.failure()
    assert not cb.open
    cb.failure()
    assert cb.open
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert percentile([5], 0.95) == 5
