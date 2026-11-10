"""Deploy bundle + gateway: gate refusal, integrity, prompt pinning, and the /v1/predict contract."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pytest
import yaml

from grpo_trainer.bundle import run_bundle, verify_bundle
from grpo_trainer.config import load_config
from grpo_trainer.errors import DataError, GateFailedError
from grpo_trainer.evaluate import held_out_examples
from grpo_trainer.manifest import atomic_write_json
from grpo_trainer.serve.core import BundleError, CircuitBreaker, GatewaySettings, IdempotencyCache
from grpo_trainer.synth.invoices import generate as synth_generate
from grpo_trainer.tasks import task_for

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from grpo_trainer.serve.app import create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trained_run(tmp_path):
    synth_generate(tmp_path / "data", n_train=10, n_eval=8, seed=4)
    spec = yaml.safe_load((ROOT / "specs" / "invoice.yaml").read_text())
    spec["data"]["splits"] = {"train": "data/train.jsonl", "eval": "data/eval.jsonl"}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    cfg = load_config(
        ROOT / "configs" / "extraction.yaml",
        "gpu16",
        [f"task.spec={tmp_path / 'spec.yaml'}", f"run.output_root={tmp_path / 'runs'}"],
    )
    merged = cfg.run_dir / "final" / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "qwen2"}')
    (merged / "model.safetensors").write_bytes(b"\x00" * 1024)
    atomic_write_json(
        cfg.run_dir / "final" / "COMPLETE", {"global_step": 300, "eval_model_path": str(merged)}
    )
    return cfg


def pass_gate(cfg, passed=True):
    atomic_write_json(
        cfg.run_dir / "eval" / "gate.json", {"passed": passed, "reasons": [] if passed else ["too low"]}
    )


def test_bundle_refuses_without_passing_gate(trained_run):
    with pytest.raises(GateFailedError, match="no gate result"):
        run_bundle(trained_run)
    pass_gate(trained_run, False)
    with pytest.raises(GateFailedError, match="too low"):
        run_bundle(trained_run)
    out = run_bundle(trained_run, allow_failed_gate=True)
    assert json.loads((out / "BUNDLE.json").read_text())["gate"]["passed"] is False


def test_bundle_contents_and_integrity(trained_run):
    pass_gate(trained_run)
    out = run_bundle(trained_run, smoke_examples=3)
    for rel in (
        "model/model.safetensors",
        "serving/settings.json",
        "serving/system_prompt.txt",
        "serving/spec.yaml",
        "gateway/Dockerfile",
        "gateway/requirements.txt",
        "gateway/src/grpo_trainer/serve/app.py",
        "docker-compose.yml",
        "smoke_test.py",
        "README.md",
        "BUNDLE.json",
    ):
        assert (out / rel).is_file(), rel
    assert not list(out.rglob("__pycache__"))
    meta = json.loads((out / "BUNDLE.json").read_text())
    assert meta["gate"]["passed"] and len(meta["bundle_id"]) == 16
    assert verify_bundle(out) == []
    smoke = [json.loads(line) for line in (out / "smoke" / "examples.jsonl").read_text().splitlines()]
    assert len(smoke) == 3 and isinstance(smoke[0]["expected"], dict)
    py_compile.compile(str(out / "smoke_test.py"), doraise=True)
    compose = yaml.safe_load((out / "docker-compose.yml").read_text())
    assert "ports" not in compose["services"]["vllm"] and compose["services"]["gateway"]["ports"]
    assert f"--served-model-name={trained_run.run.name}" in compose["services"]["vllm"]["command"]

    (out / "README.md").write_text("tampered")
    assert verify_bundle(out) == ["modified README.md"]
    with pytest.raises(DataError, match="exists"):
        run_bundle(trained_run)
    assert run_bundle(trained_run, force=True) == out


def test_gateway_refuses_tampered_prompt(trained_run):
    pass_gate(trained_run)
    out = run_bundle(trained_run)
    GatewaySettings.load(out, env={})
    (out / "serving" / "system_prompt.txt").write_text("You are a helpful assistant.")
    with pytest.raises(BundleError, match="sha256"):
        GatewaySettings.load(out, env={})


class FakeVLLM:
    """httpx MockTransport handler mimicking vLLM's OpenAI-compatible server."""

    def __init__(self, settings, reply):
        self.settings, self.reply, self.calls, self.status = settings, reply, [], 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": self.settings.served_model_name}]})
        payload = json.loads(request.content)
        self.calls.append(payload)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "boom"})
        user = payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": self.reply(user)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 250},
            },
        )


@pytest.fixture
def gateway(trained_run):
    pass_gate(trained_run)
    out = run_bundle(trained_run)
    settings = GatewaySettings.load(
        out,
        env={
            "GATEWAY_API_KEY": "secret",
            "MAX_INPUT_CHARS": "5000",
            "BREAKER_FAILURES": "2",
            "BREAKER_COOLDOWN_S": "999",
        },
    )
    task = task_for(trained_run)
    ideal = {ex.question: task.ideal_output(ex.ground_truth) for ex in held_out_examples(trained_run, task)}

    def reply(user):
        return (
            "<think>found the invoice fields in the text</think><answer>"
            + json.dumps(ideal.get(user, {}))
            + "</answer>"
        )

    fake = FakeVLLM(settings, reply)
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake), base_url="http://vllm")
    app = create_app(settings, client=client)
    with TestClient(app) as tc:
        yield tc, fake, next(iter(ideal.items())), settings


AUTH = {"Authorization": "Bearer secret"}


def test_predict_returns_typed_valid_output_with_training_prompt(gateway):
    tc, fake, (doc, expected), settings = gateway
    r = tc.post("/v1/predict", json={"input": doc, "request_id": "r1"}, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True and body["output"] == expected and body["id"] == "r1"
    assert body["usage"] == {"prompt_tokens": 900, "completion_tokens": 250}
    sent = fake.calls[0]
    assert sent["temperature"] == 0.0 and sent["messages"][0]["content"] == settings.system_prompt
    assert sent["model"] == settings.served_model_name
    assert tc.get("/readyz").json()["ready"] is True
    assert tc.get("/v1/info").json()["system_prompt_sha256"] == settings.system_prompt_sha256


def test_invalid_model_output_is_flagged_not_failed(gateway):
    tc, fake, (doc, _), _ = gateway
    fake.reply = lambda user: '<answer>{"total_amount": "$12.00"}</answer>'
    body = tc.post("/v1/predict", json={"input": doc}, headers=AUTH).json()
    assert body["valid"] is False and body["errors"]
    assert body["output"]["total_amount"] == 12.0 and body["output"]["vendor_name"] is None


def test_auth_size_validation_errors(gateway):
    tc, _, (doc, _), _ = gateway
    r = tc.post("/v1/predict", json={"input": doc})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    r = tc.post("/v1/predict", json={"input": "x" * 5001}, headers=AUTH)
    assert r.status_code == 413
    r = tc.post("/v1/predict", json={"input": ""}, headers=AUTH)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = tc.post("/v1/predict", json={"input": "a", "surprise": 1}, headers=AUTH)
    assert r.status_code == 422


def test_idempotency(gateway):
    tc, fake, (doc, _), _ = gateway
    h = {**AUTH, "Idempotency-Key": "k1"}
    a = tc.post("/v1/predict", json={"input": doc}, headers=h).json()
    b = tc.post("/v1/predict", json={"input": doc}, headers=h).json()
    assert a == b and len(fake.calls) == 1
    r = tc.post("/v1/predict", json={"input": doc + " changed"}, headers=h)
    assert r.status_code == 409 and r.json()["error"]["code"] == "idempotency_conflict"


def test_upstream_failures_open_the_breaker(gateway):
    tc, fake, (doc, _), _ = gateway
    fake.status = 500
    for _ in range(2):
        r = tc.post("/v1/predict", json={"input": doc}, headers=AUTH)
        assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
    r = tc.post("/v1/predict", json={"input": doc}, headers=AUTH)
    assert r.status_code == 503 and r.headers["retry-after"] == "999"
    assert len(fake.calls) == 2  # the open breaker shed the third request


def test_core_primitives():
    t = [0.0]
    cb = CircuitBreaker(2, 10, clock=lambda: t[0])
    cb.record(False)
    cb.record(False)
    assert not cb.allow()
    t[0] = 11
    assert cb.allow()  # half-open trial
    cb.record(True)
    assert cb.state == "closed"
    ic = IdempotencyCache(2, 5, clock=lambda: t[0])
    ic.put("a", "f", {"x": 1})
    assert ic.get("a", "f") == {"x": 1} and ic.get("a", "g") is False
    ic.put("b", "f", {})
    ic.put("c", "f", {})
    assert ic.get("a", "f") is None  # evicted (LRU, 2 entries)
    t[0] = 100
    assert ic.get("c", "f") is None  # expired
