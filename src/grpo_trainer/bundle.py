"""`grpo bundle`: package a gated model as something the customer owns and runs with one command.

    <run>/bundle/
      model/                  merged weights + tokenizer (hard-linked when possible)
      serving/settings.json   task, served name, limits, system-prompt sha256
      serving/system_prompt.txt   the exact training prompt (verified by the gateway at startup)
      serving/spec.yaml       extraction spec (extraction tasks)
      gateway/                Dockerfile, pinned requirements, grpo_trainer source (pure Python)
      docker-compose.yml      vLLM (GPU) + gateway; only the gateway port is published
      smoke/examples.jsonl    a few held-out inputs with expected outputs
      smoke_test.py           stdlib-only end-to-end check against the running stack
      BUNDLE.json             sha256 of every file, lineage, gate result
      README.md

Refuses to package a model whose gate failed or never ran, unless explicitly overridden.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config
from .errors import DataError, GateFailedError
from .evaluate import held_out_examples
from .manifest import atomic_write_json
from .tasks import task_for

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
# Unverified tag: matches the pinned vllm==0.32.0. Override with --vllm-image if your registry differs.
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.32.0"
GATEWAY_REQUIREMENTS = """\
# Gateway runtime (pure Python; the GPU work happens in the vLLM container).
fastapi==0.162.4
uvicorn==0.54.0
httpx==0.28.1
pydantic==2.14.3
pyyaml==6.0.3
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            _link_or_copy(p, dst / p.relative_to(src))


def _check_gate(cfg: Config, allow_failed_gate: bool) -> dict[str, Any] | None:
    gate_path = cfg.run_dir / "eval" / "gate.json"
    if not gate_path.is_file():
        if allow_failed_gate:
            log.warning("bundling WITHOUT a gate result (--allow-failed-gate)")
            return None
        raise GateFailedError(
            f"no gate result at {gate_path}; run `grpo gate` first (or --allow-failed-gate)"
        )
    gate = json.loads(gate_path.read_text())
    if not gate.get("passed"):
        if not allow_failed_gate:
            raise GateFailedError(
                "gate failed: " + "; ".join(gate.get("reasons", [])) + " (refusing to bundle)"
            )
        log.warning("bundling a model that FAILED its gate (--allow-failed-gate)")
    return gate


def run_bundle(
    cfg: Config,
    *,
    out_dir: Path | None = None,
    allow_failed_gate: bool = False,
    smoke_examples: int = 5,
    vllm_image: str = DEFAULT_VLLM_IMAGE,
    force: bool = False,
) -> Path:
    complete_path = cfg.run_dir / "final" / "COMPLETE"
    if not complete_path.is_file():
        raise DataError(f"{complete_path} not found; training has not finished")
    complete = json.loads(complete_path.read_text())
    model_dir = Path(complete["eval_model_path"])
    if not (model_dir / "config.json").is_file():
        raise DataError(f"{model_dir} is not a model directory (no config.json); expected merged weights")
    gate = _check_gate(cfg, allow_failed_gate)

    out = out_dir or (cfg.run_dir / "bundle")
    if out.exists():
        if not force:
            raise DataError(f"{out} exists; pass --force to rebuild it")
        shutil.rmtree(out)
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    task = task_for(cfg)
    system_prompt = task.system_prompt(cfg.task.system_prompt)
    served = cfg.run.name
    max_tokens = cfg.eval_max_completion_length
    max_model_len = cfg.vllm.max_model_length or (max_tokens + 2048)
    spec = getattr(task, "spec", None)
    max_input_chars = spec.data.max_input_chars if spec is not None and spec.data is not None else 8000

    log.info("bundling %s -> %s", model_dir, out)
    _copy_tree(model_dir, staging / "model")
    serving = staging / "serving"
    serving.mkdir()
    (serving / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    spec_file = None
    if cfg.task.spec:
        shutil.copy2(cfg.task.spec, serving / "spec.yaml")
        spec_file = "serving/spec.yaml"
    _copy_tree(PACKAGE_DIR, staging / "gateway" / "src" / "grpo_trainer")
    (staging / "gateway" / "requirements.txt").write_text(GATEWAY_REQUIREMENTS)
    (staging / "gateway" / "Dockerfile").write_text(_gateway_dockerfile())

    model_hashes = {
        str(p.relative_to(staging)): _sha256(p) for p in sorted((staging / "model").rglob("*")) if p.is_file()
    }
    prompt_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    bundle_id = hashlib.sha256(
        json.dumps(
            {"model": model_hashes, "prompt": prompt_sha, "grading": task.grading_fingerprint()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    settings = {
        "task": cfg.task.name,
        "spec_file": spec_file,
        "system_prompt_file": "serving/system_prompt.txt",
        "system_prompt_sha256": prompt_sha,
        "served_model_name": served,
        "max_tokens": max_tokens,
        "max_input_chars": max_input_chars,
        "bundle_id": bundle_id,
    }
    atomic_write_json(serving / "settings.json", settings)
    (staging / "docker-compose.yml").write_text(
        _compose(served, max_model_len, cfg.model.dtype, cfg.run.seed, vllm_image)
    )
    (staging / ".env.example").write_text(
        "# Copy to .env. GATEWAY_API_KEY empty = no auth (only on a private network).\n"
        "GATEWAY_PORT=8080\nGATEWAY_API_KEY=\nMAX_CONCURRENCY=32\nREQUEST_TIMEOUT_S=60\n"
        "VLLM_GPU_MEMORY_UTILIZATION=0.90\n"
    )

    smoke = staging / "smoke"
    smoke.mkdir()
    with (smoke / "examples.jsonl").open("w", encoding="utf-8") as f:
        examples = held_out_examples(cfg, task)[:smoke_examples] if smoke_examples else []
        for ex in examples:
            expected = (
                task.ideal_output(ex.ground_truth) if hasattr(task, "ideal_output") else ex.ground_truth
            )
            f.write(
                json.dumps({"id": ex.id, "input": ex.question, "expected": expected}, ensure_ascii=False)
                + "\n"
            )
    (staging / "smoke_test.py").write_text(SMOKE_TEST)
    (staging / "README.md").write_text(_readme(cfg, served, bundle_id, gate))

    files = {
        str(p.relative_to(staging)): model_hashes.get(str(p.relative_to(staging))) or _sha256(p)
        for p in sorted(staging.rglob("*"))
        if p.is_file()
    }
    manifest_path = cfg.run_dir / "manifest.json"
    run_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    atomic_write_json(
        staging / "BUNDLE.json",
        {
            "schema_version": 1,
            "bundle_id": bundle_id,
            "created_at": datetime.now(UTC).isoformat(),
            "served_model_name": served,
            "base_model": cfg.model.name_or_path,
            "base_model_revision": cfg.model.revision,
            "task": cfg.task.name,
            "grading_fingerprint": task.grading_fingerprint(),
            "system_prompt_sha256": prompt_sha,
            "config_fingerprint": run_manifest.get("config_fingerprint"),
            "dataset_fingerprint": run_manifest.get("dataset_fingerprint"),
            "git_revision": run_manifest.get("git_revision"),
            "global_step": complete.get("global_step"),
            "gate": gate,
            "vllm_image": vllm_image,
            "files": files,
        },
    )
    staging.replace(out)
    log.info("bundle %s ready: %s", bundle_id, out)
    return out


def verify_bundle(bundle_dir: Path) -> list[str]:
    """Problems found re-hashing every file listed in BUNDLE.json (empty list = intact)."""
    meta = json.loads((bundle_dir / "BUNDLE.json").read_text())
    problems = []
    for rel, want in meta["files"].items():
        p = bundle_dir / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
        elif _sha256(p) != want:
            problems.append(f"modified {rel}")
    return problems


def _gateway_dockerfile() -> str:
    return """\
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src BUNDLE_DIR=/bundle
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src /app/src
RUN useradd --system --uid 10001 gateway
USER gateway
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \\
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
CMD ["uvicorn", "grpo_trainer.serve.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
"""


def _compose(served: str, max_model_len: int, dtype: str, seed: int, image: str) -> str:
    return f"""\
# docker compose up -d   (needs an NVIDIA GPU + NVIDIA Container Toolkit on the host)
services:
  vllm:
    image: {image}
    command:
      - --model=/model
      - --served-model-name={served}
      - --max-model-len={max_model_len}
      - --dtype={dtype}
      - --seed={seed}
      - --gpu-memory-utilization=${{VLLM_GPU_MEMORY_UTILIZATION:-0.90}}
    volumes:
      - ./model:/model:ro
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test:
        - CMD
        - python3
        - -c
        - "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"
      interval: 15s
      timeout: 5s
      retries: 40
      start_period: 60s
    restart: unless-stopped
    # no published port: only the gateway talks to vLLM

  gateway:
    build: ./gateway
    environment:
      VLLM_BASE_URL: http://vllm:8000
      BUNDLE_DIR: /bundle
      GATEWAY_API_KEY: ${{GATEWAY_API_KEY:-}}
      MAX_CONCURRENCY: ${{MAX_CONCURRENCY:-32}}
      REQUEST_TIMEOUT_S: ${{REQUEST_TIMEOUT_S:-60}}
    volumes:
      - ./serving:/bundle/serving:ro
    ports:
      - "${{GATEWAY_PORT:-8080}}:8080"
    depends_on:
      vllm:
        condition: service_healthy
    restart: unless-stopped
"""


def _readme(cfg: Config, served: str, bundle_id: str, gate: dict[str, Any] | None) -> str:
    gate_line = "passed" if gate and gate.get("passed") else ("FAILED" if gate else "not run")
    return f"""\
# {served} (bundle {bundle_id})

Self-hosted model for task `{cfg.task.name}`, fine-tuned from `{cfg.model.name_or_path}` with GRPO.
Promotion gate: **{gate_line}**. Full evidence: the proof report delivered with this bundle.

## Run

Requires an NVIDIA GPU host with Docker and the NVIDIA Container Toolkit.

    cp .env.example .env            # set GATEWAY_API_KEY for anything beyond a private network
    docker compose up -d --build
    python3 smoke_test.py           # end-to-end check against held-out examples

## API (v1)

    curl -s localhost:8080/v1/predict \\
      -H 'content-type: application/json' \\
      -H "Authorization: Bearer $GATEWAY_API_KEY" \\
      -H 'Idempotency-Key: <unique per document>' \\
      -d '{{"input": "<document text>"}}'

Response: `output` (typed result), `valid` (false = output broke the contract; route to review),
`errors`, `usage`, `latency_ms`. Errors use `{{"error": {{"code", "message", "request_id"}}}}`.
`GET /readyz` reports when the model is loaded; `GET /v1/info` shows the bundle id and prompt hash.

## Integrity

`BUNDLE.json` lists the sha256 of every file. The gateway refuses to start if the system prompt
does not match its recorded hash: the model was trained on that exact prompt.
"""


SMOKE_TEST = '''\
#!/usr/bin/env python3
"""End-to-end smoke test for a running bundle. Stdlib only.

    python3 smoke_test.py [--url http://localhost:8080] [--api-key KEY]

Fails (exit 1) if the gateway is not ready, any request errors, or any response is not contract-valid.
Exact-match against the expected output is reported for information only.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path


def call(url, payload=None, key=None, timeout=120):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"content-type": "application/json",
                                          **({"authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--api-key", default=os.environ.get("GATEWAY_API_KEY") or None)
    ap.add_argument("--wait", type=int, default=600, help="seconds to wait for /readyz")
    a = ap.parse_args()
    deadline = time.time() + a.wait
    while True:
        try:
            if call(a.url + "/readyz", timeout=5)[0] == 200:
                break
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if time.time() > deadline:
            print("FAIL: gateway not ready", file=sys.stderr)
            return 1
        time.sleep(5)
    lines = (Path(__file__).parent / "smoke" / "examples.jsonl").read_text().splitlines()
    examples = [json.loads(line) for line in lines if line.strip()]
    failures, exact = 0, 0
    for ex in examples:
        try:
            payload = {"input": ex["input"], "request_id": ex["id"]}
            status, body = call(a.url + "/v1/predict", payload, a.api_key)
        except urllib.error.HTTPError as e:
            print(f"FAIL {ex['id']}: HTTP {e.code} {e.read()[:200]!r}")
            failures += 1
            continue
        ok = body.get("valid") is True
        match = body.get("output") == ex["expected"]
        exact += match
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {ex['id']}: valid={body.get('valid')} exact_match={match} "
              f"latency_ms={body.get('latency_ms')} errors={body.get('errors')}")
    print(f"{len(examples) - failures}/{len(examples)} contract-valid, {exact}/{len(examples)} exact match")
    return 1 if failures or not examples else 0


if __name__ == "__main__":
    sys.exit(main())
'''
