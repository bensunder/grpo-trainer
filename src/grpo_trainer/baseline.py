"""Frontier baseline: run the customer's current API model on the exact held-out split, prompt and grader.

Without this there is no replacement claim, only "the small model got better". The output is an
eval-compatible report (`baseline-<name>.json` + `.examples.jsonl`) plus measured cost and latency, so the
proof report can pair it question-by-question with the trained model.

This stage spends money. Guards, in order:
  * `--dry-run` prints a worst-case cost estimate and sends nothing;
  * a run whose worst-case estimate exceeds `baseline.max_cost_usd` is refused before the first call;
  * actual spend is tracked and the run stops when it reaches the limit;
  * every successful response is cached (keyed by provider, model, prompt and parameters), so a re-run
    after an interruption only pays for what is missing;
  * consecutive failures trip a circuit breaker; 401/403 abort immediately.

Transport is stdlib (urllib) behind a small interface so tests run without network or extra packages.
API keys are read from the environment and never written to disk or logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .config import BaselineProvider, Config
from .errors import BudgetError, ConfigError, ProviderError
from .evaluate import Scorer, aggregate_detail, held_out_examples, system_prompt_sha
from .manifest import atomic_write_json
from .tasks import Example, dataset_fingerprint, task_for

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com"}
ANTHROPIC_VERSION = "2023-06-01"
CHARS_PER_TOKEN_ESTIMATE = 3.0  # conservative (over-estimates tokens) for the pre-run budget check
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
MAX_RETRY_AFTER_S = 60.0


# ------------------------------------------------------------------ transport


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class TransportError(Exception):
    """Network-level failure (DNS, connect, reset, timeout). Retryable."""


class Transport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], payload: dict[str, Any], timeout: float
    ) -> HttpResponse: ...


class UrllibTransport:
    """HTTPS POST via urllib. Honours HTTPS_PROXY and the system CA store."""

    def post(
        self, url: str, headers: Mapping[str, str], payload: dict[str, Any], timeout: float
    ) -> HttpResponse:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={**headers, "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL comes from config
                return HttpResponse(resp.status, dict(resp.headers), resp.read())
        except urllib.error.HTTPError as e:
            return HttpResponse(e.code, dict(e.headers or {}), e.read() or b"")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise TransportError(str(e)) from e


# ------------------------------------------------------------------ resilience primitives


class RateLimiter:
    """Thread-safe request spacing: at most `per_minute` request starts per rolling minute."""

    def __init__(
        self,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / per_minute
        self._next = 0.0
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self._interval
        delay = start - now
        if delay > 0:
            self._sleep(delay)


class CircuitBreaker:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self._consecutive = 0
        self._lock = threading.Lock()
        self.open = False

    def success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def failure(self) -> None:
        with self._lock:
            self._consecutive += 1
            if self._consecutive >= self.threshold:
                self.open = True


class Budget:
    def __init__(self, limit_usd: float) -> None:
        self.limit = limit_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent += usd

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


# ------------------------------------------------------------------ provider client


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    truncated: bool
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float


class RequestFailed(Exception):
    """One request failed for good (non-retryable status, or retries exhausted). Not fatal to the run."""


def _retry_after(headers: Mapping[str, str]) -> float | None:
    for k, v in headers.items():
        if k.lower() == "retry-after":
            try:
                return min(MAX_RETRY_AFTER_S, max(0.0, float(v)))
            except ValueError:
                return None
    return None


class ProviderClient:
    def __init__(
        self,
        provider: BaselineProvider,
        api_key: str,
        transport: Transport,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.p = provider
        self._key = api_key
        self._transport = transport
        self._sleep, self._clock = sleep, clock
        self._rng = rng or random.Random()

    # request shape per API family ---------------------------------------------------------

    def build_request(
        self, messages: list[dict[str, str]], max_tokens: int, seed: int
    ) -> tuple[str, dict, dict]:
        p = self.p
        if p.kind == "anthropic":
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            payload: dict[str, Any] = {
                "model": p.model,
                "max_tokens": max_tokens,
                "temperature": p.temperature,
                "messages": [m for m in messages if m["role"] != "system"],
                **p.extra_body,
            }
            if system:
                payload["system"] = system
            base = (p.base_url or DEFAULT_BASE_URL["anthropic"]).rstrip("/")
            headers = {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION}
            return f"{base}/v1/messages", headers, payload
        token_param = p.max_tokens_param or "max_completion_tokens"
        payload = {
            "model": p.model,
            "messages": messages,
            "temperature": p.temperature,
            token_param: max_tokens,
            "seed": seed,
            **p.extra_body,
        }
        if p.kind == "azure_openai":
            base = (p.base_url or "").rstrip("/")
            url = f"{base}/openai/deployments/{p.model}/chat/completions?api-version={p.api_version}"
            return url, {"api-key": self._key}, payload
        base = (p.base_url or DEFAULT_BASE_URL["openai"]).rstrip("/")
        return f"{base}/chat/completions", {"authorization": f"Bearer {self._key}"}, payload

    def parse_response(self, body: bytes) -> tuple[str, str | None, bool, int | None, int | None]:
        data = json.loads(body)
        if self.p.kind == "anthropic":
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            reason = data.get("stop_reason")
            usage = data.get("usage") or {}
            return text, reason, reason == "max_tokens", usage.get("input_tokens"), usage.get("output_tokens")
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        return text, reason, reason == "length", usage.get("prompt_tokens"), usage.get("completion_tokens")

    # one request with retries ---------------------------------------------------------------

    def complete(self, messages: list[dict[str, str]], max_tokens: int, seed: int) -> Completion:
        url, headers, payload = self.build_request(messages, max_tokens, seed)
        last = "no attempt made"
        for attempt in range(self.p.max_retries + 1):
            t0 = self._clock()
            retry_after = None
            try:
                resp = self._transport.post(url, headers, payload, self.p.timeout_s)
            except TransportError as e:
                last = f"transport error: {e}"
            else:
                if 200 <= resp.status < 300:
                    try:
                        text, reason, trunc, tin, tout = self.parse_response(resp.body)
                    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
                        raise RequestFailed(f"unparseable response: {e}") from e
                    return Completion(text, reason, trunc, tin, tout, self._clock() - t0)
                snippet = resp.body[:300].decode("utf-8", "replace")
                if resp.status in (401, 403):
                    raise ProviderError(
                        f"{self.p.name}: HTTP {resp.status} (check {self.p.api_key_env}): {snippet}"
                    )
                if resp.status not in RETRYABLE_STATUS:
                    raise RequestFailed(f"HTTP {resp.status}: {snippet}")
                last = f"HTTP {resp.status}: {snippet}"
                retry_after = _retry_after(resp.headers)
            if attempt == self.p.max_retries:
                break
            backoff = min(30.0, 2.0**attempt) * (0.5 + self._rng.random() / 2)
            self._sleep(retry_after if retry_after is not None else backoff)
        raise RequestFailed(f"gave up after {self.p.max_retries + 1} attempts: {last}")


# ------------------------------------------------------------------ cost, cache, estimate


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in (0, 1])."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def request_cost(p: BaselineProvider, input_tokens: int | None, output_tokens: int | None) -> float:
    return ((input_tokens or 0) * p.input_usd_per_mtok + (output_tokens or 0) * p.output_usd_per_mtok) / 1e6


def estimate_cost(
    p: BaselineProvider, messages: list[list[dict[str, str]]], max_tokens: int
) -> dict[str, float]:
    in_tok = sum(sum(len(m["content"]) for m in conv) for conv in messages) / CHARS_PER_TOKEN_ESTIMATE
    out_worst = len(messages) * max_tokens
    return {
        "requests": len(messages),
        "input_tokens_est": round(in_tok),
        "output_tokens_worst_case": out_worst,
        "usd_worst_case": request_cost(p, int(in_tok), out_worst),
        "usd_if_half_max_output": request_cost(p, int(in_tok), out_worst // 2),
    }


def cache_key(p: BaselineProvider, messages: list[dict[str, str]], max_tokens: int, seed: int) -> str:
    ident = {
        "kind": p.kind,
        "model": p.model,
        "base_url": p.base_url,
        "api_version": p.api_version,
        "temperature": p.temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": p.extra_body,
        "messages": messages,
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


class ResponseCache:
    """Append-only JSONL of successful responses. A torn last line (crash mid-write) is ignored."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.entries[rec["key"]] = rec
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

    def put(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.entries[rec["key"]] = rec


# ------------------------------------------------------------------ run


def find_provider(cfg: Config, name: str) -> BaselineProvider:
    for p in cfg.baseline.providers:
        if p.name == name:
            return p
    known = [p.name for p in cfg.baseline.providers]
    raise ConfigError(f"no baseline provider {name!r} in config; declared: {known or 'none'}")


def baseline_paths(cfg: Config, name: str) -> tuple[Path, Path, Path]:
    d = cfg.run_dir / "eval"
    return (
        d / f"baseline-{name}.json",
        d / f"baseline-{name}.examples.jsonl",
        d / f"baseline-{name}.cache.jsonl",
    )


def run_baseline(
    cfg: Config,
    provider_name: str,
    *,
    dry_run: bool = False,
    transport: Transport | None = None,
    env: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Path | dict[str, Any]:
    """Returns the report path, or (dry run) the cost estimate."""
    p = find_provider(cfg, provider_name)
    task = task_for(cfg)
    examples = held_out_examples(cfg, task)
    system_prompt = task.system_prompt(cfg.task.system_prompt)
    max_tokens = p.max_tokens or cfg.eval_max_completion_length
    seed = cfg.run.seed
    convs = [e.messages(system_prompt) for e in examples]
    report_path, examples_path, cache_path = baseline_paths(cfg, p.name)
    cache = ResponseCache(cache_path)
    keys = [cache_key(p, c, max_tokens, seed) for c in convs]
    todo = [i for i, k in enumerate(keys) if k not in cache.entries]

    estimate = (
        estimate_cost(p, [convs[i] for i in todo], max_tokens) if todo else estimate_cost(p, [], max_tokens)
    )
    estimate.update({"cached": len(examples) - len(todo), "limit_usd": cfg.baseline.max_cost_usd})
    if dry_run:
        return {"provider": p.name, "model": p.model, "examples": len(examples), **estimate}
    if todo and estimate["usd_worst_case"] > cfg.baseline.max_cost_usd:
        raise BudgetError(
            f"{p.name}: worst-case estimate ${estimate['usd_worst_case']:.2f} "
            f"for {len(todo)} uncached requests "
            f"exceeds baseline.max_cost_usd=${cfg.baseline.max_cost_usd:.2f}. Lower eval.limit or "
            f"{p.name}.max_tokens, or raise the limit (--set baseline.max_cost_usd=...)."
        )

    env = os.environ if env is None else env
    failures: dict[int, str] = {}
    if todo:
        key = env.get(p.api_key_env, "")
        if not key:
            raise ProviderError(f"{p.name}: environment variable {p.api_key_env} is not set")
        client = ProviderClient(p, key, transport or UrllibTransport(), sleep=sleep)
        limiter = RateLimiter(p.requests_per_minute, sleep=sleep)
        breaker = CircuitBreaker(cfg.baseline.circuit_breaker_failures)
        budget = Budget(cfg.baseline.max_cost_usd)
        fatal: list[BaseException] = []
        lock = threading.Lock()

        def work(i: int) -> None:
            if fatal or breaker.open or budget.exhausted:
                return
            limiter.acquire()
            try:
                c = client.complete(convs[i], max_tokens, seed)
            except RequestFailed as e:
                breaker.failure()
                with lock:
                    failures[i] = str(e)
                return
            except ProviderError as e:
                with lock:
                    fatal.append(e)
                return
            breaker.success()
            usd = request_cost(p, c.input_tokens, c.output_tokens)
            budget.charge(usd)
            cache.put(
                {
                    "key": keys[i],
                    "id": examples[i].id,
                    "text": c.text,
                    "finish_reason": c.finish_reason,
                    "truncated": c.truncated,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "latency_s": c.latency_s,
                    "cost_usd": usd,
                    "usage_reported": c.input_tokens is not None and c.output_tokens is not None,
                }
            )

        log.info(
            "baseline %s (%s): %d requests (%d cached)", p.name, p.model, len(todo), len(examples) - len(todo)
        )
        with ThreadPoolExecutor(max_workers=p.max_concurrency) as pool:
            list(pool.map(work, todo))
        if fatal:
            raise fatal[0]
        if breaker.open:
            raise ProviderError(
                f"{p.name}: {cfg.baseline.circuit_breaker_failures} consecutive failures; aborted. "
                f"Completed responses are cached in {cache_path}; re-run to resume. Last error: "
                f"{next(reversed(failures.values()), 'n/a')}"
            )
        if budget.exhausted:
            raise BudgetError(
                f"{p.name}: spend reached ${budget.spent:.2f} (limit ${budget.limit:.2f}); stopped. "
                f"Responses so far are cached in {cache_path}."
            )

    missing = [i for i, k in enumerate(keys) if k not in cache.entries]
    error_rate = len(missing) / len(examples)
    if error_rate > cfg.baseline.max_error_rate:
        sample = [f"{examples[i].id}: {failures.get(i, 'not attempted')}" for i in missing[:5]]
        raise ProviderError(
            f"{p.name}: {len(missing)}/{len(examples)} requests failed ({error_rate:.1%} > "
            f"baseline.max_error_rate {cfg.baseline.max_error_rate:.1%}). Re-run to retry only those. "
            f"First failures: {sample}"
        )
    return _write_report(
        cfg, p, task, examples, keys, cache, failures, max_tokens, system_prompt, report_path, examples_path
    )


def _write_report(
    cfg,
    p,
    task,
    examples: list[Example],
    keys,
    cache,
    failures,
    max_tokens,
    system_prompt,
    report_path: Path,
    examples_path: Path,
) -> Path:  # noqa: ANN001 - internal
    scorer = Scorer(cfg, task)
    rows = []
    latencies, costs = [], []
    out_tokens = n_trunc = 0
    for i, ex in enumerate(examples):
        rec_c = cache.entries.get(keys[i])
        text = rec_c["text"] if rec_c else ""
        row = scorer.row(ex, text, None)
        row.update(
            {
                "truncated": bool(rec_c and rec_c["truncated"]),
                "tokens": (rec_c or {}).get("output_tokens") or 0,
                "output": text,
                "error": None if rec_c else failures.get(i, "missing"),
                "latency_ms": round(rec_c["latency_s"] * 1000, 1) if rec_c else None,
                "cost_usd": rec_c["cost_usd"] if rec_c else 0.0,
            }
        )
        if rec_c:
            latencies.append(rec_c["latency_s"])
            costs.append(rec_c["cost_usd"])
            out_tokens += row["tokens"]
            n_trunc += row["truncated"]
        rows.append(row)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with examples_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(examples)
    answered = len(costs)
    report: dict[str, Any] = {
        "label": f"baseline-{p.name}",
        "kind": "baseline",
        "model": p.model,
        "provider": {"name": p.name, "kind": p.kind, "model": p.model, "base_url": p.base_url},
        "created_at": datetime.now(UTC).isoformat(),
        "task": cfg.task.name,
        "split": cfg.task.eval_split,
        "num_examples": n,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "grading_fingerprint": task.grading_fingerprint(),
        "max_completion_length": max_tokens,
        "system_prompt_sha": system_prompt_sha(system_prompt),
        "greedy": {
            "accuracy": sum(r["correct"] for r in rows) / n,
            "lenient_accuracy": sum(r["lenient_correct"] for r in rows) / n,
            "format_rate": sum(r["format"] for r in rows) / n,
            "truncation_rate": n_trunc / n,
            "mean_tokens": out_tokens / answered if answered else 0.0,
        },
        "cost": {
            "total_usd": sum(costs),
            "usd_per_1k": (sum(costs) / answered * 1000) if answered else None,
            "input_usd_per_mtok": p.input_usd_per_mtok,
            "output_usd_per_mtok": p.output_usd_per_mtok,
            "usage_reported_rate": (
                sum(1 for i in range(n) if (cache.entries.get(keys[i]) or {}).get("usage_reported")) / n
            ),
        },
        "latency_ms": {
            "p50": statistics.median(latencies) * 1000 if latencies else None,
            "p95": percentile(latencies, 0.95) * 1000 if latencies else None,
            "mean": (sum(latencies) / len(latencies) * 1000) if latencies else None,
        },
        "error_rate": (n - answered) / n,
        "examples_file": str(examples_path),
    }
    detail = aggregate_detail(rows)
    if detail is not None:
        report["fields"] = detail
    atomic_write_json(report_path, report)
    log.info(
        "baseline %s: accuracy %.4f format %.4f cost $%.4f (%s per 1k) -> %s",
        p.name,
        report["greedy"]["accuracy"],
        report["greedy"]["format_rate"],
        report["cost"]["total_usd"],
        f"${report['cost']['usd_per_1k']:.2f}" if report["cost"]["usd_per_1k"] is not None else "n/a",
        report_path,
    )
    return report_path
