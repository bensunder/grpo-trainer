"""Gateway core: bundle settings, idempotency cache, circuit breaker. Pure Python, no web framework.

The gateway sits in front of vLLM's OpenAI-compatible server and owns everything that must match training
exactly: the system prompt (verified against its recorded sha256 at startup), the user-message shape,
greedy decoding, and the task's output parser and contract validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SETTINGS_FILE = "serving/settings.json"


class BundleError(RuntimeError):
    """The bundle on disk is incomplete or tampered with; the gateway must not start."""


@dataclass(frozen=True)
class GatewaySettings:
    bundle_dir: Path
    task: str
    spec_file: str | None  # relative to bundle_dir
    system_prompt: str
    system_prompt_sha256: str
    served_model_name: str
    max_tokens: int
    max_input_chars: int
    bundle_id: str
    # runtime (environment)
    vllm_base_url: str = "http://vllm:8000"
    request_timeout_s: float = 60.0
    max_concurrency: int = 32
    breaker_failures: int = 5
    breaker_cooldown_s: float = 30.0
    idempotency_entries: int = 10_000
    idempotency_ttl_s: float = 24 * 3600
    api_key: str | None = None

    @classmethod
    def load(cls, bundle_dir: str | Path, env: dict[str, str] | None = None) -> GatewaySettings:
        bundle_dir = Path(bundle_dir)
        env = dict(os.environ if env is None else env)
        path = bundle_dir / SETTINGS_FILE
        if not path.is_file():
            raise BundleError(f"{path} not found; BUNDLE_DIR must point at a `grpo bundle` output")
        raw = json.loads(path.read_text(encoding="utf-8"))
        prompt_path = bundle_dir / raw["system_prompt_file"]
        prompt = prompt_path.read_text(encoding="utf-8")
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if sha != raw["system_prompt_sha256"]:
            raise BundleError(f"{prompt_path} does not match its recorded sha256; the bundle was modified")

        def num(key: str, default: float, cast: Callable[[str], Any] = float) -> Any:
            v = env.get(key)
            return default if v in (None, "") else cast(v)

        return cls(
            bundle_dir=bundle_dir,
            task=raw["task"],
            spec_file=raw.get("spec_file"),
            system_prompt=prompt,
            system_prompt_sha256=sha,
            served_model_name=raw["served_model_name"],
            max_tokens=int(raw["max_tokens"]),
            max_input_chars=int(num("MAX_INPUT_CHARS", raw["max_input_chars"], int)),
            bundle_id=raw["bundle_id"],
            vllm_base_url=env.get("VLLM_BASE_URL", "http://vllm:8000").rstrip("/"),
            request_timeout_s=num("REQUEST_TIMEOUT_S", 60.0),
            max_concurrency=num("MAX_CONCURRENCY", 32, int),
            breaker_failures=num("BREAKER_FAILURES", 5, int),
            breaker_cooldown_s=num("BREAKER_COOLDOWN_S", 30.0),
            idempotency_entries=num("IDEMPOTENCY_ENTRIES", 10_000, int),
            idempotency_ttl_s=num("IDEMPOTENCY_TTL_S", 24 * 3600.0),
            api_key=env.get("GATEWAY_API_KEY") or None,
        )

    def build_task(self):  # noqa: ANN201 - returns a Task
        from ..tasks import get_task

        spec = str(self.bundle_dir / self.spec_file) if self.spec_file else None
        return get_task(self.task, spec=spec)

    def messages(self, user_input: str) -> list[dict[str, str]]:
        """Exactly the conversation shape used in training (tasks.base.Example.messages)."""
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_input}]


class CircuitBreaker:
    """Opens after N consecutive upstream failures; half-opens after a cooldown (one trial request)."""

    def __init__(self, failures: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.failures, self.cooldown_s, self._clock = failures, cooldown_s, clock
        self._consecutive = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._clock() - self._opened_at >= self.cooldown_s:
                self._opened_at = self._clock()  # half-open: let one through, re-arm the window
                return True
            return False

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._consecutive, self._opened_at = 0, None
            else:
                self._consecutive += 1
                if self._consecutive >= self.failures:
                    self._opened_at = self._clock()

    @property
    def state(self) -> str:
        return "closed" if self._opened_at is None else "open"


class IdempotencyCache:
    """Idempotency-Key -> (request fingerprint, response). Bounded LRU with TTL, in-process.

    Same key + same body -> the stored response. Same key + different body -> conflict (409).
    Single-replica scope: behind several gateway replicas, route by key or use a shared store.
    """

    def __init__(self, max_entries: int, ttl_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._data: OrderedDict[str, tuple[float, str, dict[str, Any]]] = OrderedDict()
        self.max_entries, self.ttl_s, self._clock = max_entries, ttl_s, clock
        self._lock = threading.Lock()

    @staticmethod
    def fingerprint(body: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def get(self, key: str, fp: str) -> dict[str, Any] | None | bool:
        """Stored response, None (unknown key), or False (key reused with a different body)."""
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            at, stored_fp, resp = hit
            if self._clock() - at > self.ttl_s:
                del self._data[key]
                return None
            if stored_fp != fp:
                return False
            self._data.move_to_end(key)
            return resp

    def put(self, key: str, fp: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = (self._clock(), fp, response)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
