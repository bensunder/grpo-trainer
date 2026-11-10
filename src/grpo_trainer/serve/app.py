"""HTTP gateway (FastAPI) in front of vLLM. Versioned contract:

    POST /v1/predict   {"input": "<document text>", "request_id": "optional"}
                       headers: Authorization: Bearer <GATEWAY_API_KEY> (if set), Idempotency-Key (optional)
      200 {"id", "model", "bundle_id", "output", "valid", "errors", "finish_reason", "usage", "latency_ms"}
          `valid` is false when the model output breaks the contract; `output` is then best-effort (or null)
          so the caller can route it to human review instead of failing the batch.
      401 unauthorized | 409 idempotency_conflict | 413 input_too_large | 422 validation_error
      429 overloaded (Retry-After) | 502 upstream_error | 503 upstream_unavailable | 504 upstream_timeout
    GET /healthz  liveness          GET /readyz  vLLM reachable and serving this model
    GET /v1/info  bundle id, task, prompt sha, limits

Errors share one envelope: {"error": {"code", "message", "request_id"}}. Document text is never logged.

Run: uvicorn grpo_trainer.serve.app:app (reads BUNDLE_DIR) or create_app(settings) in tests.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .core import CircuitBreaker, GatewaySettings, IdempotencyCache

log = logging.getLogger("grpo_trainer.gateway")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str = Field(min_length=1)
    request_id: str | None = Field(default=None, max_length=128)


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class PredictResponse(BaseModel):
    id: str
    model: str
    bundle_id: str
    output: Any
    valid: bool
    errors: list[str]
    finish_reason: str | None
    usage: Usage
    latency_ms: float


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def _error(
    status: int, code: str, message: str, request_id: str | None, headers: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers=headers,
    )


def create_app(settings: GatewaySettings, client: httpx.AsyncClient | None = None) -> FastAPI:
    task = settings.build_task()
    breaker = CircuitBreaker(settings.breaker_failures, settings.breaker_cooldown_s)
    idem = IdempotencyCache(settings.idempotency_entries, settings.idempotency_ttl_s)
    slots = asyncio.Semaphore(settings.max_concurrency)
    http = client or httpx.AsyncClient(
        base_url=settings.vllm_base_url,
        timeout=httpx.Timeout(settings.request_timeout_s, connect=5.0),
        limits=httpx.Limits(max_connections=settings.max_concurrency * 2),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # noqa: ANN202
        yield
        if client is None:  # only close a client we created
            await http.aclose()

    app = FastAPI(
        title="grpo-trainer gateway", version="1", docs_url="/docs", redoc_url=None, lifespan=lifespan
    )

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, getattr(request.state, "rid", None), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        return _error(
            422, "validation_error", detail or "invalid request", getattr(request.state, "rid", None)
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        try:
            r = await http.get("/v1/models", timeout=3.0)
            served = {m.get("id") for m in r.json().get("data", [])} if r.status_code == 200 else set()
        except (httpx.HTTPError, ValueError):
            served = set()
        ready = settings.served_model_name in served
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"ready": ready, "model": settings.served_model_name, "breaker": breaker.state},
        )

    @app.get("/v1/info")
    async def info() -> dict[str, Any]:
        return {
            "bundle_id": settings.bundle_id,
            "task": settings.task,
            "model": settings.served_model_name,
            "system_prompt_sha256": settings.system_prompt_sha256,
            "max_input_chars": settings.max_input_chars,
            "max_tokens": settings.max_tokens,
            "grading_fingerprint": task.grading_fingerprint(),
        }

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(
        body: PredictRequest,
        request: Request,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, max_length=256),
    ) -> PredictResponse:
        rid = body.request_id or str(uuid.uuid4())
        request.state.rid = rid
        if settings.api_key and authorization != f"Bearer {settings.api_key}":
            raise ApiError(401, "unauthorized", "missing or invalid bearer token")
        if len(body.input) > settings.max_input_chars:
            raise ApiError(413, "input_too_large", f"input exceeds {settings.max_input_chars} characters")
        fp = IdempotencyCache.fingerprint({"input": body.input})
        if idempotency_key:
            hit = idem.get(idempotency_key, fp)
            if hit is False:
                raise ApiError(409, "idempotency_conflict", "Idempotency-Key reused with a different input")
            if hit is not None:
                return PredictResponse(**hit)
        if not breaker.allow():
            raise ApiError(
                503,
                "upstream_unavailable",
                "model server failing; circuit open",
                {"Retry-After": str(int(settings.breaker_cooldown_s))},
            )
        if slots.locked():
            raise ApiError(429, "overloaded", "too many concurrent requests", {"Retry-After": "1"})
        async with slots:
            t0 = time.monotonic()
            payload = {
                "model": settings.served_model_name,
                "messages": settings.messages(body.input),
                "temperature": 0.0,
                "max_tokens": settings.max_tokens,
            }
            try:
                r = await http.post("/v1/chat/completions", json=payload)
            except httpx.TimeoutException:
                breaker.record(False)
                raise ApiError(504, "upstream_timeout", "model server timed out") from None
            except httpx.HTTPError as e:
                breaker.record(False)
                raise ApiError(
                    503, "upstream_unavailable", f"model server unreachable: {type(e).__name__}"
                ) from None
            if r.status_code != 200:
                breaker.record(r.status_code < 500)
                raise ApiError(502, "upstream_error", f"model server returned HTTP {r.status_code}")
            try:
                data = r.json()
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError):
                breaker.record(False)
                raise ApiError(502, "upstream_error", "malformed model server response") from None
            breaker.record(True)
            pred = task.parse_prediction(text)
            usage = data.get("usage") or {}
            resp = PredictResponse(
                id=rid,
                model=settings.served_model_name,
                bundle_id=settings.bundle_id,
                output=pred.output,
                valid=pred.valid,
                errors=list(pred.errors),
                finish_reason=choice.get("finish_reason"),
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens")
                ),
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )
        log.info(
            "predict id=%s valid=%s finish=%s latency_ms=%.1f",
            rid,
            resp.valid,
            resp.finish_reason,
            resp.latency_ms,
        )
        if idempotency_key:
            idem.put(idempotency_key, fp, resp.model_dump())
        return resp

    return app


def _from_env() -> FastAPI:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    return create_app(GatewaySettings.load(os.environ.get("BUNDLE_DIR", "/bundle")))


def __getattr__(name: str) -> Any:  # `uvicorn grpo_trainer.serve.app:app` builds lazily from the environment
    if name == "app":
        return _from_env()
    raise AttributeError(name)
