"""Bounded inference gateway with pinned deployment leases and safe model switching."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import ipaddress
import json
from pathlib import Path
import time
from typing import Any, Callable, Literal
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .contracts import Backend, GenerationConfig
from .registry import ArtifactError, Registry


@dataclass(frozen=True)
class Settings:
    registry_path: Path = Path("registry")
    initial_model: str | None = None
    host: str = "127.0.0.1"
    api_key: str | None = field(default=None, repr=False)
    admin_key: str | None = field(default=None, repr=False)
    max_input_tokens: int = 2048
    max_output_tokens: int = 512
    max_body_bytes: int = 128 * 1024
    max_concurrency: int = 1
    max_queue: int = 8
    queue_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 120.0
    switch_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "localhost", "::1"} and (not self.api_key or not self.admin_key):
            raise ValueError("Non-local serving requires separate API and administrative keys")
        if self.api_key and self.admin_key and self.api_key == self.admin_key:
            raise ValueError("Administrative and inference keys must differ")
        for value in (self.max_input_tokens, self.max_output_tokens, self.max_body_bytes, self.max_concurrency):
            if value < 1:
                raise ValueError("Limits must be positive")
        if self.max_queue < 0 or min(self.queue_timeout_seconds, self.request_timeout_seconds,
                                     self.switch_timeout_seconds) <= 0:
            raise ValueError("Queue capacity and deadlines are invalid")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1, max_length=64)
    messages: list[ChatMessage] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    stream: bool = False
    stop: str | list[str] | None = None
    stream_options: dict[str, bool] | None = None


class PromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_id: str = Field(pattern=r"^[a-f0-9]{64}$")


class InferenceFailure(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


@dataclass
class Deployment:
    backend: Backend
    package: Path
    manifest: dict[str, Any]


@dataclass
class Lease:
    deployment: Deployment
    queued_at: float
    admitted_at: float


def default_backend_factory(manifest: dict[str, Any]) -> Backend:
    if manifest["runtime"]["backend"] == "transformers":
        from .backends import TransformersBackend
        return TransformersBackend()
    from .backends import VLLMBackend
    return VLLMBackend()


class ModelManager:
    def __init__(self, registry: Registry, settings: Settings, factory: Callable[[dict], Backend]):
        self.registry, self.settings, self.factory = registry, settings, factory
        self.current: Deployment | None = None
        self.switching = False
        self.closing = False
        self.last_load_error: str | None = None
        self.leases = 0
        self.running = 0
        self._condition = asyncio.Condition()
        self._switch_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(settings.max_concurrency)

    async def start(self) -> None:
        artifact_id = self.settings.initial_model or self.registry.deployment()["current"]
        if artifact_id:
            await self.switch(artifact_id)

    async def switch(self, artifact_id: str) -> dict[str, Any]:
        """Stage with the old deployment resident; commit only after a successful load.

        This trades temporary double residency for failure isolation. A small GPU
        can safely refuse a switch instead of sacrificing its healthy deployment.
        """
        async with self._switch_lock:
            package, manifest = await asyncio.to_thread(self.registry.verify, artifact_id)
            if self.current and self.current.manifest["artifact_id"] == artifact_id:
                return self.registry.deployment()
            async with self._condition:
                if self.closing:
                    raise InferenceFailure(503, "shutting_down", "Gateway is shutting down")
                self.switching = True
            candidate = None
            try:
                async with asyncio.timeout(self.settings.switch_timeout_seconds):
                    async with self._condition:
                        await self._condition.wait_for(lambda: self.leases == 0)
                    candidate = self.factory(manifest)
                    await candidate.load(package, manifest)
                    if candidate.health().get("ready") is not True:
                        raise ArtifactError("Candidate backend did not become ready")
                    # Pointer failures occur before changing the in-memory deployment.
                    state = await asyncio.to_thread(self.registry.promote, artifact_id)
                    old = self.current
                    self.current = Deployment(candidate, package, manifest)
                    candidate = None
                    self.last_load_error = None
                    if old:
                        try:
                            await old.backend.unload()
                        except Exception:
                            # Serving succeeded, but fail readiness conservatively if
                            # the prior runtime could not release its GPU allocation.
                            self.last_load_error = "Previous backend cleanup failed"
                    return state
            except BaseException as exc:
                self.last_load_error = type(exc).__name__
                raise
            finally:
                try:
                    if candidate:
                        await candidate.unload()
                finally:
                    async with self._condition:
                        self.switching = False
                        self._condition.notify_all()

    @asynccontextmanager
    async def acquire(self, artifact_id: str):
        queued_at = time.perf_counter()
        async with self._condition:
            if self.closing or self.switching or self.current is None:
                raise InferenceFailure(503, "model_unavailable", "Model is not ready")
            if self.current.manifest["artifact_id"] != artifact_id:
                raise InferenceFailure(404, "model_not_loaded", "Select the active immutable model identifier")
            if self.leases >= self.settings.max_concurrency + self.settings.max_queue:
                raise InferenceFailure(429, "queue_full", "Inference queue capacity exceeded")
            deployment = self.current
            self.leases += 1
        acquired = False
        try:
            try:
                async with asyncio.timeout(self.settings.queue_timeout_seconds):
                    await self._slots.acquire()
                acquired = True
            except TimeoutError as exc:
                raise InferenceFailure(504, "queue_timeout", "Inference queue deadline exceeded") from exc
            self.running += 1
            yield Lease(deployment, queued_at, time.perf_counter())
        finally:
            if acquired:
                self.running -= 1
                self._slots.release()
            async with self._condition:
                self.leases -= 1
                self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self.closing = True
        async with self._switch_lock:
            async with self._condition:
                await self._condition.wait_for(lambda: self.leases == 0)
            if self.current:
                await self.current.backend.unload()
                self.current = None


def create_app(settings: Settings | None = None, registry: Registry | None = None,
               backend_factory: Callable[[dict], Backend] | None = None) -> FastAPI:
    settings = settings or Settings()
    registry = registry or Registry(settings.registry_path)
    manager = ModelManager(registry, settings, backend_factory or default_backend_factory)
    metrics = CollectorRegistry()
    requests = Counter("llm_requests_total", "Completed inference requests", ["status"], registry=metrics)
    http_requests = Counter("llm_http_requests_total", "HTTP responses including validation and authorization errors",
                            ["route", "status"], registry=metrics)
    duration = Histogram("llm_request_seconds", "Gateway end to end request duration", registry=metrics)
    queue_delay = Histogram("llm_queue_seconds", "Time spent waiting for execution capacity", registry=metrics)
    execution = Histogram("llm_execution_seconds", "Backend stream lifetime including downstream transport", registry=metrics)
    first_token = Histogram("llm_ttft_seconds", "Request to first nonempty generated text", registry=metrics)
    output_tokens = Counter("llm_output_tokens_total", "Generated output tokens", registry=metrics)
    active = Gauge("llm_active_requests", "Active backend requests", registry=metrics)
    queued = Gauge("llm_queued_requests", "Requests waiting for an execution slot", registry=metrics)
    model_ready = Gauge("llm_model_ready", "Model runtime health", registry=metrics)
    gpu_memory = Gauge("llm_gpu_memory_bytes", "Runtime-reported GPU allocated bytes", registry=metrics)
    deployment_info = Gauge("llm_deployment_info", "Current immutable model deployment identity",
                            ["artifact_id", "base_revision", "backend"], registry=metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await manager.start()
        except Exception as exc:
            # Keep health and admin available so a failed initial package can be repaired.
            manager.last_load_error = type(exc).__name__
        yield
        await manager.close()

    app = FastAPI(title="LLM Adaptation Gateway", version="0.1.0", lifespan=lifespan)
    app.state.manager, app.state.registry, app.state.settings = manager, registry, settings

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request.state.request_id = "req-" + uuid.uuid4().hex
        if request.url.path == "/v1/chat/completions":
            # Bound input before Pydantic parses it, including chunked request bodies.
            length = 0
            chunks = []
            async for chunk in request.stream():
                length += len(chunk)
                if length > settings.max_body_bytes:
                    http_requests.labels("/v1/chat/completions", "413").inc()
                    return JSONResponse({"error": {"code": "body_too_large", "message": "Request body exceeds limit"}}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        # Use declared routes to avoid unbounded metric cardinality from arbitrary URLs.
        route = request.scope.get("route")
        http_requests.labels(getattr(route, "path", "unmatched"), str(response.status_code)).inc()
        return response

    @app.exception_handler(InferenceFailure)
    async def failure_handler(request: Request, exc: InferenceFailure):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message,
                                       "request_id": request.state.request_id}}, status_code=exc.status)

    def check_key(request: Request, expected: str | None, administrative: bool = False) -> None:
        if expected is None:
            if administrative:
                raise HTTPException(status_code=403, detail="Administrative access is disabled")
            try:
                local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
            except ValueError:
                local = False
            if not local:
                raise HTTPException(status_code=403, detail="Unauthenticated inference is restricted to loopback clients")
            return
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + expected):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    async def authorize(request: Request):
        check_key(request, settings.api_key)

    async def admin(request: Request):
        check_key(request, settings.admin_key, administrative=True)

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        healthy = bool(manager.current and manager.current.backend.health().get("ready") and
                       not manager.switching and not manager.closing)
        return JSONResponse({"status": "ready" if healthy else "not_ready",
                             "model": manager.current.manifest["artifact_id"] if manager.current else None,
                             "switching": manager.switching, "last_load_error": manager.last_load_error},
                            status_code=200 if healthy else 503)

    @app.get("/metrics", dependencies=[Depends(authorize)])
    async def metric_endpoint():
        health = manager.current.backend.health() if manager.current else {}
        active.set(manager.running)
        queued.set(manager.leases - manager.running)
        model_ready.set(int(bool(health.get("ready"))))
        gpu_memory.set(float(health.get("gpu_memory_bytes", health.get("cuda_allocated_bytes", 0)) or 0))
        deployment_info.clear()
        if manager.current:
            manifest = manager.current.manifest
            deployment_info.labels(manifest["artifact_id"], manifest["base_model"]["revision"],
                                   manifest["runtime"]["backend"]).set(1)
        return Response(generate_latest(metrics), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        manifests = await asyncio.to_thread(registry.list)
        return {"object": "list", "data": [
            {"id": item["artifact_id"], "object": "model", "owned_by": "local",
             "base_model": item["base_model"], "loaded": bool(manager.current and
                manager.current.manifest["artifact_id"] == item["artifact_id"])} for item in manifests]}

    @app.get("/admin/status", dependencies=[Depends(admin)])
    async def admin_status():
        return {"deployment": registry.deployment(), "switching": manager.switching,
                "last_load_error": manager.last_load_error,
                "engine": manager.current.backend.health() if manager.current else {"ready": False}}

    async def perform_switch(artifact_id: str):
        try:
            return await manager.switch(artifact_id)
        except InferenceFailure:
            raise
        except ArtifactError as exc:
            raise InferenceFailure(409, "invalid_artifact", str(exc)) from exc
        except TimeoutError as exc:
            raise InferenceFailure(504, "load_timeout", "Model load deadline exceeded; prior deployment retained") from exc
        except Exception as exc:
            raise InferenceFailure(503, "load_failed", "Model load failed; prior deployment retained") from exc

    @app.post("/admin/promote", dependencies=[Depends(admin)])
    async def promote(body: PromotionRequest):
        return await perform_switch(body.artifact_id)

    @app.post("/admin/rollback", dependencies=[Depends(admin)])
    async def rollback():
        previous = registry.deployment()["previous"]
        if previous is None:
            raise InferenceFailure(409, "no_previous_deployment", "No prior verified deployment exists")
        return await perform_switch(previous)

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat(body: ChatRequest, request: Request):
        started = time.perf_counter()
        request_id = request.state.request_id
        if body.max_tokens > settings.max_output_tokens:
            raise InferenceFailure(422, "output_limit", "Requested generation exceeds configured output limit")
        stops = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or [])
        if len(stops) > 4 or any(not isinstance(x, str) or not x or len(x) > 256 for x in stops):
            raise InferenceFailure(422, "invalid_stop", "Provide up to four nonempty stop strings of at most 256 characters")
        if body.stream_options and (set(body.stream_options) - {"include_usage"}):
            raise InferenceFailure(422, "unsupported_stream_option", "Only include_usage is supported")
        messages = [message.model_dump() for message in body.messages]
        config = GenerationConfig(body.max_tokens, body.temperature, body.seed, stops)
        lease_context = manager.acquire(body.model)
        try:
            lease = await lease_context.__aenter__()
        except InferenceFailure as exc:
            requests.labels(exc.code).inc()
            raise
        backend = lease.deployment.backend
        try:
            input_tokens = await asyncio.to_thread(backend.count_tokens, messages)
            if input_tokens > settings.max_input_tokens:
                raise InferenceFailure(422, "input_limit", "Prompt exceeds configured input token limit")
            context_limit = backend.health().get("max_model_len")
            if context_limit is not None and input_tokens + body.max_tokens > context_limit:
                raise InferenceFailure(422, "context_limit", "Prompt and requested generation exceed the model context limit")
        except InferenceFailure:
            await lease_context.__aexit__(None, None, None)
            raise
        except BaseException as exc:
            await lease_context.__aexit__(None, None, None)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise InferenceFailure(503, "tokenizer_error", "Tokenization failed") from exc
        manifest = lease.deployment.manifest
        model = manifest["artifact_id"]
        created = int(time.time())
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        metadata = {"artifact_id": model, "base_model_revision": manifest["base_model"]["revision"],
                    "tokenizer_revision": manifest["tokenizer"]["revision"],
                    "backend": manifest["runtime"]["backend"], "request_id": request_id}
        queue_delay.observe(lease.admitted_at - lease.queued_at)
        released = False

        async def release(status: str):
            nonlocal released
            if released:
                return
            released = True
            try:
                await backend.cancel(request_id)
            finally:
                await lease_context.__aexit__(None, None, None)
                execution.observe(time.perf_counter() - lease.admitted_at)
                duration.observe(time.perf_counter() - started)
                requests.labels(status).inc()

        async def generate():
            status = "ok"
            observed_first = False
            token_total = 0
            try:
                async with asyncio.timeout(settings.request_timeout_seconds):
                    iterator = backend.stream(messages, config, request_id).__aiter__()
                    try:
                        while True:
                            # Stream disconnects cancel this generator. For synchronous
                            # callers poll the transport while the backend is waiting.
                            next_chunk = asyncio.create_task(anext(iterator))
                            try:
                                while not next_chunk.done():
                                    done, _ = await asyncio.wait({next_chunk}, timeout=0.1)
                                    if done:
                                        break
                                    if await request.is_disconnected():
                                        raise InferenceFailure(499, "client_disconnected", "Client disconnected")
                                try:
                                    chunk = next_chunk.result()
                                except StopAsyncIteration:
                                    break
                            finally:
                                if not next_chunk.done():
                                    next_chunk.cancel()
                                    try:
                                        await next_chunk
                                    except (asyncio.CancelledError, StopAsyncIteration):
                                        pass
                            if chunk.token_count < 0:
                                raise RuntimeError("Backend returned invalid token count")
                            token_total += chunk.token_count
                            if token_total > body.max_tokens:
                                raise RuntimeError("Backend exceeded generation token limit")
                            if chunk.text and not observed_first:
                                first_token.observe(time.perf_counter() - started)
                                observed_first = True
                            output_tokens.inc(chunk.token_count)
                            yield chunk
                    finally:
                        close = getattr(iterator, "aclose", None)
                        if close:
                            await close()
            except TimeoutError as exc:
                status = "generation_timeout"
                raise InferenceFailure(504, status, "Generation deadline exceeded") from exc
            except InferenceFailure as exc:
                status = exc.code
                raise
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                status = "engine_error"
                raise InferenceFailure(503, status, "Model execution failed") from exc
            finally:
                await release(status)

        if not body.stream:
            content, total, finish = [], 0, "stop"
            async for chunk in generate():
                content.append(chunk.text)
                total += chunk.token_count
                if chunk.finish_reason:
                    finish = chunk.finish_reason
            return {"id": completion_id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(content)},
                                 "finish_reason": finish}],
                    "usage": {"prompt_tokens": input_tokens, "completion_tokens": total,
                              "total_tokens": input_tokens + total}, "system_fingerprint": model,
                    "metadata": metadata}

        def event(delta: dict, finish: str | None = None, usage: dict | None = None):
            value = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                     "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                     "system_fingerprint": model, "metadata": metadata}
            if usage is not None:
                value["usage"] = usage
                value["choices"] = []
            return "data: " + json.dumps(value) + "\n\n"

        async def events():
            # Keep ownership in this generator, including cancellation before its
            # first model chunk. The initial role event carries the pinned identity.
            generator = generate()
            total, finish = 0, "stop"
            try:
                yield event({"role": "assistant"})
                async for chunk in generator:
                    total += chunk.token_count
                    if chunk.text:
                        yield event({"content": chunk.text})
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
                yield event({}, finish)
                if body.stream_options and body.stream_options.get("include_usage"):
                    yield event({}, usage={"prompt_tokens": input_tokens, "completion_tokens": total,
                                           "total_tokens": input_tokens + total})
                yield "data: [DONE]\n\n"
            except InferenceFailure as exc:
                yield "data: " + json.dumps({"error": {"code": exc.code, "message": exc.message,
                                                        "request_id": request_id}}) + "\n\n"
            finally:
                await generator.aclose()
                await release("cancelled")

        class ManagedStreamingResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    # Also handles disconnects before the events generator starts.
                    await release("cancelled")

        return ManagedStreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                          "X-Model-Revision": model})

    return app
