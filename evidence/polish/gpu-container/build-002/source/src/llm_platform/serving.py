"""Bounded inference gateway with pinned deployment leases and safe model switching."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import ipaddress
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Literal
import uuid

import anyio

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import ClientDisconnect
from starlette.routing import Match

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
    body_timeout_seconds: float = 15.0
    queue_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 120.0
    switch_timeout_seconds: float = 120.0
    cleanup_timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "localhost", "::1"} and (not self.api_key or not self.admin_key):
            raise ValueError("Non-local serving requires separate API and administrative keys")
        if self.api_key and self.admin_key and self.api_key == self.admin_key:
            raise ValueError("Administrative and inference keys must differ")
        for value in (self.max_input_tokens, self.max_output_tokens, self.max_body_bytes, self.max_concurrency):
            if value < 1:
                raise ValueError("Limits must be positive")
        if self.max_queue < 0 or any(not math.isfinite(value) or value <= 0 for value in (
                self.body_timeout_seconds, self.queue_timeout_seconds, self.request_timeout_seconds,
                self.switch_timeout_seconds, self.cleanup_timeout_seconds)):
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
        self.last_switch_status = "idle"
        self.commit_uncertain = False
        self.commit_warning: str | None = None
        self.shutdown_warning: str | None = None
        # Retain ownership even if a backend cannot finish unloading by the
        # response deadline. Cancelling an asyncio task does not stop a thread or
        # CUDA kernel; these tasks are never cancelled merely to meet a timeout.
        self._cleanup: dict[int, dict[str, Any]] = {}
        self.leases = 0
        self.running = 0
        self._condition = asyncio.Condition()
        self._switch_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(settings.max_concurrency)

    async def start(self) -> None:
        artifact_id = self.settings.initial_model or self.registry.deployment()["current"]
        if artifact_id:
            await self.switch(artifact_id)

    async def switch(self, artifact_id: str | None, *, _rollback: bool = False) -> dict[str, Any]:
        """Stage with the old deployment resident; commit only after a successful load.

        This trades temporary double residency for failure isolation. A small GPU
        can safely refuse a switch instead of sacrificing its healthy deployment.
        """
        async with self._switch_lock:
            if self.closing:
                raise InferenceFailure(503, "shutting_down", "Gateway is shutting down")
            if self.commit_uncertain:
                raise InferenceFailure(503, "deployment_state_uncertain", "Deployment state requires operator reconciliation")
            if _rollback:
                artifact_id = (await asyncio.to_thread(self.registry.deployment))["previous"]
                if artifact_id is None:
                    raise InferenceFailure(409, "no_previous_deployment", "No prior verified deployment exists")
            package, manifest = await asyncio.to_thread(self.registry.verify, artifact_id)
            if self.current and self.current.manifest["artifact_id"] == artifact_id:
                return self._switch_result(self.registry.deployment())
            async with self._condition:
                if self.closing:
                    raise InferenceFailure(503, "shutting_down", "Gateway is shutting down")
                if self.commit_uncertain:
                    raise InferenceFailure(503, "deployment_state_uncertain", "Deployment state requires operator reconciliation")
                if self.cleanup_status():
                    raise InferenceFailure(503, "cleanup_pending", "Previous runtime cleanup must finish before switching")
                self.switching = True
                self.last_switch_status = "staging"
                self.commit_warning = None
            candidate = None
            committed = False
            try:
                # Only reversible staging is subject to the load deadline.
                # The durable commit and old-runtime cleanup are separate stages.
                async with asyncio.timeout(self.settings.switch_timeout_seconds):
                    async with self._condition:
                        await self._condition.wait_for(lambda: self.leases == 0)
                    candidate = self.factory(manifest)
                    await candidate.load(package, manifest)
                    if candidate.health().get("ready") is not True:
                        raise ArtifactError("Candidate backend did not become ready")

                old = self.current
                deployment = Deployment(candidate, package, manifest)

                async def commit_and_adopt():
                    self.last_switch_status = "committing"
                    try:
                        state = await asyncio.to_thread(self.registry.promote, artifact_id)
                    except Exception as exc:
                        # A filesystem exception after atomic replacement (for
                        # example while removing a lock) does not mean no commit
                        # happened. Reconcile the durable pointer before deciding.
                        try:
                            observed = await asyncio.to_thread(self.registry.deployment)
                        except Exception:
                            self.commit_uncertain = True
                            self.last_switch_status = "commit_uncertain"
                            raise InferenceFailure(503, "deployment_state_uncertain",
                                                   "Cannot reconcile durable deployment state; operator recovery required") from exc
                        if observed["current"] != artifact_id:
                            raise
                        state = observed
                        self.commit_warning = "Durable commit succeeded but registry completion reported an error"
                    # No await separates completed durable commit from adoption.
                    self.current = deployment
                    self.last_load_error = None
                    self.last_switch_status = "committed"
                    return state

                commit_task = asyncio.create_task(commit_and_adopt(), name=f"deployment-commit-{artifact_id[:12]}")
                interrupted = False
                # A cancelled await cannot stop registry.promote's thread. Wait
                # for this owned transaction to settle before freeing a candidate
                # or reporting cancellation to the administrative caller.
                while not commit_task.done():
                    try:
                        await asyncio.shield(commit_task)
                    except asyncio.CancelledError:
                        interrupted = True
                state = commit_task.result()
                committed = True
                candidate = None
                cleanup_task = self._start_cleanup(old, "previous_deployment") if old else None
                if interrupted:
                    # Cleanup is already owned by the manager when cancellation
                    # leaves the request; selection is durably committed.
                    raise asyncio.CancelledError
                if cleanup_task:
                    await asyncio.wait({cleanup_task}, timeout=self.settings.cleanup_timeout_seconds)
                return self._switch_result(state)
            except BaseException as exc:
                if not committed:
                    self.last_load_error = type(exc).__name__
                    if not self.commit_uncertain:
                        self.last_switch_status = "failed_before_commit"
                raise
            finally:
                try:
                    if candidate:
                        task = self._start_cleanup(Deployment(candidate, package, manifest), "failed_candidate")
                        await asyncio.wait({task}, timeout=self.settings.cleanup_timeout_seconds)
                finally:
                    async with self._condition:
                        self.switching = False
                        self._condition.notify_all()

    async def rollback(self) -> dict[str, Any]:
        """Choose the previous identity inside the same transaction as switching."""
        return await self.switch(None, _rollback=True)

    def cleanup_status(self) -> list[dict[str, Any]]:
        return [{"artifact_id": record["deployment"].manifest["artifact_id"],
                 "stage": record["stage"], "status": "failed" if record["error"] else "pending",
                 "error": record["error"]} for record in self._cleanup.values()]

    def _switch_result(self, state: dict[str, Any]) -> dict[str, Any]:
        cleanup = self.cleanup_status()
        return {**state, "committed": True, "cleanup_pending": bool(cleanup), "cleanup": cleanup,
                "warning": self.commit_warning or (
                    "Deployment committed; runtime cleanup unresolved and inference unavailable until cleanup completes"
                    if cleanup else None)}

    def _start_cleanup(self, deployment: Deployment | None, stage: str) -> asyncio.Task:
        if deployment is None:
            raise ValueError("Cleanup requires an owned deployment")
        key = id(deployment.backend)
        if key in self._cleanup:
            return self._cleanup[key]["task"]
        task = asyncio.create_task(deployment.backend.unload(), name=f"runtime-cleanup-{stage}")
        record = {"deployment": deployment, "stage": stage, "task": task, "error": None}
        self._cleanup[key] = record

        def completed(done: asyncio.Task):
            # Retrieve exceptions to avoid unobserved task failures. On failure
            # retain the backend and its explicit recovery state for the operator.
            if done.cancelled():
                record["error"] = "CancelledError"
            elif done.exception() is not None:
                record["error"] = type(done.exception()).__name__
            else:
                self._cleanup.pop(key, None)
                if self.closing and self.current and self.current.backend is deployment.backend:
                    self.current = None

        task.add_done_callback(completed)
        return task

    @asynccontextmanager
    async def acquire(self, artifact_id: str):
        queued_at = time.perf_counter()
        async with self._condition:
            if self.closing or self.switching or self.current is None or self.commit_uncertain or self.cleanup_status():
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

    async def close(self) -> dict[str, Any]:
        async with self._condition:
            self.closing = True
        acquired = False
        try:
            async with asyncio.timeout(self.settings.cleanup_timeout_seconds):
                await self._switch_lock.acquire()
                acquired = True
                async with self._condition:
                    await self._condition.wait_for(lambda: self.leases == 0)
            if self.current:
                self._start_cleanup(self.current, "shutdown")
            tasks = {record["task"] for record in self._cleanup.values() if not record["task"].done()}
            if tasks:
                await asyncio.wait(tasks, timeout=self.settings.cleanup_timeout_seconds)
        except TimeoutError:
            self.shutdown_warning = "Shutdown drain deadline exceeded; runtime ownership retained for operator recovery"
        finally:
            if acquired:
                self._switch_lock.release()
        cleanup = self.cleanup_status()
        incomplete = bool(cleanup or self.leases or self._switch_lock.locked() or self.current)
        if incomplete and self.shutdown_warning is None:
            self.shutdown_warning = "Runtime cleanup did not finish; process recovery may be required"
        if not incomplete:
            self.shutdown_warning = None
        return {"status": "incomplete" if incomplete else "complete", "cleanup": cleanup,
                "warning": self.shutdown_warning}


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
    gpu_memory = Gauge("llm_gpu_memory_bytes", "Runtime-reported GPU allocated bytes; NaN when unavailable", registry=metrics)
    gpu_peak = Gauge("llm_gpu_peak_allocated_bytes", "Runtime session peak allocated bytes; NaN when unavailable", registry=metrics)
    gpu_reserved = Gauge("llm_gpu_reserved_bytes", "Runtime allocator reserved bytes; NaN when unavailable", registry=metrics)
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

    def metric_route(request: Request) -> str:
        # Middleware runs before route resolution. Match only declared routes so
        # rejected uploads cannot create arbitrary metric-label cardinality.
        for route in app.router.routes:
            if route.matches(request.scope)[0] == Match.FULL:
                return getattr(route, "path", "unmatched")
        return "unmatched"

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request.state.request_id = "req-" + uuid.uuid4().hex
        if request.method in {"POST", "PUT", "PATCH"}:
            # Bound all write bodies before parsing or dependency resolution,
            # including unauthenticated administration and chunked uploads.
            length = 0
            chunks = []

            def upload_error(status: int, code: str, message: str):
                http_requests.labels(metric_route(request), str(status)).inc()
                return JSONResponse({"error": {"code": code, "message": message,
                                               "request_id": request.state.request_id}}, status_code=status,
                                    headers={"X-Request-ID": request.state.request_id})

            try:
                async with asyncio.timeout(settings.body_timeout_seconds):
                    async for chunk in request.stream():
                        length += len(chunk)
                        if length > settings.max_body_bytes:
                            return upload_error(413, "body_too_large", "Request body exceeds limit")
                        chunks.append(chunk)
            except TimeoutError:
                return upload_error(408, "body_timeout", "Request body upload deadline exceeded")
            except ClientDisconnect:
                return upload_error(400, "client_disconnected", "Client disconnected before completing request body")
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

    @app.exception_handler(ArtifactError)
    async def unavailable_registry(request: Request, exc: ArtifactError):
        return JSONResponse({"error": {"code": "registry_unavailable", "message": "Registry state could not be verified",
                                       "request_id": request.state.request_id}}, status_code=503)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Pydantic's default response contains raw `input` and validator context.
        # Unknown field names can themselves contain incident data, so locations
        # also use only known schema names and bounded numeric list indexes.
        allowed_locations = {"body", "query", "path", "header", *ChatRequest.model_fields,
                             *ChatMessage.model_fields, *PromotionRequest.model_fields}
        details = []
        for error in exc.errors():
            location = [part if (isinstance(part, str) and part in allowed_locations) or
                         (type(part) is int and 0 <= part < 100_000) else "<field>"
                         for part in error.get("loc", ())]
            kind = error.get("type", "invalid_value")
            if not isinstance(kind, str) or not kind.isascii() or not kind.replace("_", "").isalnum() or len(kind) > 80:
                kind = "invalid_value"
            message = {"missing": "Required field missing", "extra_forbidden": "Unsupported field",
                       "json_invalid": "Malformed JSON"}.get(kind, "Invalid field value")
            details.append({"loc": location, "type": kind, "message": message})
        return JSONResponse({"error": {"code": "invalid_request", "message": "Request validation failed",
                                       "request_id": request.state.request_id, "details": details}}, status_code=422)

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
        if not hmac.compare_digest(supplied.encode("utf-8"), ("Bearer " + expected).encode("utf-8")):
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
                       not manager.switching and not manager.closing and not manager.cleanup_status()
                       and not manager.commit_uncertain)
        return JSONResponse({"status": "ready" if healthy else "not_ready",
                             "model": manager.current.manifest["artifact_id"] if manager.current else None,
                             "switching": manager.switching, "last_load_error": manager.last_load_error,
                             "switch_status": manager.last_switch_status,
                             "cleanup": manager.cleanup_status(), "commit_uncertain": manager.commit_uncertain},
                            status_code=200 if healthy else 503)

    @app.get("/metrics", dependencies=[Depends(authorize)])
    async def metric_endpoint():
        health = manager.current.backend.health() if manager.current else {}
        active.set(manager.running)
        queued.set(manager.leases - manager.running)
        model_ready.set(int(bool(health.get("ready"))))
        def observed_memory(*names: str) -> float:
            for name in names:
                value = health.get(name)
                if type(value) not in {int, float}:
                    continue
                try:
                    measured = float(value)
                except OverflowError:
                    continue
                if math.isfinite(measured) and measured >= 0:
                    return measured
            # Zero is a measurement, not a fallback for unsupported telemetry.
            return float("nan")

        gpu_memory.set(observed_memory("gpu_memory_bytes", "cuda_allocated_bytes"))
        gpu_peak.set(observed_memory("gpu_peak_allocated_bytes", "cuda_peak_allocated_bytes"))
        gpu_reserved.set(observed_memory("gpu_reserved_bytes", "cuda_reserved_bytes"))
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
        try:
            durable = registry.deployment()
        except ArtifactError:
            durable = {"error": "durable_state_unreadable"}
        return {"deployment": durable, "switching": manager.switching,
                "last_load_error": manager.last_load_error,
                "active_model": manager.current.manifest["artifact_id"] if manager.current else None,
                "switch_status": manager.last_switch_status, "cleanup": manager.cleanup_status(),
                "commit_uncertain": manager.commit_uncertain, "commit_warning": manager.commit_warning,
                "shutdown_warning": manager.shutdown_warning,
                "engine": manager.current.backend.health() if manager.current else {"ready": False}}

    async def perform_switch(artifact_id: str | None = None, *, rollback: bool = False):
        try:
            return await manager.rollback() if rollback else await manager.switch(artifact_id)
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
        return await perform_switch(rollback=True)

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
            terminal = False
            deadline = asyncio.timeout(settings.request_timeout_seconds)
            try:
                async with deadline:
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
                            if terminal:
                                raise RuntimeError("Backend emitted a chunk after its terminal finish")
                            if type(chunk.token_count) is not int or chunk.token_count < 0:
                                raise RuntimeError("Backend returned invalid token count")
                            if not isinstance(chunk.text, str):
                                raise RuntimeError("Backend returned non-text content")
                            if chunk.finish_reason is not None:
                                if chunk.finish_reason not in {"stop", "length"}:
                                    raise RuntimeError("Backend returned unsupported finish reason")
                                terminal = True
                            token_total += chunk.token_count
                            if token_total > body.max_tokens:
                                raise RuntimeError("Backend exceeded generation token limit")
                            if chunk.text and not observed_first:
                                first_token.observe(time.perf_counter() - started)
                                observed_first = True
                            output_tokens.inc(chunk.token_count)
                            yield chunk
                        if not terminal:
                            raise RuntimeError("Backend stream omitted a terminal finish")
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
            except GeneratorExit:
                status = "generation_timeout" if deadline.expired() else "cancelled"
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
                    async def close_owned_stream():
                        try:
                            await self.body_iterator.aclose()
                        finally:
                            # Also handles disconnects before events() starts.
                            await release("cancelled")

                    cleanup = asyncio.create_task(close_owned_stream(), name=f"stream-cleanup-{request_id}")
                    interrupted = False
                    # Socket errors/cancellation can occur during transport send,
                    # while both event and backend generators are suspended at a
                    # yield. Own and close the whole chain before releasing its
                    # deployment lease. Shield AnyIO's level cancellation as well
                    # as explicit asyncio task cancellation until ownership settles.
                    with anyio.CancelScope(shield=True):
                        while not cleanup.done():
                            try:
                                await asyncio.shield(cleanup)
                            except asyncio.CancelledError:
                                interrupted = True
                        cleanup.result()
                    if interrupted:
                        raise asyncio.CancelledError

        return ManagedStreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                          "X-Model-Revision": model})

    return app
