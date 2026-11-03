import asyncio
from contextlib import asynccontextmanager
import json
import threading
import time

import httpx
import pytest

from llm_platform.contracts import TokenChunk
from llm_platform.registry import Registry
from llm_platform.serving import InferenceFailure, ModelManager, Settings, create_app
from test_registry import make_package


class FixtureBackend:
    """CPU-only fixture. It must never be selected by the production factory."""

    def __init__(self, *, delay=0.0, fail_load=False, fail_generation=False):
        self.delay, self.fail_load, self.fail_generation = delay, fail_load, fail_generation
        self.ready = False
        self.cancelled = []
        self.loaded = None
        self.unloaded = False
        self.running = 0
        self.peak = 0

    async def load(self, package, manifest):
        if self.fail_load:
            raise RuntimeError("fixture load failure")
        self.loaded = manifest["artifact_id"]
        self.ready = True

    async def unload(self):
        self.ready = False
        self.unloaded = True

    def health(self):
        return {"ready": self.ready, "gpu_memory_bytes": 0}

    def count_tokens(self, messages):
        return sum(len(message["content"].split()) for message in messages)

    async def stream(self, messages, config, request_id):
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if self.fail_generation:
                raise RuntimeError("fixture execution failure")
            await asyncio.sleep(self.delay)
            yield TokenChunk("hello", 1)
            await asyncio.sleep(self.delay)
            yield TokenChunk(" world", 1)
            yield TokenChunk("", 0, "stop")
        finally:
            self.running -= 1

    async def cancel(self, request_id):
        self.cancelled.append(request_id)


@asynccontextmanager
async def client_for(tmp_path, backend=None, **overrides):
    package, manifest = make_package(tmp_path)
    registry = Registry(tmp_path / "registry")
    registry.register(package)
    settings = Settings(registry_path=registry.root, initial_model=manifest["artifact_id"], **overrides)
    fixture = backend or FixtureBackend()
    app = create_app(settings, registry, lambda metadata: fixture)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            yield client, app, fixture, manifest


def body(manifest, **overrides):
    return {"model": manifest["artifact_id"], "messages": [{"role": "user", "content": "test incident"}],
            "max_tokens": 8, **overrides}


async def test_synchronous_streaming_identity_health_and_metrics(tmp_path):
    async with client_for(tmp_path) as (client, app, backend, manifest):
        assert (await client.get("/health/ready")).status_code == 200
        result = await client.post("/v1/chat/completions", json=body(manifest))
        assert result.status_code == 200
        result_json = result.json()
        assert result_json["choices"][0]["message"]["content"] == "hello world"
        assert result_json["usage"]["completion_tokens"] == 2
        assert result_json["metadata"]["artifact_id"] == manifest["artifact_id"]
        assert result.headers["x-request-id"] == result_json["metadata"]["request_id"]
        response = await client.post("/v1/chat/completions", json=body(manifest, stream=True, stream_options={"include_usage": True}))
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        packets = [json.loads(event) for event in events[:-1]]
        assert packets[-1]["usage"]["completion_tokens"] == 2
        assert all(packet["model"] == manifest["artifact_id"] for packet in packets)
        assert app.state.manager.leases == 0
        assert len(backend.cancelled) == 2
        metrics = (await client.get("/metrics")).text
        assert 'llm_requests_total{status="ok"} 2.0' in metrics
        assert f'llm_deployment_info{{artifact_id="{manifest["artifact_id"]}"' in metrics
        models = (await client.get("/v1/models")).json()["data"]
        assert models[0]["loaded"] is True


async def test_auth_limits_and_admin_protection(tmp_path):
    async with client_for(tmp_path, api_key="inference-fixture", admin_key="admin-fixture", max_input_tokens=2, max_output_tokens=8) as (client, app, backend, manifest):
        assert (await client.get("/v1/models")).status_code == 401
        client.headers["Authorization"] = "Bearer inference-fixture"
        assert (await client.get("/v1/models")).status_code == 200
        assert (await client.get("/admin/status")).status_code == 401
        assert (await client.post("/v1/chat/completions", json=body(manifest, max_tokens=9))).status_code == 422
        too_long = body(manifest, messages=[{"role": "user", "content": "one two three"}])
        assert (await client.post("/v1/chat/completions", json=too_long)).json()["error"]["code"] == "input_limit"
        assert app.state.manager.leases == 0
        bad_type = body(manifest, max_tokens="8")
        assert (await client.post("/v1/chat/completions", json=bad_type)).status_code == 422
        assert (await client.post("/v1/chat/completions", json=body(manifest, model="wrong"))).status_code == 404
        assert (await client.post("/v1/chat/completions", json=body(manifest, stop=[""]))).status_code == 422


def test_nonlocal_requires_different_keys():
    with pytest.raises(ValueError, match="Non-local"):
        Settings(host="0.0.0.0")
    with pytest.raises(ValueError, match="differ"):
        Settings(host="0.0.0.0", api_key="same-fixture", admin_key="same-fixture")
    settings = Settings(host="0.0.0.0", api_key="api-fixture", admin_key="admin-fixture")
    assert "api-fixture" not in repr(settings)


async def test_bounded_queue_timeout_and_recovery(tmp_path):
    fixture = FixtureBackend(delay=0.1)
    async with client_for(tmp_path, fixture, max_queue=0, max_concurrency=1) as (client, app, backend, manifest):
        task = asyncio.create_task(client.post("/v1/chat/completions", json=body(manifest)))
        await asyncio.sleep(0.04)
        overloaded = await client.post("/v1/chat/completions", json=body(manifest))
        assert overloaded.status_code == 429
        assert (await task).status_code == 200
        assert (await client.post("/v1/chat/completions", json=body(manifest))).status_code == 200
        assert backend.peak == 1
        assert app.state.manager.leases == 0


async def test_queue_deadline_and_generation_deadline_release_capacity(tmp_path):
    fixture = FixtureBackend(delay=0.1)
    async with client_for(tmp_path, fixture, max_queue=1, max_concurrency=1, queue_timeout_seconds=0.02) as (client, app, backend, manifest):
        first = asyncio.create_task(client.post("/v1/chat/completions", json=body(manifest)))
        await asyncio.sleep(0.01)
        second = await client.post("/v1/chat/completions", json=body(manifest))
        assert second.status_code == 504
        assert second.json()["error"]["code"] == "queue_timeout"
        assert (await first).status_code == 200
        assert app.state.manager.leases == 0
    other = tmp_path / "other"
    other.mkdir()
    async with client_for(other, FixtureBackend(delay=0.1), request_timeout_seconds=0.02) as (client, app, backend, manifest):
        response = await client.post("/v1/chat/completions", json=body(manifest))
        assert response.status_code == 504
        assert backend.cancelled
        assert app.state.manager.leases == backend.running == 0
        backend.delay = 0
        assert (await client.post("/v1/chat/completions", json=body(manifest))).status_code == 200


async def test_stream_error_is_typed_and_not_generated_text(tmp_path):
    async with client_for(tmp_path, FixtureBackend(fail_generation=True)) as (client, app, backend, manifest):
        response = await client.post("/v1/chat/completions", json=body(manifest, stream=True))
        assert '"engine_error"' in response.text
        assert "[DONE]" not in response.text
        assert app.state.manager.leases == 0
        response = await client.post("/v1/chat/completions", json=body(manifest))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "engine_error"


async def test_failed_load_preserves_deployment_then_successful_rollback(tmp_path):
    one_package, one = make_package(tmp_path)
    two_package, two = make_package(tmp_path, "two")
    registry = Registry(tmp_path / "registry")
    registry.register(one_package)
    registry.register(two_package)
    instances = []
    fail = True

    def factory(manifest):
        instance = FixtureBackend(fail_load=(fail and manifest["artifact_id"] == two["artifact_id"]))
        instances.append(instance)
        return instance

    app = create_app(Settings(initial_model=one["artifact_id"], admin_key="admin-fixture"), registry, factory)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.post("/admin/promote", json={"artifact_id": two["artifact_id"]}, headers={"Authorization": "Bearer admin-fixture"})
            assert response.status_code == 503
            assert registry.deployment()["current"] == one["artifact_id"]
            assert instances[0].ready
            assert instances[1].unloaded
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.post("/v1/chat/completions", json=body(one))).status_code == 200
            fail = False
            response = await client.post("/admin/promote", json={"artifact_id": two["artifact_id"]}, headers={"Authorization": "Bearer admin-fixture"})
            assert response.status_code == 200
            assert instances[0].unloaded
            assert (await client.post("/v1/chat/completions", json=body(one))).status_code == 404
            assert (await client.post("/v1/chat/completions", json=body(two))).status_code == 200
            response = await client.post("/admin/rollback", headers={"Authorization": "Bearer admin-fixture"})
            assert response.status_code == 200
            assert registry.deployment()["current"] == one["artifact_id"]


async def test_switch_waits_for_request_identity_lease(tmp_path):
    async with client_for(tmp_path, FixtureBackend(delay=0.08)) as (client, app, backend, one):
        two_package, two = make_package(tmp_path, "two")
        app.state.registry.register(two_package)
        app.state.manager.factory = lambda manifest: FixtureBackend()
        request = asyncio.create_task(client.post("/v1/chat/completions", json=body(one)))
        await asyncio.sleep(0.02)
        switch = asyncio.create_task(app.state.manager.switch(two["artifact_id"]))
        await asyncio.sleep(0.02)
        assert not switch.done()
        unavailable = await client.post("/v1/chat/completions", json=body(one))
        assert unavailable.status_code == 503
        response = await request
        assert response.json()["model"] == one["artifact_id"]
        await switch
        assert app.state.manager.current.manifest["artifact_id"] == two["artifact_id"]


async def test_bounded_request_body(tmp_path):
    async with client_for(tmp_path, max_body_bytes=300) as (client, app, backend, manifest):
        response = await client.post("/v1/chat/completions", json=body(manifest, messages=[{"role": "user", "content": "x" * 400}]))
        assert response.status_code == 413
        assert app.state.manager.leases == 0


async def test_no_initial_model_is_observable_and_admin_disabled(tmp_path):
    app = create_app(Settings(registry_path=tmp_path / "registry"), backend_factory=lambda manifest: FixtureBackend())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/admin/status")).status_code == 403


@pytest.mark.parametrize("stream", [False, True])
async def test_cancelled_client_releases_execution_and_queue(tmp_path, stream):
    async with client_for(tmp_path, FixtureBackend(delay=0.2)) as (client, app, backend, manifest):
        request = asyncio.create_task(client.post("/v1/chat/completions", json=body(manifest, stream=stream)))
        await asyncio.sleep(0.04)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert app.state.manager.leases == 0
        assert backend.running == 0
        assert backend.cancelled
        backend.delay = 0
        assert (await client.post("/v1/chat/completions", json=body(manifest))).status_code == 200


async def test_combined_prompt_and_output_context_limit_is_client_error(tmp_path):
    class ShortContextBackend(FixtureBackend):
        def health(self):
            return {**super().health(), "max_model_len": 8}

    async with client_for(tmp_path, ShortContextBackend(), max_input_tokens=20, max_output_tokens=20) as (client, app, backend, manifest):
        # Each independent gateway bound permits this body, but the combined
        # model-specific context bound does not (2 input + 8 output > 8).
        response = await client.post("/v1/chat/completions", json=body(manifest, max_tokens=8))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_limit"
        assert app.state.manager.leases == 0
        assert backend.cancelled == []
        response = await client.post("/v1/chat/completions", json=body(manifest, max_tokens=6))
        assert response.status_code == 200


@pytest.mark.parametrize("endpoint", ["/v1/models", "/admin/status"])
async def test_non_ascii_bearer_is_unauthorized_not_server_error(tmp_path, endpoint):
    async with client_for(tmp_path, api_key="api-fixture", admin_key="admin-fixture") as (client, app, backend, manifest):
        response = await client.get(endpoint, headers=[(b"authorization", b"Bearer invalid-\xff")])
        assert response.status_code == 401
        assert app.state.registry.deployment()["current"] == manifest["artifact_id"]
        assert app.state.manager.leases == 0


async def test_unauthenticated_admin_body_is_bounded_before_parsing(tmp_path):
    async with client_for(tmp_path, admin_key="admin-fixture", max_body_bytes=128) as (client, app, backend, manifest):
        previous = app.state.registry.deployment()
        response = await client.post("/admin/promote", content=b"x" * 129,
                                     headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "body_too_large"
        assert response.headers["x-request-id"] == response.json()["error"]["request_id"]
        assert app.state.registry.deployment() == previous
        assert app.state.manager.current.manifest["artifact_id"] == manifest["artifact_id"]
        metrics = (await client.get("/metrics")).text
        assert 'llm_http_requests_total{route="/admin/promote",status="413"} 1.0' in metrics
        # An arbitrary unknown URL contributes only the fixed unmatched label.
        response = await client.post("/unknown-attacker-label", content=b"x" * 129)
        assert response.status_code == 413
        metrics = (await client.get("/metrics")).text
        assert "unknown-attacker-label" not in metrics


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/admin/promote"])
async def test_slow_body_upload_times_out_before_admission_or_promotion(tmp_path, endpoint):
    async with client_for(tmp_path, admin_key="admin-fixture", body_timeout_seconds=0.02) as (client, app, backend, manifest):
        previous = app.state.registry.deployment()

        async def slow_body():
            yield b'{"partial":'
            await asyncio.sleep(0.2)
            yield b'"upload"}'

        response = await client.post(endpoint, content=slow_body(), headers={"Content-Type": "application/json"})
        assert response.status_code == 408
        assert response.json()["error"]["code"] == "body_timeout"
        assert app.state.registry.deployment() == previous
        assert app.state.manager.leases == backend.running == 0
        assert backend.cancelled == []


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_upload_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError, match="deadlines"):
        Settings(body_timeout_seconds=timeout)


async def test_cancelled_delayed_pointer_commit_settles_before_candidate_release(tmp_path, monkeypatch):
    one_package, one = make_package(tmp_path)
    two_package, two = make_package(tmp_path, "two")
    registry = Registry(tmp_path / "registry")
    registry.register(one_package)
    registry.register(two_package)
    backends = []

    def factory(manifest):
        backend = FixtureBackend()
        backends.append(backend)
        return backend

    manager = ModelManager(registry, Settings(initial_model=one["artifact_id"],
                                             switch_timeout_seconds=0.02, cleanup_timeout_seconds=0.05), factory)
    await manager.start()
    entered, release = threading.Event(), threading.Event()
    original = registry.promote

    def delayed_promote(artifact_id):
        entered.set()
        if not release.wait(2):
            raise RuntimeError("Test commit release deadline exceeded")
        return original(artifact_id)

    monkeypatch.setattr(registry, "promote", delayed_promote)
    switch = asyncio.create_task(manager.switch(two["artifact_id"]))
    assert await asyncio.to_thread(entered.wait, 1)
    switch.cancel()
    await asyncio.sleep(0.03)  # longer than the reversible-stage deadline
    switch.cancel()
    await asyncio.sleep(0.01)
    assert not switch.done()
    assert registry.deployment()["current"] == manager.current.manifest["artifact_id"] == one["artifact_id"]
    assert not backends[1].unloaded
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await switch
    await asyncio.sleep(0)
    assert registry.deployment()["current"] == manager.current.manifest["artifact_id"] == two["artifact_id"]
    assert not backends[1].unloaded
    assert backends[0].unloaded
    assert manager.last_switch_status == "committed"
    assert (await manager.close())["status"] == "complete"


async def test_postcommit_old_cleanup_timeout_reports_committed_and_recovers(tmp_path):
    class GatedUnload(FixtureBackend):
        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()
            self.cancelled_unload = False

        async def unload(self):
            try:
                await self.release.wait()
                await super().unload()
            except asyncio.CancelledError:
                self.cancelled_unload = True
                raise

    one_package, one = make_package(tmp_path)
    two_package, two = make_package(tmp_path, "two")
    registry = Registry(tmp_path / "registry")
    registry.register(one_package)
    registry.register(two_package)
    old = GatedUnload()
    active = FixtureBackend()
    settings = Settings(initial_model=one["artifact_id"], admin_key="admin-fixture",
                        switch_timeout_seconds=0.02, cleanup_timeout_seconds=0.02)
    app = create_app(settings, registry, lambda manifest: old if manifest["artifact_id"] == one["artifact_id"] else active)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            started = time.perf_counter()
            response = await client.post("/admin/promote", json={"artifact_id": two["artifact_id"]},
                                         headers={"Authorization": "Bearer admin-fixture"})
            assert time.perf_counter() - started < 1
            assert response.status_code == 200
            state = response.json()
            assert state["committed"] is True
            assert state["cleanup_pending"] is True
            assert state["current"] == registry.deployment()["current"] == app.state.manager.current.manifest["artifact_id"] == two["artifact_id"]
            assert "committed" in state["warning"]
            assert "retained" not in state["warning"]
            assert not old.unloaded and not old.cancelled_unload
            assert app.state.manager.last_load_error is None
            assert (await client.get("/health/ready")).status_code == 503
            status = (await client.get("/admin/status", headers={"Authorization": "Bearer admin-fixture"})).json()
            assert status["active_model"] == status["deployment"]["current"] == two["artifact_id"]
            assert status["switch_status"] == "committed"
            assert status["cleanup"][0]["status"] == "pending"
            assert (await client.post("/v1/chat/completions", json=body(two))).status_code == 503
            old.release.set()
            await asyncio.wait_for(next(iter(app.state.manager._cleanup.values()))["task"], 1)
            await asyncio.sleep(0)
            assert old.unloaded and not old.cancelled_unload
            assert app.state.manager.cleanup_status() == []
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.post("/v1/chat/completions", json=body(two))).status_code == 200


async def test_shutdown_cleanup_deadline_retains_backend_task_ownership(tmp_path):
    class StalledUnload(FixtureBackend):
        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()

        async def unload(self):
            await self.release.wait()
            await super().unload()

    package, manifest = make_package(tmp_path)
    registry = Registry(tmp_path / "registry")
    registry.register(package)
    backend = StalledUnload()
    manager = ModelManager(registry, Settings(initial_model=manifest["artifact_id"], cleanup_timeout_seconds=0.02),
                           lambda manifest: backend)
    await manager.start()
    started = time.perf_counter()
    state = await manager.close()
    assert time.perf_counter() - started < 0.5
    assert state["status"] == "incomplete"
    assert state["cleanup"][0]["status"] == "pending"
    record = next(iter(manager._cleanup.values()))
    assert record["deployment"].backend is backend
    assert not record["task"].done()
    assert manager.current.backend is backend
    assert not backend.unloaded
    backend.release.set()
    await asyncio.wait_for(record["task"], 1)
    await asyncio.sleep(0)
    assert (await manager.close())["status"] == "complete"
    assert manager.current is None


async def test_pointer_error_after_atomic_commit_reconciles_active_runtime(tmp_path, monkeypatch):
    one_package, one = make_package(tmp_path)
    two_package, two = make_package(tmp_path, "two")
    registry = Registry(tmp_path / "registry")
    registry.register(one_package)
    registry.register(two_package)
    manager = ModelManager(registry, Settings(initial_model=one["artifact_id"]), lambda manifest: FixtureBackend())
    await manager.start()
    original = registry.promote

    def postcommit_error(artifact_id):
        original(artifact_id)
        raise OSError("fixture failure after atomic replacement")

    monkeypatch.setattr(registry, "promote", postcommit_error)
    state = await manager.switch(two["artifact_id"])
    assert state["committed"] is True
    assert state["warning"] == "Durable commit succeeded but registry completion reported an error"
    assert state["current"] == registry.deployment()["current"] == manager.current.manifest["artifact_id"] == two["artifact_id"]
    assert (await manager.close())["status"] == "complete"


async def test_nested_validation_errors_do_not_echo_incident_content(tmp_path):
    secret_marker = "PRIVATE_INCIDENT_MARKER_do_not_echo"
    async with client_for(tmp_path) as (client, app, backend, manifest):
        malformed = body(manifest, messages=[{"role": secret_marker, "content": {"private": secret_marker},
                                              secret_marker: secret_marker}], **{secret_marker: secret_marker})
        response = await client.post("/v1/chat/completions", json=malformed)
        assert response.status_code == 422
        assert secret_marker not in response.text
        assert '"input"' not in response.text
        assert '"ctx"' not in response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["request_id"] == response.headers["x-request-id"]
        locations = [detail["loc"] for detail in error["details"]]
        assert ["body", "messages", 0, "role"] in locations
        assert ["body", "messages", 0, "content"] in locations
        assert ["body", "<field>"] in locations
        assert app.state.manager.leases == 0
        assert backend.cancelled == []
        assert (await client.post("/v1/chat/completions", json=body(manifest))).status_code == 200


async def test_failed_postcommit_cleanup_retains_explicit_resource_ownership(tmp_path):
    class FailedUnload(FixtureBackend):
        async def unload(self):
            raise RuntimeError("fixture cleanup failure")

    one_package, one = make_package(tmp_path)
    two_package, two = make_package(tmp_path, "two")
    registry = Registry(tmp_path / "registry")
    registry.register(one_package)
    registry.register(two_package)
    old, new = FailedUnload(), FixtureBackend()
    manager = ModelManager(registry, Settings(initial_model=one["artifact_id"], cleanup_timeout_seconds=0.02),
                           lambda manifest: old if manifest["artifact_id"] == one["artifact_id"] else new)
    await manager.start()
    result = await manager.switch(two["artifact_id"])
    assert result["committed"] and result["cleanup_pending"]
    assert result["current"] == registry.deployment()["current"] == manager.current.manifest["artifact_id"] == two["artifact_id"]
    assert result["cleanup"][0]["status"] == "failed"
    assert result["cleanup"][0]["error"] == "RuntimeError"
    assert next(iter(manager._cleanup.values()))["deployment"].backend is old
    with pytest.raises(InferenceFailure) as failure:
        async with manager.acquire(two["artifact_id"]):
            pytest.fail("Failed old-runtime cleanup must keep inference unavailable")
    assert failure.value.status == 503
    assert (await manager.close())["status"] == "incomplete"
    assert new.unloaded
    assert not old.unloaded


async def test_unreadable_pointer_after_commit_error_reports_uncertain_not_retained(tmp_path, monkeypatch):
    one_package, one = make_package(tmp_path)
    two_package, two = make_package(tmp_path, "two")
    registry = Registry(tmp_path / "registry")
    registry.register(one_package)
    registry.register(two_package)
    app = create_app(Settings(initial_model=one["artifact_id"], admin_key="admin-fixture"), registry,
                     lambda manifest: FixtureBackend())
    async with app.router.lifespan_context(app):
        def ambiguous_write(artifact_id):
            registry.state_path.write_text("unreadable deployment fixture")
            raise OSError("fixture write failure with ambiguous disk outcome")

        monkeypatch.setattr(registry, "promote", ambiguous_write)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.post("/admin/promote", json={"artifact_id": two["artifact_id"]},
                                         headers={"Authorization": "Bearer admin-fixture"})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "deployment_state_uncertain"
            assert "retained" not in response.text
            assert app.state.manager.commit_uncertain
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.post("/v1/chat/completions", json=body(one))).status_code == 503
            status = (await client.get("/admin/status", headers={"Authorization": "Bearer admin-fixture"})).json()
            assert status["commit_uncertain"]
            assert status["deployment"]["error"] == "durable_state_unreadable"
