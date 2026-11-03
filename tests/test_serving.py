import asyncio
from contextlib import asynccontextmanager
import json

import httpx
import pytest

from llm_platform.contracts import TokenChunk
from llm_platform.registry import Registry
from llm_platform.serving import Settings, create_app
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
