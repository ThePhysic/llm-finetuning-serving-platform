"""Execution-control tests with explicit test doubles, never GPU evidence."""
import asyncio
from contextlib import nullcontext
import json
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from llm_platform.backends import TransformersBackend, VLLMBackend
from llm_platform.backends.vllm_backend import verify_runtime_identity
from llm_platform.backends.launch_vllm import launch_command
from llm_platform.contracts import GenerationConfig


def identity_fixture():
    manifest = {"artifact_id": "a" * 64, "base_model": {"revision": "b" * 40},
                "tokenizer": {"revision": "c" * 40, "chat_template_sha256": "d" * 64},
                "runtime": {"quantization": "none", "dtype": "bfloat16"}}
    identity = {"artifact_id": "a" * 64, "base_revision": "b" * 40,
                "tokenizer_revision": "c" * 40, "chat_template_sha256": "d" * 64,
                "quantization": "none", "dtype": "bfloat16", "model_name": "a" * 64}
    return manifest, identity


@pytest.mark.parametrize("field", ["artifact_id", "base_revision", "tokenizer_revision",
                                   "chat_template_sha256", "quantization", "dtype", "model_name"])
def test_vllm_rejects_unbound_runtime(field):
    manifest, identity = identity_fixture()
    identity[field] = "incorrect"
    with pytest.raises(ValueError):
        verify_runtime_identity(manifest, identity)


async def test_vllm_uses_authoritative_usage_and_rejects_truncated_stream():
    backend = VLLMBackend()
    backend.model_name = "a" * 64
    finish = {"model": backend.model_name, "choices": [{"delta": {}, "finish_reason": "stop"}],
              "usage": {"completion_tokens": 7}}
    body = "data: " + json.dumps({"model": backend.model_name, "choices": [{"delta": {"content": "hello"}}]})
    body += "\n\ndata: " + json.dumps(finish) + "\n\ndata: [DONE]\n\n"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    backend.client = httpx.AsyncClient(transport=transport, base_url="http://test")
    chunks = [chunk async for chunk in backend.stream([], GenerationConfig(), "r1")]
    assert "".join(c.text for c in chunks) == "hello"
    assert sum(c.token_count for c in chunks) == 7
    assert chunks[-1].finish_reason == "stop"
    await backend.unload()
    backend.model_name = "a" * 64
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body.replace("data: [DONE]", ""))), base_url="http://test")
    with pytest.raises(RuntimeError, match="Incomplete"):
        _ = [chunk async for chunk in backend.stream([], GenerationConfig(), "r2")]
    assert not backend._active
    await backend.unload()


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self

    def numel(self):
        return len(self.values)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.values[item[1]]
        return self.values[item]


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("return_dict"):
            return {"input_ids": FakeTensor([1, 2])}
        return [1, 2]

    def decode(self, values, **kwargs):
        return "".join(chr(v) for v in values)


class FakeStreamer:
    def __init__(self, tokenizer, skip_prompt, **kwargs):
        self.skip_prompt = skip_prompt
        self.next_tokens_are_prompt = True
        self.tokenizer = tokenizer

    def put(self, value):
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.on_finalized_text(self.tokenizer.decode(value.values))

    def end(self):
        self.on_finalized_text("", stream_end=True)


class FakeModel:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=1000)

    def __init__(self, text="safe<STOP>leak", fail=False):
        self.text = text
        self.fail = fail
        self.finished = False

    def generate(self, input_ids, streamer, stopping_criteria, **kwargs):
        generated = [1, 2]
        try:
            streamer.put(input_ids)
            if self.fail:
                raise RuntimeError("private prompt details must not escape")
            for character in self.text:
                time.sleep(0.005)
                generated.append(ord(character))
                streamer.put(FakeTensor([ord(character)]))
                if any(rule(FakeTensor(generated), None) for rule in stopping_criteria):
                    break
            streamer.end()
        finally:
            self.finished = True


@pytest.fixture
def reference_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: None, inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list, TextIteratorStreamer=FakeStreamer))
    backend = TransformersBackend(device="cpu", dtype="float32")
    backend.model = FakeModel()
    backend.tokenizer = FakeTokenizer()
    return backend


async def test_reference_stop_strings_never_leak_partial_stop(reference_backend):
    backend = reference_backend
    chunks = [chunk async for chunk in backend.stream([], GenerationConfig(stop=("<STOP>",)), "x")]
    assert "".join(c.text for c in chunks) == "safe"
    assert chunks[-1].finish_reason == "stop"
    assert backend.model.finished
    assert not backend._workers and not backend._cancel


async def test_reference_disconnect_joins_worker_before_releasing_model(reference_backend):
    backend = reference_backend
    backend.model = FakeModel("x" * 500)
    model = backend.model

    async def consume():
        iterator = backend.stream([], GenerationConfig(max_tokens=600), "disconnect")
        try:
            async for _ in iterator:
                await asyncio.sleep(0)
        finally:
            await iterator.aclose()

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.finished
    assert not backend._workers
    await backend.unload()
    assert backend.model is None


async def test_reference_errors_are_separate_from_generated_text(reference_backend):
    backend = reference_backend
    backend.model = FakeModel(fail=True)
    with pytest.raises(RuntimeError, match="Model execution failed: RuntimeError") as captured:
        _ = [chunk async for chunk in backend.stream([], GenerationConfig(), "failure")]
    assert "private prompt" not in str(captured.value)
    assert not backend._workers


async def test_load_cancellation_waits_for_loader_and_frees_candidate(monkeypatch, tmp_path):
    backend = TransformersBackend()
    manifest = {"artifact_id": "a" * 64}
    monkeypatch.setattr("llm_platform.registry.verify_package", lambda path: manifest)
    completed = []

    def slow_load(package, meta):
        time.sleep(0.07)
        backend.model = object()
        completed.append(True)

    async def unload():
        backend.model = None

    monkeypatch.setattr(backend, "_load_sync", slow_load)
    monkeypatch.setattr(backend, "unload", unload)
    task = asyncio.create_task(backend.load(tmp_path, manifest))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed and backend.model is None


async def test_encoding_failure_does_not_leave_active_request(reference_backend):
    class BrokenTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise ValueError("Bad template")

    reference_backend.tokenizer = BrokenTokenizer()
    with pytest.raises(ValueError, match="Bad template"):
        _ = [c async for c in reference_backend.stream([], GenerationConfig(), "bad-encode")]
    assert not reference_backend._cancel and not reference_backend._workers


def test_launcher_pins_revision_and_explicit_cache_state(tmp_path):
    manifest, _ = identity_fixture()
    manifest.update(kind="base", base_model={"id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "b" * 40},
                    tokenizer={"revision": "c" * 40, "path": "tokenizer", "chat_template_sha256": "d" * 64},
                    runtime={"backend": "vllm", "quantization": "none", "dtype": "bfloat16"})
    command, identity = launch_command(tmp_path / "path with spaces", manifest)
    assert command[command.index("--revision") + 1] == "b" * 40
    assert command[command.index("--served-model-name") + 1] == "a" * 64
    assert "--no-enable-prefix-caching" in command
    assert "--disable-log-requests" in command
    assert "--enforce-eager" in command
    assert identity["command"] == command


async def test_vllm_readiness_detects_crash_and_identity_change():
    backend = VLLMBackend()
    backend.model_name = "a" * 64
    healthy = [True]
    model_name = [backend.model_name]

    def route(request):
        if not healthy[0]:
            raise httpx.ConnectError("engine exited", request=request)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"data": [{"id": model_name[0]}]})

    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(route), base_url="http://engine")
    await backend._probe_once()
    assert backend.health()["ready"]
    healthy[0] = False
    await backend._probe_once()
    assert not backend.health()["ready"]
    healthy[0] = True
    model_name[0] = "wrong-revision"
    await backend._probe_once()
    assert not backend.health()["ready"]
    model_name[0] = backend.model_name
    await backend._probe_once()
    assert backend.health()["ready"]
    await backend.unload()
    assert not backend.health()["ready"]


async def test_vllm_context_overflow_rejected_before_network_request():
    backend = VLLMBackend()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: pytest.fail("Context overflow must not reach the engine")), base_url="http://engine")
    backend.tokenizer = FakeTokenizer()
    backend.max_model_len = 32
    with pytest.raises(ValueError, match="context length"):
        _ = [c async for c in backend.stream([], GenerationConfig(max_tokens=64), "overflow")]
    await backend.unload()
