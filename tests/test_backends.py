"""Execution-control tests with explicit test doubles, never GPU evidence."""
import asyncio
from contextlib import nullcontext
import hashlib
import json
import struct
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from llm_platform.backends import TransformersBackend, VLLMBackend
from llm_platform.backends.vllm_backend import verify_runtime_identity
from llm_platform.backends.launch_vllm import launch_command
from llm_platform.backends.precision import adapter_storage_dtype, inspect_lora_compute_dtype
from llm_platform.contracts import GenerationConfig, serving_decoding_policy


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
    backend._decoding_policy = serving_decoding_policy(151645, 151643, 151643)
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
    backend._decoding_policy = serving_decoding_policy(151645, 151643, 151643)
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
    eos_token_id = 9
    bos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("return_dict"):
            return {"input_ids": FakeTensor([1, 2])}
        return [1, 2]

    def decode(self, values, **kwargs):
        return "".join(chr(v) for v in values)


@pytest.mark.parametrize("backend_type", [TransformersBackend, VLLMBackend])
@pytest.mark.parametrize("batched", [False, True])
def test_backend_token_counts_use_mapping_ids_not_keys(backend_type, batched):
    class MappingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            values = [1, 2, 3, 4, 5]
            return {"input_ids": [values] if batched else values, "attention_mask": [1] * 5}

    backend = backend_type()
    backend.tokenizer = MappingTokenizer()
    assert backend.count_tokens([{"role": "user", "content": "x"}]) == 5


async def test_vllm_mapping_prompt_count_refuses_overflow_before_worker_dispatch():
    backend = VLLMBackend()
    backend.max_model_len = 7
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: {
        "input_ids": [[1, 2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1, 1]]})
    dispatched = []
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: dispatched.append(request) or httpx.Response(500)), base_url="http://worker")
    try:
        with pytest.raises(ValueError, match="context length"):
            _ = [chunk async for chunk in backend.stream([{"role": "user", "content": "x"}],
                                                         GenerationConfig(max_tokens=3), "overflow")]
        assert not dispatched and not backend._active
    finally:
        await backend.unload()


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
        self.generation_config = SimpleNamespace(repetition_penalty=1.1, top_p=0.8, top_k=20,
                                                 eos_token_id=[9, 10], bad_words_ids=[[11]])
        self.generated_kwargs = None

    def generate(self, input_ids, streamer, stopping_criteria, **kwargs):
        self.generated_kwargs = kwargs
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
    class FakeHFGenerationConfig:
        def __init__(self, **kwargs):
            self.bad_words_ids = None
            self.forced_bos_token_id = None
            self.__dict__.update(kwargs)

        @classmethod
        def from_model_config(cls, config):
            return cls(bos_token_id=7, eos_token_id=[9, 10], repetition_penalty=1.5)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list, TextIteratorStreamer=FakeStreamer,
        GenerationConfig=FakeHFGenerationConfig))
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


@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_publisher_generation_settings_do_not_override_serving_policy(reference_backend, temperature):
    model = reference_backend.model
    publisher = model.generation_config
    _ = [c async for c in reference_backend.stream([], GenerationConfig(temperature=temperature), "policy")]
    passed = model.generated_kwargs
    assert passed["generation_config"] is not publisher
    assert passed["generation_config"].bad_words_ids is None
    assert passed["generation_config"].forced_bos_token_id is None
    assert passed["generation_config"].eos_token_id == 9
    assert passed["eos_token_id"] == 9  # Publisher supplied a wider [9,10] EOS list.
    assert passed["repetition_penalty"] == 1.0
    assert passed["top_p"] == 1.0 and passed["top_k"] == 0
    assert passed["num_beams"] == 1 and passed["num_return_sequences"] == 1
    assert passed["do_sample"] is (temperature > 0)
    assert passed["temperature"] == (temperature if temperature else 1.0)
    assert passed["min_new_tokens"] == 0


async def test_vllm_request_explicitly_matches_platform_sampling_and_single_eos():
    backend = VLLMBackend()
    backend.model_name = "a" * 64
    backend._decoding_policy = serving_decoding_policy(151645, 151643, 151643)
    observed = []
    def route(request):
        observed.append(json.loads(request.content))
        response = {"model": backend.model_name,
                    "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1}}
        return httpx.Response(200, text="data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n")
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(route), base_url="http://engine")
    _ = [c async for c in backend.stream([], GenerationConfig(temperature=0.7, stop=("STOP",)), "policy")]
    payload = observed[0]
    assert payload["repetition_penalty"] == 1.0
    assert payload["top_p"] == 1.0 and payload["top_k"] == -1 and payload["min_p"] == 0
    assert payload["presence_penalty"] == 0 and payload["frequency_penalty"] == 0
    assert payload["n"] == 1 and payload["use_beam_search"] is False
    assert payload["stop_token_ids"] == [151645] and payload["ignore_eos"] is False
    assert payload["min_tokens"] == 0 and payload["stop"] == ["STOP"]
    assert backend.health()["decoding_policy"] == serving_decoding_policy(151645, 151643, 151643)
    await backend.unload()


def test_decoding_policy_hash_covers_effective_eos_and_is_copy_safe():
    first = serving_decoding_policy(9, 7, 0)
    second = serving_decoding_policy(9, 7, 0)
    assert first == second
    assert first["sha256"] != serving_decoding_policy(10, 7, 0)["sha256"]
    first["eos_token_ids"].append(99)
    assert second["eos_token_ids"] == [9]
    with pytest.raises(ValueError, match="single eos"):
        serving_decoding_policy([9, 10], 7, 0)


async def test_cuda_cleanup_runs_off_event_loop_and_retains_ownership_on_cancellation(reference_backend, monkeypatch):
    backend = reference_backend
    entered = asyncio.Event()
    released = []
    loop = asyncio.get_running_loop()

    def slow_cleanup():
        loop.call_soon_threadsafe(entered.set)
        time.sleep(0.07)
        backend.model = None
        released.append(True)

    monkeypatch.setattr(backend, "_unload_sync", slow_cleanup)
    task = asyncio.create_task(backend.unload())
    await asyncio.wait_for(entered.wait(), 1)
    assert backend.health()["cleanup_pending"]
    assert not backend.health()["ready"]
    task.cancel()
    # The cancellation cannot abandon the thread. The event loop still ticks
    # while cleanup is in progress, allowing health and admin requests.
    await asyncio.sleep(0.01)
    assert not task.done() and backend._cleanup_task is not None
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released and backend.model is None and backend._cleanup_task is None


async def test_vllm_rejects_additional_architectural_eos_before_connecting(monkeypatch, tmp_path):
    manifest, identity = identity_fixture()
    manifest["kind"] = "base"
    manifest["base_model"]["id"] = "base-model"
    manifest["tokenizer"].update(id="base-model", revision=manifest["base_model"]["revision"], path="tokenizer",
                                  chat_template_sha256=hashlib.sha256(b"template").hexdigest())
    manifest["runtime"]["backend"] = "vllm"
    identity.update(tokenizer_revision=manifest["tokenizer"]["revision"],
                    chat_template_sha256=manifest["tokenizer"]["chat_template_sha256"], max_model_len=32)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = "template"
    monkeypatch.setattr("llm_platform.registry.verify_package", lambda path: manifest)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(
            eos_token_id=[9, 10], bos_token_id=7))))
    backend = VLLMBackend(identity_path=identity_path)
    with pytest.raises(ValueError, match="architectural EOS"):
        await backend.load(tmp_path, manifest)
    assert backend.client is None and not backend.health()["ready"]


def write_adapter_weights(path, dtype="F32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"layer.lora_A.weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


def test_adapter_storage_precision_comes_from_header_not_manifest(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    write_adapter_weights(path, "F32")
    assert adapter_storage_dtype(path) == "float32"
    write_adapter_weights(path, "BF16")
    assert adapter_storage_dtype(path) == "bfloat16"
    path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="header length"):
        adapter_storage_dtype(path)


def test_lora_runtime_parameter_precision_cannot_silently_upcast():
    model = SimpleNamespace(named_parameters=lambda: iter([
        ("layer.lora_A.default.weight", SimpleNamespace(dtype="torch.bfloat16")),
        ("layer.lora_B.default.weight", SimpleNamespace(dtype="torch.float32")),
    ]))
    with pytest.raises(ValueError, match="runtime dtype"):
        inspect_lora_compute_dtype(model, "bfloat16")


@pytest.mark.parametrize("ignore_disable_flag", [False, True])
def test_actual_adapter_loader_disables_peft_upcast_and_checks_parameters(monkeypatch, tmp_path, ignore_disable_flag):
    adapter_config = {"base_model_name_or_path": "base-model", "revision": "a" * 40, "r": 2}
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text(json.dumps(adapter_config))
    write_adapter_weights(tmp_path / "adapter" / "adapter_model.safetensors")
    manifest = {"kind": "adapter", "base_model": {"id": "base-model", "revision": "a" * 40},
        "tokenizer": {"id": "base-model", "revision": "a" * 40, "path": "tokenizer",
                      "chat_template_sha256": hashlib.sha256(b"template").hexdigest()},
        "adapter": {"path": "adapter", "config": adapter_config},
        "runtime": {"backend": "transformers", "dtype": "bfloat16", "quantization": "none"}}
    class LoaderTokenizer(FakeTokenizer):
        chat_template = "template"
        def __len__(self):
            return 32
    class LoaderModel:
        config = SimpleNamespace()
        adapter_dtype = "torch.float32"
        def get_input_embeddings(self):
            return SimpleNamespace(num_embeddings=64, weight=SimpleNamespace(dtype="torch.bfloat16"))
        def to(self, device):
            return self
        def named_parameters(self):
            return iter([("layer.lora_A.default.weight", SimpleNamespace(dtype=self.adapter_dtype)),
                         ("layer.lora_B.default.weight", SimpleNamespace(dtype=self.adapter_dtype))])
        def eval(self):
            return self
    observed = []
    def peft_load(model, path, **kwargs):
        observed.append(kwargs)
        # Reproduce PEFT's default inference upcast when the disable flag is
        # omitted; the second case emulates a runtime that ignores the flag.
        model.adapter_dtype = "torch.float32" if kwargs.get("autocast_adapter_dtype", True) or ignore_disable_flag else "torch.bfloat16"
        return model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="torch.bfloat16", cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: LoaderModel()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: LoaderTokenizer()),
        BitsAndBytesConfig=object))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=peft_load)))
    backend = TransformersBackend(device="cpu", dtype="bfloat16")
    # Exercise the GPU-style loader without CUDA allocations using a test-only
    # device string; production accepts only a real available CUDA device.
    backend.device = "test-device"
    monkeypatch.setattr(backend, "_generation_parameters", lambda *args: {
        "generation_config": SimpleNamespace(bos_token_id=7)})
    if ignore_disable_flag:
        with pytest.raises(ValueError, match="runtime dtype"):
            backend._load_sync(tmp_path, manifest)
        assert backend.model is None
    else:
        backend._load_sync(tmp_path, manifest)
        assert backend.health()["base_compute_dtype"] == "bfloat16"
        assert backend.health()["adapter_compute_dtype"] == "bfloat16"
        assert backend.health()["adapter_storage_dtype"] == "float32"
    assert observed[0]["autocast_adapter_dtype"] is False
    assert observed[0]["is_trainable"] is False


def test_adapter_launcher_explicitly_selects_dtype_and_records_storage(tmp_path):
    manifest, _ = identity_fixture()
    manifest.update(kind="adapter", base_model={"id": "base-model", "revision": "b" * 40},
        tokenizer={"revision": "b" * 40, "path": "tokenizer", "chat_template_sha256": "d" * 64},
        adapter={"path": "adapter", "config": {"r": 8}},
        runtime={"backend": "vllm", "quantization": "none", "dtype": "bfloat16"})
    write_adapter_weights(tmp_path / "adapter" / "adapter_model.safetensors")
    command, identity = launch_command(tmp_path, manifest)
    assert command[command.index("--lora-dtype") + 1] == "bfloat16"
    assert identity["precision"] == {"base_compute_dtype": "bfloat16", "adapter_compute_dtype": "bfloat16",
                                     "adapter_storage_dtype": "float32"}


@pytest.mark.parametrize("missing_precision", [True, False])
async def test_vllm_rejects_missing_or_inconsistent_adapter_precision(monkeypatch, tmp_path, missing_precision):
    manifest, identity = identity_fixture()
    manifest.update(kind="adapter", adapter={"path": "adapter", "config": {"r": 8}},
                    runtime={"backend": "vllm", "dtype": "bfloat16", "quantization": "none"})
    write_adapter_weights(tmp_path / "adapter" / "adapter_model.safetensors")
    if not missing_precision:
        identity["precision"] = {"base_compute_dtype": "bfloat16", "adapter_compute_dtype": "float32",
                                 "adapter_storage_dtype": "float32"}
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    monkeypatch.setattr("llm_platform.registry.verify_package", lambda path: manifest)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=object, AutoTokenizer=object))
    backend = VLLMBackend(identity_path=identity_path)
    with pytest.raises(ValueError, match="runtime precision"):
        await backend.load(tmp_path, manifest)
    assert backend.client is None


async def test_vllm_prepare_keeps_loop_responsive_and_settles_cancelled_worker(monkeypatch, tmp_path):
    backend = VLLMBackend()
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    ownership = []

    def blocked_prepare(package, manifest):
        ownership.append(threading.get_ident())
        entered.set()
        if not release.wait(2):
            raise RuntimeError("CPU preparation fixture was not released")
        ownership.append("settled")
        return {"manifest": manifest}

    monkeypatch.setattr(backend, "_prepare_sync", blocked_prepare)
    loading = asyncio.create_task(backend.load(tmp_path, {"artifact_id": "a" * 64}))
    assert await asyncio.to_thread(entered.wait, 1)
    heartbeat = 0
    for _ in range(3):
        await asyncio.sleep(0.005)
        heartbeat += 1
    assert heartbeat == 3 and ownership[0] != main_thread
    loading.cancel()
    await asyncio.sleep(0.01)
    assert not loading.done()
    assert backend.client is None and backend._probe_task is None
    loading.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(loading, 1)
    assert ownership[-1] == "settled"
    assert backend.client is backend.tokenizer is backend.manifest is None
    assert backend._probe_task is None and not backend.health()["ready"]


@pytest.mark.parametrize("template", [None, "", "  ", {"default": "template"}, ["template"]])
async def test_vllm_invalid_chat_template_fails_before_http_client_creation(monkeypatch, tmp_path, template):
    manifest, identity = identity_fixture()
    manifest.update(kind="base")
    manifest["base_model"]["id"] = "base-model"
    manifest["tokenizer"].update(id="base-model", revision=manifest["base_model"]["revision"], path="tokenizer")
    manifest["runtime"]["backend"] = "vllm"
    identity.update(tokenizer_revision=manifest["tokenizer"]["revision"], max_model_len=32)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    monkeypatch.setattr("llm_platform.registry.verify_package", lambda path: manifest)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(chat_template=template)),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: pytest.fail("Invalid template must fail before architecture resolution"))))
    backend = VLLMBackend(identity_path=identity_path)
    with pytest.raises(ValueError, match="nonempty string chat template"):
        await backend.load(tmp_path, manifest)
    assert backend.client is None and backend._probe_task is None
