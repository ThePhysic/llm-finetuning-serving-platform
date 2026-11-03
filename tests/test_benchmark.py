import json
from copy import deepcopy

import httpx
import pytest

from llm_platform.benchmark import (BenchmarkConfig, _adapter_precision, _request,
                                    _runtime_gpu_observation, compare_reports, summarize)


def test_percentiles_and_failed_requests_do_not_count_as_throughput():
    rows = [dict(success=True, output_tokens=10, input_tokens=5, latency_seconds=1,
                 ttft_seconds=.2, tpot_seconds=.01, status_code=200, error=None),
            dict(success=False, output_tokens=100, input_tokens=5, latency_seconds=2,
                 ttft_seconds=None, tpot_seconds=None, status_code=429, error="http_429")]
    report = summarize(rows, 2)
    assert report["output_tokens_per_second"] == 5
    assert report["rejection_rate"] == .5
    assert report["latency_seconds"]["p99"] == 1


@pytest.mark.parametrize("kwargs", [{"requests": 0}, {"request_rate": float("nan")},
                                    {"concurrency": 0}, {"compute_price_per_hour": -1}])
def test_invalid_load_configuration(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


async def test_stream_requires_measured_usage_complete_event_and_identity():
    model = "a" * 64
    def handler(request):
        chunks = [dict(model=model, choices=[dict(delta={"content": "hello"}, finish_reason=None)]),
                  dict(model=model, choices=[dict(delta={}, finish_reason="stop")]),
                  dict(model=model, choices=[], usage={"prompt_tokens": 5, "completion_tokens": 1})]
        body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        row = await _request(client, model, [{"role": "user", "content": "hi"}], BenchmarkConfig(), 0)
    assert row["success"] is True
    assert row["output_tokens"] == 1
    assert row["tpot_seconds"] is None


async def test_stream_error_is_not_generated_text_or_success():
    def handler(request):
        return httpx.Response(200, text='data: {"error":{"message":"private prompt"}}\n\ndata: [DONE]\n\n')
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        row = await _request(client, "a" * 64, [], BenchmarkConfig(), 0)
    assert not row["success"]
    assert "private" not in str(row)


@pytest.mark.parametrize("payload", [[], {"choices": [42]}, {"choices": [{"delta": []}]},
                                     {"choices": [{"delta": {"content": 42}}]}])
async def test_malformed_sse_is_recorded_as_failure_without_aborting_benchmark(payload):
    def handler(request):
        return httpx.Response(200, text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n")
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        row = await _request(client, "a" * 64, [], BenchmarkConfig(), 0)
    assert row["success"] is False
    assert row["error"] == "invalid_response_shape"
    assert row["latency_seconds"] >= 0


@pytest.mark.parametrize("payload", [[], {"choices": []}, {"choices": [42]},
                                     {"choices": [{"message": {"content": 42}}]}])
async def test_malformed_sync_response_is_recorded_as_failure(payload):
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))) as client:
        row = await _request(client, "a" * 64, [], BenchmarkConfig(stream=False), 0)
    assert row["success"] is False
    assert row["error"] == "invalid_response_shape"


def test_comparison_rejects_changed_weights():
    report = {"metadata": {"weights_id": "base", "tokenizer_id": "token", "prompt_set_sha256": "p"},
              "config": {k: 1 for k in ("max_tokens", "temperature", "seed", "stream", "requests", "warmup")}}
    changed = deepcopy(report)
    changed["metadata"]["weights_id"] = "adapted"
    with pytest.raises(ValueError, match="weights_id"):
        compare_reports([report, changed])


async def test_runtime_memory_sampler_preserves_unavailable_instead_of_fabricating_zero():
    def handler(request):
        assert request.url.path == "/metrics"
        return httpx.Response(200, text=("llm_gpu_memory_bytes 123.0\n"
            "llm_gpu_peak_allocated_bytes 456.0\nllm_gpu_reserved_bytes NaN\n"))
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        result = await _runtime_gpu_observation(client)
    assert result == {"allocated_bytes": 123, "session_peak_allocated_bytes": 456, "reserved_bytes": None}


def test_serialized_weight_identity_does_not_hide_adapter_compute_precision():
    mixed = {"engine_settings": {"adapter": True, "dtype": "bfloat16", "base_compute_dtype": "bfloat16",
                                 "adapter_compute_dtype": "float32", "adapter_storage_dtype": "float32"}}
    with pytest.raises(ValueError, match="compute precision"):
        _adapter_precision(mixed)
    del mixed["engine_settings"]["adapter_compute_dtype"]
    with pytest.raises(ValueError, match="verified"):
        _adapter_precision(mixed)
    mixed["engine_settings"]["adapter_compute_dtype"] = "bfloat16"
    assert _adapter_precision(mixed) == ("bfloat16", "bfloat16", "float32")
