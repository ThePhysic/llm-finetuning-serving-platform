"""Bounded HTTP load generation with raw observations and explicit metric definitions."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import httpx


@dataclass(frozen=True)
class BenchmarkConfig:
    requests: int = 24
    concurrency: int = 1
    warmup: int = 2
    max_tokens: int = 64
    temperature: float = 0.0
    seed: int = 42
    stream: bool = True
    timeout_seconds: float = 120.0
    request_rate: float | None = None
    repetition: int = 0
    gpu_sample_interval: float = 0.5
    compute_price_per_hour: float | None = None

    def __post_init__(self) -> None:
        if min(self.requests, self.concurrency, self.max_tokens) < 1 or self.warmup < 0:
            raise ValueError("Request, concurrency and token limits must be positive")
        for value in (self.timeout_seconds, self.gpu_sample_interval):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Deadlines and sampling interval must be finite and positive")
        if self.request_rate is not None and (not math.isfinite(self.request_rate) or self.request_rate <= 0):
            raise ValueError("Request rate must be finite and positive")
        if self.compute_price_per_hour is not None and (
            not math.isfinite(self.compute_price_per_hour) or self.compute_price_per_hour < 0
        ):
            raise ValueError("Compute price assumption must be finite and nonnegative")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be within [0, 2]")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def environment() -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "peft", "vllm", "httpx", "fastapi"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        gpu = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {"platform": platform.platform(), "python": platform.python_version(),
            "client_software": versions, "gpu_inventory_csv": gpu}


def _gpu_observation() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        observations = []
        for line in result.stdout.strip().splitlines():
            index, memory, utilization = [x.strip() for x in line.split(",")]
            observations.append({"index": int(index), "device_used_memory_mib": float(memory),
                                 "utilization_percent": float(utilization)})
        return observations
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


async def _events(response: httpx.Response):
    """Parse SSE data blocks; token boundaries cannot be inferred from HTTP chunks."""
    lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if lines:
                yield "\n".join(lines)
                lines.clear()
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines:
        yield "\n".join(lines)


async def _request(client: httpx.AsyncClient, model: str, messages: list[dict[str, str]],
                   config: BenchmarkConfig, index: int) -> dict[str, Any]:
    started = time.perf_counter()
    observation: dict[str, Any] = {"index": index, "success": False, "status_code": None,
        "latency_seconds": None, "ttft_seconds": None, "tpot_seconds": None,
        "output_tokens": None, "input_tokens": None, "error": None, "model": None,
        "content_events": 0, "inter_content_event_seconds": []}
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": config.max_tokens,
        "temperature": config.temperature, "seed": config.seed, "stream": config.stream}
    if config.stream:
        payload["stream_options"] = {"include_usage": True}
    content: list[str] = []
    last_content: float | None = None
    first_content: float | None = None
    usage = None
    finish = None
    try:
        async with asyncio.timeout(config.timeout_seconds):
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                observation["status_code"] = response.status_code
                response.raise_for_status()
                if config.stream:
                    done = False
                    async for data in _events(response):
                        if data == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(data)
                        if "error" in chunk:
                            raise ValueError("backend_stream_error")
                        identity = chunk.get("model")
                        if identity:
                            if observation["model"] is not None and identity != observation["model"]:
                                raise ValueError("model_identity_changed_during_stream")
                            observation["model"] = identity
                        if chunk.get("usage") is not None:
                            usage = chunk["usage"]
                        for choice in chunk.get("choices", []):
                            if choice.get("finish_reason"):
                                finish = choice["finish_reason"]
                            text = choice.get("delta", {}).get("content")
                            if text:
                                now = time.perf_counter()
                                content.append(text)
                                observation["content_events"] += 1
                                if first_content is None:
                                    first_content = now
                                    observation["ttft_seconds"] = now - started
                                if last_content is not None:
                                    observation["inter_content_event_seconds"].append(now - last_content)
                                last_content = now
                    if not done or finish not in {"stop", "length"}:
                        raise ValueError("incomplete_stream")
                else:
                    body = json.loads(await response.aread())
                    if "error" in body:
                        raise ValueError("backend_response_error")
                    observation["model"] = body.get("model")
                    content = [body["choices"][0]["message"]["content"]]
                    finish = body["choices"][0]["finish_reason"]
                    usage = body.get("usage")
                    if finish not in {"stop", "length"}:
                        raise ValueError("invalid_finish_reason")
        if observation["model"] != model:
            raise ValueError("served_model_identity_mismatch")
        if not isinstance(usage, dict):
            raise ValueError("missing_measured_token_usage")
        for name in ("prompt_tokens", "completion_tokens"):
            if not isinstance(usage.get(name), int) or isinstance(usage[name], bool) or usage[name] < 0:
                raise ValueError("invalid_measured_token_usage")
        observation["input_tokens"] = usage["prompt_tokens"]
        observation["output_tokens"] = usage["completion_tokens"]
        if (observation["content_events"] > 1 and first_content is not None
                and last_content is not None and usage["completion_tokens"] > 1):
            observation["tpot_seconds"] = (last_content - first_content) / (usage["completion_tokens"] - 1)
        observation["success"] = True
        observation["finish_reason"] = finish
    except httpx.HTTPStatusError as exc:
        observation["error"] = f"http_{exc.response.status_code}"
    except (TimeoutError, httpx.TimeoutException):
        observation["error"] = "timeout"
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
        # Never put server response bodies or prompts into operational error logs.
        observation["error"] = str(exc) if isinstance(exc, ValueError) and str(exc).isidentifier() else type(exc).__name__
    observation["latency_seconds"] = time.perf_counter() - started
    observation["output_sha256"] = hashlib.sha256("".join(content).encode()).hexdigest()
    return observation


def _distribution(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99)}


def summarize(observations: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    if duration <= 0:
        raise ValueError("Measured duration must be positive")
    good = [row for row in observations if row["success"]]
    count = len(observations)
    total_tokens = sum(row["output_tokens"] or 0 for row in good)
    errors = {}
    for row in observations:
        if not row["success"]:
            key = row["error"] or "unknown"
            errors[key] = errors.get(key, 0) + 1
    return {"requests": count, "successful_requests": len(good), "wall_seconds": duration,
        "request_per_second": len(good) / duration, "output_tokens_per_second": total_tokens / duration,
        "total_output_tokens": total_tokens, "error_rate": (count - len(good)) / count if count else None,
        "rejection_rate": sum(row["status_code"] in {429, 503} for row in observations) / count if count else None,
        "timeout_rate": sum(row["error"] in {"timeout", "http_504"} for row in observations) / count if count else None,
        "latency_seconds": _distribution([row["latency_seconds"] for row in good]),
        "ttft_seconds": _distribution([row["ttft_seconds"] for row in good if row["ttft_seconds"] is not None]),
        "tpot_seconds": _distribution([row["tpot_seconds"] for row in good if row["tpot_seconds"] is not None]),
        "input_tokens": _distribution([row["input_tokens"] for row in good]),
        "output_tokens": _distribution([row["output_tokens"] for row in good]), "errors": errors}


async def run_benchmark(base_url: str, model: str, prompts: list[list[dict[str, str]]],
                        output_dir: Path, config: BenchmarkConfig,
                        metadata: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    """Run a bounded closed-loop workload (or paced arrivals with client backpressure)."""
    if not prompts:
        raise ValueError("At least one prompt is required")
    for required in ("weights_id", "tokenizer_id", "engine", "engine_settings", "cache_state"):
        if required not in metadata:
            raise ValueError(f"Benchmark metadata is missing {required}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt_hash = hashlib.sha256(_json(prompts).encode()).hexdigest()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    raw: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    gpu_stop = asyncio.Event()
    async def sample_gpu():
        while not gpu_stop.is_set():
            gpu_samples.append({"time": time.time(), "devices": await asyncio.to_thread(_gpu_observation)})
            try:
                await asyncio.wait_for(gpu_stop.wait(), config.gpu_sample_interval)
            except TimeoutError:
                pass
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers,
                                timeout=config.timeout_seconds,
                                limits=httpx.Limits(max_connections=config.concurrency,
                                                   max_keepalive_connections=config.concurrency)) as client:
        warmups = []
        for i in range(config.warmup):
            warmups.append(await _request(client, model, prompts[i % len(prompts)], config, -i - 1))
        if any(not row["success"] for row in warmups):
            (output_dir / "warmup.json").write_text(_json(warmups), encoding="utf-8")
            raise RuntimeError("Warmup failed; inspect sanitized warmup.json observations")
        sampler = asyncio.create_task(sample_gpu())
        started = time.perf_counter()
        counter = 0
        lock = asyncio.Lock()
        async def worker():
            nonlocal counter
            while True:
                async with lock:
                    if counter >= config.requests:
                        return
                    index = counter
                    counter += 1
                scheduled = started + index / config.request_rate if config.request_rate else time.perf_counter()
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                dispatched = time.perf_counter()
                row = await _request(client, model, prompts[index % len(prompts)], config, index)
                row["client_schedule_delay_seconds"] = max(0, dispatched - scheduled)
                raw.append(row)
        try:
            await asyncio.gather(*(worker() for _ in range(config.concurrency)))
            wall_seconds = time.perf_counter() - started
        finally:
            gpu_stop.set()
            await sampler
    raw.sort(key=lambda row: row["index"])
    summary = summarize(raw, wall_seconds)
    memory = [device["device_used_memory_mib"] for sample in gpu_samples for device in sample["devices"]]
    summary["gpu_device_used_peak_mib"] = max(memory) if memory else None
    summary["gpu_device_used_median_mib"] = sorted(memory)[len(memory) // 2] if memory else None
    if config.compute_price_per_hour is not None:
        price = config.compute_price_per_hour * wall_seconds / 3600
        summary["estimated_compute_cost"] = price
        summary["estimated_compute_cost_per_million_output_tokens"] = (
            price * 1_000_000 / summary["total_output_tokens"] if summary["total_output_tokens"] else None)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model, "config": asdict(config), "metadata": {**metadata, "prompt_set_sha256": prompt_hash},
        "environment": environment(), "summary": summary,
        "definitions": {"latency": "HTTP dispatch to complete response, including transport and queue",
            "TTFT": "HTTP dispatch to first nonempty content event received by the client",
            "TPOT": "(last content event time - first content event time)/(reported completion tokens-1); streaming event batching can bias this estimate",
            "inter_content_event": "Time between nonempty SSE content events; events may contain multiple tokens",
            "throughput": "Successful measured requests or reported output tokens divided by full measured workload wall time",
            "GPU": "nvidia-smi sampled device-wide used memory, includes other processes and runtime reservations; not PyTorch allocated memory",
            "arrival": "Bounded closed-loop workers, optionally paced; late arrivals wait at client and schedule delay is recorded"},
        "limitations": ["Small samples give unstable tail estimates; retain repetitions and raw observations.",
                         "GPU sampling is best effort and may miss short peaks; it adds monitoring overhead.",
                         "Client software inventory does not establish a remote engine's software; supply engine versions in metadata."]}
    (output_dir / "requests.jsonl").write_text("".join(_json(row) + "\n" for row in raw), encoding="utf-8")
    (output_dir / "gpu.jsonl").write_text("".join(_json(row) + "\n" for row in gpu_samples), encoding="utf-8")
    (output_dir / "report.json").write_text(_json(report) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        f"# Observed serving benchmark\n\nModel `{model}`; engine `{metadata['engine']}`.\n\n"
        f"{summary['successful_requests']}/{summary['requests']} successful requests in {wall_seconds:.3f}s. "
        f"{summary['output_tokens_per_second']:.3f} output tokens/s; {summary['request_per_second']:.3f} requests/s.\n\n"
        f"Latency seconds: `{_json(summary['latency_seconds'])}`. TTFT seconds: `{_json(summary['ttft_seconds'])}`.\n\n"
        "Raw observations and metric definitions are in report.json and requests.jsonl. Tail estimates require adequate samples.\n",
        encoding="utf-8")
    return report


def compare_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Refuse accidental weight/prompt/decoding changes before engine comparison."""
    if len(reports) < 2:
        raise ValueError("At least two reports are required")
    first = reports[0]
    for report in reports[1:]:
        for key in ("weights_id", "tokenizer_id", "prompt_set_sha256"):
            if report["metadata"][key] != first["metadata"][key]:
                raise ValueError(f"Unfair comparison: {key} differs")
        for key in ("max_tokens", "temperature", "seed", "stream", "requests", "warmup",
                    "request_rate", "timeout_seconds", "gpu_sample_interval"):
            if report["config"][key] != first["config"][key]:
                raise ValueError(f"Unfair comparison: {key} differs")
    return {"schema_version": 1, "comparison": [{"model": r["model"], "engine": r["metadata"]["engine"],
        "config": r["config"], "engine_settings": r["metadata"]["engine_settings"],
        "concurrency": r["config"]["concurrency"], "cache_state": r["metadata"]["cache_state"],
        "summary": r["summary"]} for r in reports],
        "limitations": ["Review engine settings, cache state, device contention and repetitions before attributing differences.",
                         "Serving performance does not establish equivalent generated answer quality."]}
