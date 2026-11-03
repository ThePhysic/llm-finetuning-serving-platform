"""Run real, isolated HTTP serving experiments after the quality experiment.

The process owns only its child process groups. Credentials exist exclusively in
memory and child environments, never protocol/evidence files or command lines.
No GPU experiment is launched by importing this module.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

import httpx

from llm_platform.benchmark import BenchmarkConfig, compare_reports, environment, run_benchmark
from llm_platform.prompts import ZERO_SHOT
from llm_platform.registry import Registry, create_package, verify_package


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def semantic_ids(manifest: dict[str, Any]) -> dict[str, str]:
    """Runtime-package IDs differ across engines; compare actual weight identity."""
    adapter_files = {name: value for name, value in manifest["files"].items()
                     if manifest["adapter"] and name.startswith(manifest["adapter"]["path"] + "/")}
    token_files = {name: value for name, value in manifest["files"].items()
                   if name.startswith(manifest["tokenizer"]["path"] + "/")}
    return {"weights_id": digest({"base": manifest["base_model"], "adapter_files": adapter_files,
                                   "adapter_config": manifest["adapter"],
                                   "quantization": manifest["runtime"]["quantization"],
                                   "dtype": manifest["runtime"]["dtype"]}),
            "tokenizer_id": digest({"tokenizer": manifest["tokenizer"], "files": token_files})}


def available_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def software_inventory(python: Path | str) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,platform; "
        "names=['torch','transformers','peft','vllm','fastapi','httpx']; "
        "installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
        "print(json.dumps({'python':platform.python_version(),'versions':{n:installed.get(n) for n in names}}))"
    )
    completed = subprocess.run([str(python), "-c", code], check=True, capture_output=True,
                               text=True, timeout=60)
    return json.loads(completed.stdout)


@dataclass
class OwnedProcess:
    process: subprocess.Popen
    log: Any
    started: float
    label: str

    @classmethod
    def launch(cls, command: list[str], log_path: Path, env: dict[str, str], label: str) -> "OwnedProcess":
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        try:
            child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        except BaseException:
            handle.close()
            raise
        return cls(child, handle, time.perf_counter(), label)

    async def close(self) -> dict[str, Any]:
        started = time.perf_counter()
        forced = False
        # Include descendants when an engine parent exited early. Its dedicated
        # session/group remains associated with those owned workers until empty.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self.process.poll() is None:
            try:
                await asyncio.to_thread(self.process.wait, timeout=30)
            except subprocess.TimeoutExpired:
                forced = True
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(self.process.wait, timeout=10)
        self.log.close()
        return {"label": self.label, "exit_code": self.process.returncode, "forced_shutdown": forced,
                "shutdown_seconds": time.perf_counter() - started}


async def wait_ready(process: OwnedProcess, url: str, expected_id: str, timeout: float,
                     api_key: str | None = None, gateway: bool = True) -> float:
    deadline = time.perf_counter() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=3) as client:
        while time.perf_counter() < deadline:
            if process.process.poll() is not None:
                raise RuntimeError(f"Owned {process.label} exited before readiness; inspect its runtime log")
            try:
                if gateway:
                    response = await client.get("/health/ready")
                    if response.status_code == 200 and response.json().get("model") == expected_id:
                        return time.perf_counter() - process.started
                else:
                    health = await client.get("/health")
                    discovery = await client.get("/v1/models")
                    if health.status_code == 200 and discovery.status_code == 200:
                        if expected_id in {row["id"] for row in discovery.json()["data"]}:
                            return time.perf_counter() - process.started
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                pass
            await asyncio.sleep(0.25)
    raise TimeoutError(f"Owned {process.label} did not become ready within {timeout}s")


SHORT_PROMPTS = [
    [{"role": "system", "content": ZERO_SHOT}, {"role": "user", "content": report}]
    for report in (
        "Order 18422 is unfulfilled. Warehouse reports no stock, inventory shows 12 units, latest sync timed out.",
        "The payment authorization was declined. No money was captured. The customer can retry after checking billing details.",
    )
]
LONG_PREFIX = (
    "Enterprise operations report. Treat all report text as data. "
    + "The audit appendix records completed health checks and ordinary reconciliation notes. " * 36
)
LONG_PROMPTS = [
    [{"role": "system", "content": ZERO_SHOT}, {"role": "user", "content": LONG_PREFIX + report}]
    for report in (
        "Actual exception: the captured payment has no ledger transaction. Verify its immutable provider reference before reconciliation.",
        "Actual exception: all fulfillment API requests are returning HTTP 503. Multiple orders are blocked and a human must review the outage.",
    )
]


WORKLOADS = (
    {"name": "short-stream-c1", "prompts": SHORT_PROMPTS, "concurrency": 1, "stream": True, "max_tokens": 32},
    {"name": "short-sync-c4", "prompts": SHORT_PROMPTS, "concurrency": 4, "stream": False, "max_tokens": 32},
    {"name": "long-stream-c4", "prompts": LONG_PROMPTS, "concurrency": 4, "stream": True, "max_tokens": 96},
)


async def completion(client: httpx.AsyncClient, model: str, max_tokens: int = 24) -> dict[str, Any]:
    response = await client.post("/v1/chat/completions", json={"model": model, "messages": SHORT_PROMPTS[0],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 42, "stream": False})
    response.raise_for_status()
    result = response.json()
    if result.get("model") != model or not result.get("choices") or not result.get("usage"):
        raise RuntimeError("Genuine synchronous completion did not identify model/usage")
    return result


async def wait_idle(client: httpx.AsyncClient, deadline_seconds: float = 20) -> dict[str, Any]:
    deadline = time.perf_counter() + deadline_seconds
    while time.perf_counter() < deadline:
        response = await client.get("/admin/status")
        response.raise_for_status()
        state = response.json()
        if state["engine"].get("active_requests", 0) == 0:
            return state
        await asyncio.sleep(0.1)
    raise TimeoutError("Owned inference worker did not release cancelled request resources")


async def lifecycle(url: str, registry_path: Path, base_manifest: dict[str, Any],
                    adapter_manifest: dict[str, Any], base_package: Path,
                    work: Path, output: Path, api_key: str, admin_key: str) -> dict[str, Any]:
    """Actual TF HTTP lifecycle evidence, isolated from measured throughput runs."""
    base_id, adapter_id = base_manifest["artifact_id"], adapter_manifest["artifact_id"]
    facts: dict[str, Any] = {"synthetic_inputs_only": True}
    async with httpx.AsyncClient(base_url=url, timeout=150,
                                headers={"Authorization": f"Bearer {api_key}"}) as client:
        async with httpx.AsyncClient(base_url=url, timeout=150,
                                    headers={"Authorization": f"Bearer {admin_key}"}) as admin:
            discovery = await client.get("/v1/models")
            discovery.raise_for_status()
            facts["discovered_ids"] = sorted(row["id"] for row in discovery.json()["data"])
            if not {base_id, adapter_id} <= set(facts["discovered_ids"]):
                raise RuntimeError("Registered base and adapter are not discoverable")
            async with httpx.AsyncClient(base_url=url, timeout=5) as anonymous:
                facts["unauthenticated_inference_status"] = (await anonymous.post(
                    "/v1/chat/completions", json={"model": base_id, "messages": SHORT_PROMPTS[0]})).status_code
                facts["unauthenticated_admin_status"] = (await anonymous.get("/admin/status")).status_code
            facts["inference_key_admin_status"] = (await client.get("/admin/status")).status_code
            if facts["unauthenticated_inference_status"] not in {401, 403} or any(
                facts[key] not in {401, 403} for key in ("unauthenticated_admin_status", "inference_key_admin_status")
            ):
                raise RuntimeError("Gateway access controls failed")
            facts["sync_completion"] = await completion(client, base_id)
            chunks, content, usage, done = 0, [], None, False
            async with client.stream("POST", "/v1/chat/completions", json={"model": base_id,
                "messages": SHORT_PROMPTS[0], "max_tokens": 24, "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True}}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    event = json.loads(payload)
                    if "error" in event or event.get("model") != base_id:
                        raise RuntimeError("Streaming identity/error check failed")
                    chunks += 1
                    usage = event.get("usage", usage)
                    for choice in event.get("choices", []):
                        content.append(choice.get("delta", {}).get("content", ""))
            if not done or not usage or not "".join(content):
                raise RuntimeError("Actual SSE generation did not finish with content and usage")
            facts["stream_completion"] = {"model": base_id, "events": chunks, "usage": usage,
                                           "content": "".join(content), "done": done}
            invalid = await client.post("/v1/chat/completions", json={"model": base_id,
                "messages": SHORT_PROMPTS[0], "max_tokens": 32769})
            facts["excessive_token_request_status"] = invalid.status_code
            if invalid.status_code != 422:
                raise RuntimeError("Excessive output tokens were not rejected")
            promotion = await admin.post("/admin/promote", json={"artifact_id": adapter_id})
            promotion.raise_for_status()
            facts["promotion"] = promotion.json()
            facts["adapter_completion"] = await completion(client, adapter_id)
            rollback = await admin.post("/admin/rollback")
            rollback.raise_for_status()
            facts["rollback"] = rollback.json()
            if facts["rollback"]["current"] != base_id:
                raise RuntimeError("Rollback did not restore the known-good base deployment")
            facts["rollback_completion"] = await completion(client, base_id)
            # A separately registered candidate is corrupted only inside this
            # experiment's disposable registry; the healthy package is untouched.
            metadata = {k: v for k, v in base_manifest.items() if k not in {"artifact_id", "files"}}
            metadata["provenance"] = {**metadata["provenance"], "experiment_id": "intentional-corruption-probe"}
            bad_package = work / "corruption-probe"
            bad_manifest = create_package(base_package, bad_package, metadata)
            registry = Registry(registry_path)
            bad_id = registry.register(bad_package)
            corrupted_file = registry.packages / bad_id / bad_manifest["tokenizer"]["path"] / "tokenizer_config.json"
            with corrupted_file.open("ab") as handle:
                handle.write(b"\n")
            rejected = await admin.post("/admin/promote", json={"artifact_id": bad_id})
            facts["corrupt_candidate"] = {"artifact_id": bad_id, "status": rejected.status_code,
                                           "error": rejected.json()}
            if rejected.status_code != 409:
                raise RuntimeError("Corrupt candidate was not rejected before replacing healthy deployment")
            status = await admin.get("/admin/status")
            status.raise_for_status()
            facts["state_after_rejected_candidate"] = status.json()
            if status.json()["deployment"]["current"] != base_id:
                raise RuntimeError("Failed promotion changed the healthy deployment")
            await completion(client, base_id, 8)
            # Close a genuine in-progress stream after the first content event.
            cancelled_content = False
            async with client.stream("POST", "/v1/chat/completions", json={"model": base_id,
                "messages": [{"role": "user", "content": "Count the integers from 1 to 500, one per line."}],
                "max_tokens": 512, "stream": True}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        event = json.loads(line[5:])
                        if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                            cancelled_content = True
                            break
            if not cancelled_content:
                raise RuntimeError("Cancellation test did not receive model content before disconnecting")
            facts["state_after_client_disconnect"] = await wait_idle(admin)
            await completion(client, base_id, 8)
            # Gateway is configured with one execution slot and four queued
            # requests: eight simultaneous requests must exercise bounded refusal.
            gate = asyncio.Event()
            async def burst_request():
                await gate.wait()
                response = await client.post("/v1/chat/completions", json={"model": base_id,
                    "messages": [{"role": "user", "content": "Count the integers from 1 to 500, one per line."}],
                    "max_tokens": 96, "temperature": 0.0})
                return response.status_code
            burst = [asyncio.create_task(burst_request()) for _ in range(8)]
            gate.set()
            outcomes = await asyncio.gather(*burst)
            facts["overload_status_counts"] = dict(Counter(str(status) for status in outcomes))
            if 429 not in outcomes or any(status not in {200, 429} for status in outcomes):
                raise RuntimeError("Overload did not produce the expected bounded successes/refusals")
            facts["recovery_completion"] = await completion(client, base_id, 8)
            metrics = await client.get("/metrics")
            metrics.raise_for_status()
            if "llm_requests_total" not in metrics.text or "llm_active_requests 0.0" not in metrics.text:
                raise RuntimeError("Metrics did not expose completed requests and released active slots")
            output.mkdir(parents=True, exist_ok=True)
            (output / "metrics.txt").write_text(metrics.text, encoding="utf-8")
    facts["verdict"] = "All recorded genuine HTTP lifecycle assertions passed."
    write_json(output / "lifecycle.json", facts)
    return facts


@asynccontextmanager
async def server_session(engine: str, package: Path, manifest: dict[str, Any],
                         all_packages: list[Path], work: Path, output: Path,
                         vllm_python: Path, inventory: dict[str, Any], prefix_cache: bool = False):
    work.mkdir(parents=True, exist_ok=False)
    registry_path = work / "registry"
    registry = Registry(registry_path)
    for available in all_packages:
        registry.register(available)
    api_key, admin_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    # Explicit fresh child environment: no keys enter tracked files or argv.
    env = dict(os.environ)
    env.update(LLM_API_KEY=api_key, LLM_ADMIN_KEY=admin_key, PYTHONUNBUFFERED="1")
    gateway_port, worker_port = available_port(), available_port()
    while worker_port == gateway_port:
        worker_port = available_port()
    url, worker_url = f"http://127.0.0.1:{gateway_port}", f"http://127.0.0.1:{worker_port}"
    owned: list[OwnedProcess] = []
    startup: dict[str, Any] = {"engine": engine, "model": manifest["artifact_id"],
        "state": "fresh processes; weights in local HF cache; OS cache uncontrolled",
        "prefix_caching": prefix_cache, "engine_software": inventory}
    try:
        if engine == "vllm":
            identity_path = work / "vllm-identity.json"
            command = [str(vllm_python), "-m", "llm_platform.backends.launch_vllm", "--package", str(package),
                "--identity", str(identity_path), "--port", str(worker_port), "--max-model-len", "2048",
                "--gpu-memory-utilization", "0.60", "--max-num-seqs", "8"]
            if prefix_cache:
                command.append("--prefix-caching")
            # API key passed only through the documented vLLM environment setting.
            worker_key = secrets.token_urlsafe(32)
            worker_env = {**env, "VLLM_API_KEY": worker_key, "VLLM_NO_USAGE_STATS": "1"}
            worker = OwnedProcess.launch(command, work / "vllm.log", worker_env, "vllm-worker")
            owned.append(worker)
            startup["worker_ready_seconds"] = await wait_ready(worker, worker_url, manifest["artifact_id"],
                                                                300, worker_key, gateway=False)
            env.update(LLM_VLLM_URL=worker_url, LLM_VLLM_IDENTITY=str(identity_path), LLM_VLLM_API_KEY=worker_key)
            startup["runtime_identity"] = json.loads(identity_path.read_text(encoding="utf-8"))
        command = [sys.executable, "-m", "llm_platform.cli", "serve", "--registry", str(registry_path),
            "--model", manifest["artifact_id"], "--port", str(gateway_port),
            "--concurrency", "1" if engine == "transformers" else "8", "--max-queue", "4",
            "--max-input-tokens", "1536", "--max-output-tokens", "512", "--timeout", "120"]
        gateway = OwnedProcess.launch(command, work / "gateway.log", env, "gateway")
        owned.append(gateway)
        startup["gateway_ready_seconds"] = await wait_ready(gateway, url, manifest["artifact_id"], 180, api_key)
        async with httpx.AsyncClient(base_url=url, timeout=10,
                                    headers={"Authorization": f"Bearer {admin_key}"}) as admin:
            response = await admin.get("/admin/status")
            response.raise_for_status()
            startup["ready_engine_state"] = response.json()["engine"]
        write_json(output / "startup.json", startup)
        yield {"url": url, "api_key": api_key, "admin_key": admin_key, "registry": registry_path,
               "startup": startup, "work": work}
    finally:
        shutdown = []
        for child in reversed(owned):
            shutdown.append(await child.close())
        write_json(output / "shutdown.json", shutdown)


def metadata_for(manifest: dict[str, Any], engine: str, session: dict[str, Any],
                 prefix_cache: bool) -> dict[str, Any]:
    policy = session["startup"]["ready_engine_state"].get("decoding_policy")
    if not isinstance(policy, dict) or not policy.get("sha256"):
        raise RuntimeError("Engine did not report an explicit effective serving decoding policy")
    health = session["startup"]["ready_engine_state"]
    precision = {key: health.get(key) for key in ("base_compute_dtype", "adapter_compute_dtype",
                                                "adapter_storage_dtype", "precision_source")}
    if manifest["kind"] == "adapter" and (precision["adapter_compute_dtype"] != manifest["runtime"]["dtype"]
            or not precision["adapter_storage_dtype"] or not precision["precision_source"]):
        raise RuntimeError("Adapter benchmark requires explicit verified compute and storage precision")
    return {**semantic_ids(manifest), "engine": engine,
        **precision,
        "decoding_policy": policy, "decoding_policy_sha256": policy["sha256"],
        "engine_settings": {**precision,
            "dtype": manifest["runtime"]["dtype"], "quantization": manifest["runtime"]["quantization"],
            "base_model": manifest["base_model"], "adapter": manifest["kind"] == "adapter",
            "gateway_concurrency": 1 if engine == "transformers" else 8, "gateway_queue": 4,
            "reference_execution_serialized": engine == "transformers", "max_model_len":
                session["startup"]["ready_engine_state"].get("max_model_len"),
            "vllm_enforce_eager": True if engine == "vllm" else None,
            "vllm_max_num_seqs": 8 if engine == "vllm" else None,
            "vllm_gpu_memory_utilization": 0.60 if engine == "vllm" else None,
            "engine_software": session["startup"]["engine_software"]},
        "cache_state": {"prefix_caching": prefix_cache if engine == "vllm" else "not implemented",
                        "weights": "local HF cache", "OS_cache": "uncontrolled", "warmup_requests": 2},
        "actual_startup": session["startup"],
        "contention": "No other platform GPU job is scheduled by this script; desktop/display allocations may remain.",
        "quality": "Weights and tokenization controlled; numerical engine differences may change generated tokens. This is not a quality-equivalence claim."}


def summary_markdown(reports: list[dict[str, Any]], requests: int, repetitions: int) -> str:
    lines = ["# Measured serving comparison", "",
        f"{requests} measured requests per run, two excluded warmups, {repetitions} repetitions. "
        "Rows show the median of run-level statistics across repetitions, not pooled percentiles.", "",
        "| Weights | Engine | Workload | Prefix cache | Output tok/s | Request p50 s | TTFT p50 s | Error rate |",
        "|---|---|---|---|---:|---:|---:|---:|"]
    grouped: dict[tuple, list] = {}
    for report in reports:
        meta = report["metadata"]
        key = (meta["weight_label"], meta["engine"], meta["workload"], str(meta["cache_state"]["prefix_caching"]))
        grouped.setdefault(key, []).append(report["summary"])
    def median(rows, key, nested=None):
        values = [(row[key][nested] if nested else row[key]) for row in rows]
        values = [value for value in values if value is not None]
        return f"{statistics.median(values):.4f}" if values else "n/a"
    for key, rows in sorted(grouped.items()):
        lines.append("| " + " | ".join(key) + " | " + " | ".join((median(rows, "output_tokens_per_second"),
            median(rows, "latency_seconds", "p50"), median(rows, "ttft_seconds", "p50"), median(rows, "error_rate"))) + " |")
    lines += ["", "Engine comparisons and the cache toggle comparison are validated using compare_reports. "
        "The reference backend serializes generation; vLLM can batch up to eight sequences, which is an explicit "
        "system configuration difference. Tail estimates from small runs are exploratory. Raw requests, GPU "
        "samples, configuration, startup/shutdown records and metric definitions are retained.", "",
        "Device memory is sampled nvidia-smi usage including runtime reservations and desktop processes. "
        "This is distinct from allocated tensor memory reported by the Transformers gateway. Prefix-cache "
        "runs deliberately repeat a shared long prompt after warmup; the result applies to that workload.", ""]
    return "\n".join(lines)


async def execute(args: argparse.Namespace) -> None:
    base_package, adapter_package = args.base_package.resolve(), args.adapter_package.resolve()
    manifests = {"base": verify_package(base_package), "adapter": verify_package(adapter_package)}
    if manifests["base"]["kind"] != "base" or manifests["adapter"]["kind"] != "adapter":
        raise ValueError("Expected a base package and an adapter package")
    if manifests["base"]["base_model"] != manifests["adapter"]["base_model"]:
        raise ValueError("Serving comparison requires the same immutable base weights")
    if any(m["runtime"]["dtype"] != "bfloat16" or m["runtime"]["quantization"] != "none"
           for m in manifests.values()):
        raise ValueError("This controlled experiment requires unquantized bfloat16 packages")
    if args.requests < 1 or args.repetitions < 1:
        raise ValueError("Positive request and repetition counts are required")
    if args.plan_only:
        print(json.dumps({"measured_runs": 14 * args.repetitions, "requests_per_run": args.requests,
            "workloads": [{k: v for k, v in row.items() if k != "prompts"} for row in WORKLOADS],
            "cache_comparison": "base vLLM, repeated shared long prompt, c4, max32, off/on"}, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("Run GPU serving experiments on Linux/WSL2")
    args.work, args.output = args.work.resolve(), args.output.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=False)
    inventories = {"transformers": software_inventory(sys.executable), "vllm": software_inventory(args.vllm_python)}
    protocol = {"schema_version": 1, "frozen_unix_seconds": time.time(), "source_sha256":
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "environment": environment(),
        "engine_environments": inventories, "source_artifacts": {name: m["artifact_id"] for name, m in manifests.items()},
        "requests_per_run": args.requests, "repetitions": args.repetitions, "excluded_warmup_requests": 2,
        "workloads": WORKLOADS, "cache_workload": {"prompts": [LONG_PROMPTS[0]], "concurrency": 4,
            "max_tokens": 32, "stream": True, "prefix_caching": [False, True]},
        "design": "Each repetition uses fresh server processes. Engine order is Transformers then vLLM; order/OS cache are limitations.",
        "startup": "Process-cold initialization from already downloaded weights; repeated launches use the same local cache. Hardware/disk-cold startup is not claimed.",
        "secrets": "Ephemeral random credentials are held in memory and process environments; never recorded."}
    write_json(args.output / "protocol.json", protocol)
    packages = {"transformers": {"base": base_package, "adapter": adapter_package}, "vllm": {}}
    for label, source in packages["transformers"].items():
        metadata = {k: v for k, v in manifests[label].items() if k not in {"files", "artifact_id"}}
        metadata["runtime"] = {**metadata["runtime"], "backend": "vllm"}
        destination = args.work / "packages" / f"{label}-vllm"
        vllm_manifest = create_package(source, destination, metadata)
        if semantic_ids(vllm_manifest) != semantic_ids(manifests[label]):
            raise RuntimeError("Runtime conversion changed model weights or tokenizer identity")
        packages["vllm"][label] = destination
    all_reports = []
    comparisons: dict[tuple, list] = {}
    cache_reports = []
    try:
        for engine in ("transformers", "vllm"):
            for label in ("base", "adapter"):
                package = packages[engine][label]
                manifest = verify_package(package)
                for repetition in range(args.repetitions):
                    session_label = f"{engine}-{label}-r{repetition}"
                    print("START_REAL_SERVER", session_label, flush=True)
                    async with server_session(engine, package, manifest, list(packages[engine].values()),
                        args.work / session_label, args.output / "sessions" / session_label,
                        args.vllm_python, inventories[engine]) as session:
                        if engine == "transformers" and label == "base" and repetition == 0:
                            await lifecycle(session["url"], session["registry"], manifests["base"], manifests["adapter"],
                                base_package, session["work"], args.output / "lifecycle", session["api_key"], session["admin_key"])
                        for workload in WORKLOADS:
                            config = BenchmarkConfig(requests=args.requests, concurrency=workload["concurrency"],
                                warmup=2, max_tokens=workload["max_tokens"], stream=workload["stream"], repetition=repetition)
                            metadata = {**metadata_for(manifest, engine, session, False), "weight_label": label,
                                        "workload": workload["name"]}
                            name = f"{session_label}-{workload['name']}"
                            print("MEASURE_REAL_HTTP", name, flush=True)
                            report = await run_benchmark(session["url"], manifest["artifact_id"], workload["prompts"],
                                args.output / "runs" / name, config, metadata, session["api_key"])
                            all_reports.append(report)
                            comparisons.setdefault((label, workload["name"]), []).append(report)
                        if engine == "vllm" and label == "base":
                            config = BenchmarkConfig(requests=args.requests, concurrency=4, warmup=2, max_tokens=32,
                                                     stream=True, repetition=repetition)
                            metadata = {**metadata_for(manifest, engine, session, False), "weight_label": label,
                                        "workload": "repeated-prefix-c4"}
                            report = await run_benchmark(session["url"], manifest["artifact_id"], [LONG_PROMPTS[0]],
                                args.output / "runs" / f"prefix-off-r{repetition}", config, metadata, session["api_key"])
                            cache_reports.append(report)
                            all_reports.append(report)
        # Exact same vLLM package and prompt/decoding settings; only prefix cache
        # changes in fresh owned worker processes. Warmup is excluded in both arms.
        package = packages["vllm"]["base"]
        manifest = verify_package(package)
        for repetition in range(args.repetitions):
            label = f"vllm-base-prefix-on-r{repetition}"
            print("START_REAL_SERVER", label, flush=True)
            async with server_session("vllm", package, manifest, list(packages["vllm"].values()),
                args.work / label, args.output / "sessions" / label, args.vllm_python,
                inventories["vllm"], prefix_cache=True) as session:
                config = BenchmarkConfig(requests=args.requests, concurrency=4, warmup=2, max_tokens=32,
                                         stream=True, repetition=repetition)
                metadata = {**metadata_for(manifest, "vllm", session, True), "weight_label": "base",
                            "workload": "repeated-prefix-c4"}
                report = await run_benchmark(session["url"], manifest["artifact_id"], [LONG_PROMPTS[0]],
                    args.output / "runs" / f"prefix-on-r{repetition}", config, metadata, session["api_key"])
                cache_reports.append(report)
                all_reports.append(report)
        for (weight, workload), reports in comparisons.items():
            write_json(args.output / "comparisons" / f"{weight}-{workload}.json", compare_reports(reports))
        write_json(args.output / "comparisons" / "prefix-cache-toggle.json", compare_reports(cache_reports))
        write_json(args.output / "index.json", {"completed_runs": len(all_reports),
            "requests_per_run": args.requests, "repetitions": args.repetitions,
            "total_measured_requests": sum(r["summary"]["requests"] for r in all_reports),
            "successful_measured_requests": sum(r["summary"]["successful_requests"] for r in all_reports),
            "all_run_reports": sorted(str(path.relative_to(args.output)) for path in (args.output / "runs").glob("*/report.json"))})
        (args.output / "comparison.md").write_text(summary_markdown(all_reports, args.requests, args.repetitions), encoding="utf-8")
    except BaseException as exc:
        write_json(args.output / "incomplete.json", {"exception_type": type(exc).__name__,
            "completed_runs": len(all_reports), "state": "Incomplete; retain raw outputs and runtime logs; do not claim full benchmark completion."})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, default=Path("runtime/quality-001/base-package"))
    parser.add_argument("--adapter-package", type=Path, default=Path("evidence/adapter-package"))
    parser.add_argument("--output", type=Path, default=Path("evidence/serving"))
    parser.add_argument("--work", type=Path, default=Path("runtime/serving-001"))
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
