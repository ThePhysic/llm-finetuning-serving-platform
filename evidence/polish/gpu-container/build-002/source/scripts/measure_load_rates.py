"""Measure bounded client pacing at several request rates on two real engines.

Run after other GPU experiments finish. Importing or --plan-only never starts a
model server. This reuses the primary experiment's owned process lifecycle and
ephemeral environment-only credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import httpx

from llm_platform.benchmark import BenchmarkConfig, compare_reports, environment, run_benchmark
from llm_platform.registry import verify_package

from run_serving_experiments import (
    SHORT_PROMPTS,
    metadata_for,
    semantic_ids,
    server_session,
    software_inventory,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_identity() -> dict[str, Any]:
    """Record the actual experiment code, including edits after a source commit."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    files = [Path(__file__).resolve(), PROJECT_ROOT / "scripts" / "run_serving_experiments.py",
             PROJECT_ROOT / "pyproject.toml", *sorted((PROJECT_ROOT / "src" / "llm_platform").rglob("*.py"))]
    return {"code_revision": revision, "code_dirty": dirty,
            "source_files_sha256": {path.relative_to(PROJECT_ROOT).as_posix():
                                    hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def _metric_value(text: str, name: str) -> float | None:
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == name:
            try:
                value = float(fields[1])
            except ValueError:
                return None
            return value if math.isfinite(value) else None
    return None


async def queue_snapshot(session: dict[str, Any]) -> dict[str, Any]:
    """Observe cumulative queue metrics; failures stay explicitly unavailable."""
    observation = {"observed_at_unix_seconds": time.time(), "queue_seconds_sum": None,
                   "queue_seconds_count": None, "active": None, "queued": None,
                   "error": None, "prometheus_text": None}
    try:
        async with httpx.AsyncClient(base_url=session["url"], timeout=5,
                                    headers={"Authorization": f"Bearer {session['api_key']}"}) as client:
            response = await client.get("/metrics")
            response.raise_for_status()
            text = response.text
        observation.update(queue_seconds_sum=_metric_value(text, "llm_queue_seconds_sum"),
                           queue_seconds_count=_metric_value(text, "llm_queue_seconds_count"),
                           active=_metric_value(text, "llm_active_requests"),
                           queued=_metric_value(text, "llm_queued_requests"), prometheus_text=text)
    except (httpx.HTTPError, ValueError) as exc:
        observation["error"] = type(exc).__name__
    return observation


def pacing_observations(rows: list[dict[str, Any]], rate: float,
                        before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delays = sorted(row["client_schedule_delay_seconds"] for row in rows)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in delays):
        raise ValueError("Recorded scheduling delays must be finite and nonnegative")
    # Worker dispatch is scheduled at start + index/rate. The raw delay permits
    # reconstructing relative dispatch times without pretending every scheduled
    # arrival reached the server immediately.
    dispatch = [row["index"] / rate + row["client_schedule_delay_seconds"] for row in rows]
    span = max(dispatch) - min(dispatch) if len(dispatch) > 1 else 0
    queue_sum, queue_count = None, None
    if all(before.get(key) is not None and after.get(key) is not None
           for key in ("queue_seconds_sum", "queue_seconds_count")):
        delta_sum = after["queue_seconds_sum"] - before["queue_seconds_sum"]
        delta_count = after["queue_seconds_count"] - before["queue_seconds_count"]
        if delta_sum >= 0 and delta_count >= 0:
            queue_sum, queue_count = delta_sum, delta_count
    return {"configured_request_rate": rate, "dispatch_span_requests_per_second":
                (len(dispatch) - 1) / span if span > 0 else None,
            "client_schedule_delay_seconds": {"median": statistics.median(delays) if delays else None,
                                               "maximum": max(delays) if delays else None,
                                               "total": sum(delays)},
            "gateway_queue_seconds_sum_including_warmup": queue_sum,
            "gateway_queue_samples_including_warmup": queue_count,
            "gateway_queue_mean_seconds_including_warmup": queue_sum / queue_count
                if queue_sum is not None and queue_count else None,
            "queue_observation": "Cumulative gateway queue histogram deltas span this arm including its two warmup requests; not per-example queue timings.",
            "arrival_model": "Paced bounded closed-loop workers. Client scheduling delay is outside HTTP request latency/TTFT; gateway queue delay is inside them. No independent open-loop offered-load claim."}


def validate_args(args: argparse.Namespace) -> None:
    if type(args.requests) is not int or args.requests < 1:
        raise ValueError("Measured request count must be positive")
    if len(args.rates) < 2 or len(set(args.rates)) != len(args.rates):
        raise ValueError("Provide at least two distinct positive request rates")
    if any(not math.isfinite(rate) or rate <= 0 for rate in args.rates):
        raise ValueError("Request rates must be finite and positive")
    work, output = args.work.resolve(), args.output.resolve()
    if work == output or work in output.parents or output in work.parents:
        raise ValueError("Runtime work and evidence directories must be disjoint")
    if work.exists() or output.exists():
        raise FileExistsError("Use new work and output directories; previous experiment evidence is immutable")


def render_summary(reports: list[dict[str, Any]]) -> str:
    lines = ["# Measured bounded request-rate comparison", "",
             "Each row is one measured arm, not a repeated estimate or a stable tail-latency result. "
             "The target rate controls client pacing; it does not establish an achieved open-loop offered rate.", "",
             "| Engine | Target pace req/s | Achieved completed req/s | Dispatch-span req/s | Client delay median/max s | Gateway queue mean s (including warmup) | Error rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]

    def number(value):
        return f"{value:.5f}" if value is not None else "unavailable"

    for report in reports:
        pacing = report["pacing_observations"]
        delay = pacing["client_schedule_delay_seconds"]
        lines.append("| " + " | ".join((report["metadata"]["engine"], number(report["config"]["request_rate"]),
            number(report["summary"]["request_per_second"]), number(pacing["dispatch_span_requests_per_second"]),
            number(delay["median"]) + " / " + number(delay["maximum"]),
            number(pacing["gateway_queue_mean_seconds_including_warmup"]), number(report["summary"]["error_rate"]))) + " |")
    lines += ["", "Same-rate cross-engine pairs pass compare_reports, including actual weight/tokenizer and effective decoding policy identities. "
              "The target rate intentionally differs between rate-axis rows; those rows are not submitted as an identical-workload engine comparison.", "",
              "The client has four workers. Transformers gateway execution is serialized, while vLLM permits eight gateway slots and engine batching. "
              "Both use the same short prompts, streaming, greedy settings, 32-token cap, 12 measured requests by default, and two excluded warmups. "
              "No rejection is required: a high requested pace can become client delay rather than an HTTP overload response.", "",
              "One fresh session per engine runs rate arms in the listed order. Warm model state, OS cache, thermal drift, engine order, small samples, "
              "and the absence of repetitions limit inference. Queue histogram deltas include warmup; raw per-request scheduling delays and request/TTFT "
              "measurements remain separate. GPU/session-peak sampling has the same definitions and limitations as the primary benchmark suite.", ""]
    return "\n".join(lines)


async def execute(args: argparse.Namespace) -> None:
    validate_args(args)
    if args.plan_only:
        print(json.dumps({"state": "plan only; no model or HTTP process launched", "engines": ["transformers", "vllm"],
            "request_rates": args.rates, "measured_runs": 2 * len(args.rates), "requests_per_run": args.requests,
            "warmup_per_run": 2, "client_concurrency": 4, "stream": True, "max_tokens": 32,
            "arrival_model": "bounded closed-loop workers with paced dispatch and recorded client scheduling delay"}, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("Real model serving experiments must run on Linux/WSL2")
    packages = {"transformers": args.base_package.resolve(), "vllm": args.vllm_package.resolve()}
    manifests = {engine: verify_package(package) for engine, package in packages.items()}
    if any(manifest["kind"] != "base" or manifest["runtime"]["backend"] != engine
           or manifest["runtime"]["dtype"] != "bfloat16" or manifest["runtime"]["quantization"] != "none"
           for engine, manifest in manifests.items()):
        raise ValueError("Rate comparison requires correctly bound unquantized bfloat16 base packages")
    if semantic_ids(manifests["transformers"]) != semantic_ids(manifests["vllm"]):
        raise ValueError("Rate comparison packages differ in actual weights or tokenizer identity")
    inventories = {"transformers": software_inventory(sys.executable), "vllm": software_inventory(args.vllm_python)}
    args.work, args.output = args.work.resolve(), args.output.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"schema_version": 1, "frozen_at_unix_seconds": time.time(), "source": source_identity(),
                "environment": environment(), "engine_environments": inventories,
                "packages": {engine: {"artifact_id": manifest["artifact_id"], **semantic_ids(manifest)}
                             for engine, manifest in manifests.items()},
                "rates_in_order": args.rates, "requests_per_arm": args.requests,
                "warmup_per_arm": 2, "client_concurrency": 4, "stream": True, "max_tokens": 32,
                "prompts": SHORT_PROMPTS,
                "protocol": "One fresh owned session per engine; Transformers then vLLM; rates run in listed order with prefix caching disabled. Wait for other GPU experiments to finish before launching.",
                "arrival_model": "Paced bounded closed-loop; late arrivals wait at client and are recorded. Not independent open-loop offered load.",
                "credentials": "Ephemeral environment-only keys are generated by server_session and never written to protocol, evidence, or command lines.",
                "limitations": ["One arm per engine/rate, no repeated estimates or stable tail-latency claim.",
                                "Server queue histogram deltas include the two warmup requests.",
                                "The listed target rate need not be achieved and need not cause rejection."]}
    write_json(args.output / "protocol.json", protocol)
    reports: list[dict[str, Any]] = []
    by_rate: dict[float, list[dict[str, Any]]] = {rate: [] for rate in args.rates}
    try:
        for engine in ("transformers", "vllm"):
            package, manifest = packages[engine], manifests[engine]
            print("START_REAL_RATE_SESSION", engine, flush=True)
            async with server_session(engine, package, manifest, [package], args.work / engine,
                                      args.output / "sessions" / engine, args.vllm_python,
                                      inventories[engine], prefix_cache=False) as session:
                for index, rate in enumerate(args.rates):
                    name = f"{engine}-rate-{index:02d}-{rate:g}"
                    output = args.output / "runs" / name
                    config = BenchmarkConfig(requests=args.requests, concurrency=4, warmup=2,
                                             max_tokens=32, stream=True, request_rate=rate)
                    metadata = {**metadata_for(manifest, engine, session, False), "weight_label": "base",
                                "workload": "short-stream-c4-rate-sweep", "configured_request_rate": rate}
                    before = await queue_snapshot(session)
                    print("MEASURE_REAL_HTTP_RATE", engine, rate, flush=True)
                    report = await run_benchmark(session["url"], manifest["artifact_id"], SHORT_PROMPTS,
                                                 output, config, metadata, session["api_key"])
                    after = await queue_snapshot(session)
                    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line]
                    report["pacing_observations"] = pacing_observations(rows, rate, before, after)
                    write_json(output / "queue-before.json", before)
                    write_json(output / "queue-after.json", after)
                    write_json(output / "report.json", report)
                    reports.append(report)
                    by_rate[rate].append(report)
        for index, rate in enumerate(args.rates):
            write_json(args.output / "comparisons" / f"rate-{index:02d}-{rate:g}.json", compare_reports(by_rate[rate]))
        write_json(args.output / "index.json", {"schema_version": 1, "completed_runs": len(reports),
            "expected_runs": 2 * len(args.rates), "rates": args.rates, "requests_per_arm": args.requests,
            "total_measured_requests": sum(report["summary"]["requests"] for report in reports),
            "successful_measured_requests": sum(report["summary"]["successful_requests"] for report in reports),
            "run_reports": sorted(str(path.relative_to(args.output)) for path in (args.output / "runs").glob("*/report.json")),
            "status": "complete measurement, including honestly recorded error/rejection outcomes"})
        (args.output / "comparison.md").write_text(render_summary(reports), encoding="utf-8")
    except BaseException as exc:
        write_json(args.output / "incomplete.json", {"exception_type": type(exc).__name__, "completed_runs": len(reports),
                   "state": "Incomplete; preserve protocol, raw observations and owned runtime logs. Do not claim a completed rate sweep."})
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-package", type=Path, default=Path("runtime/quality-002/base-package"))
    result.add_argument("--vllm-package", type=Path, default=Path("runtime/serving-001/packages/base-vllm"))
    result.add_argument("--vllm-python", type=Path, required=True)
    result.add_argument("--work", type=Path, default=Path("runtime/load-rates-001"))
    result.add_argument("--output", type=Path, default=Path("evidence/load-rates"))
    result.add_argument("--requests", type=int, default=12)
    result.add_argument("--rates", type=float, nargs="+", default=[1.0, 32.0])
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    asyncio.run(execute(parser().parse_args()))


if __name__ == "__main__":
    main()
