"""Measure paired first-load and same-process warm-reload CUDA startup.

Each repetition starts a fresh owned Python process. Cached model weights and OS
disk cache are explicitly not treated as a hardware/disk-cold experiment.
Importing this module, --help and --plan-only never construct a GPU model.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any

from llm_platform.backends import TransformersBackend
from llm_platform.benchmark import environment
from llm_platform.contracts import GenerationConfig
from llm_platform.registry import verify_package


PROMPT = [{"role": "user", "content": "Reply with the single word READY."}]
GENERATION = GenerationConfig(max_tokens=8, temperature=0.0, seed=42)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    paths = [Path(__file__), *sorted((root / "src" / "llm_platform").rglob("*.py"))]
    return {"code_revision": revision, "code_dirty": dirty,
            "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths}}


def workload_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {"artifact_id": manifest["artifact_id"], "base_model": manifest["base_model"],
            "tokenizer": manifest["tokenizer"], "runtime": manifest["runtime"],
            "package_files": manifest["files"], "generation": json.loads(json.dumps(asdict(GENERATION))),
            "synthetic_prompt_sha256": hashlib.sha256(json.dumps(PROMPT, sort_keys=True).encode()).hexdigest()}


async def child_measure(args: argparse.Namespace) -> None:
    """This is the only path in the module that constructs a GPU backend."""
    manifest = verify_package(args.package)
    if manifest["artifact_id"] != args.expected_artifact:
        raise ValueError("Startup child package differs from the frozen parent protocol")
    measurements = []
    for index, label in enumerate(("first_load_fresh_process", "warm_reload_same_process")):
        backend = TransformersBackend(device="cuda")
        row: dict[str, Any] = {"condition": label, "sequence_index": index}
        try:
            began = time.perf_counter()
            await backend.load(args.package, manifest)
            # Imported for the first time only AFTER load() has imported it
            # internally. Synchronize so load timing includes completed CUDA work.
            import torch

            torch.cuda.synchronize()
            row["load_seconds"] = time.perf_counter() - began
            health = backend.health()
            if not health.get("ready") or not str(health.get("device")).startswith("cuda"):
                raise RuntimeError("Startup experiment requires an actually ready CUDA model")
            if backend.manifest is None or backend.manifest["artifact_id"] != manifest["artifact_id"]:
                raise RuntimeError("Loaded startup model identity mismatch")
            policy = health.get("decoding_policy")
            if not isinstance(policy, dict) or not policy.get("sha256"):
                raise RuntimeError("Effective startup decoding policy was not reported")
            row["served_artifact_id"] = backend.manifest["artifact_id"]
            row["decoding_policy"] = policy
            row["cuda_device"] = torch.cuda.get_device_name()
            row["cuda_runtime"] = torch.version.cuda
            row["health_after_load"] = health
            # A real short generation proves both loads are usable. It is
            # timed separately and never included in the load-only measurement.
            generated, count, finish = [], 0, None
            first_content = None
            began = time.perf_counter()
            iterator = backend.stream(PROMPT, GENERATION, f"startup-r{args.child_index}-load{index}")
            try:
                async for chunk in iterator:
                    count += chunk.token_count
                    if chunk.text:
                        if first_content is None:
                            first_content = time.perf_counter() - began
                        generated.append(chunk.text)
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
            finally:
                await iterator.aclose()
            torch.cuda.synchronize()
            text = "".join(generated)
            if not text or not 1 <= count <= GENERATION.max_tokens or finish not in {"stop", "length"}:
                raise RuntimeError("Startup generation did not return genuine bounded content and usage")
            row["generation"] = {"seconds": time.perf_counter() - began, "ttft_seconds": first_content,
                                 "completion_tokens": count, "finish_reason": finish,
                                 "content_sha256": hashlib.sha256(text.encode()).hexdigest()}
            row["health_before_unload"] = backend.health()
            row["cuda_allocated_before_unload_bytes"] = torch.cuda.memory_allocated()
            row["cuda_reserved_before_unload_bytes"] = torch.cuda.memory_reserved()
        finally:
            began = time.perf_counter()
            await backend.unload()
            row["unload_seconds"] = time.perf_counter() - began
        if backend.health().get("ready"):
            raise RuntimeError("Backend remained ready after startup unload")
        row["health_after_unload"] = backend.health()
        row["cuda_allocated_after_unload_bytes"] = torch.cuda.memory_allocated()
        row["cuda_reserved_after_unload_bytes"] = torch.cuda.memory_reserved()
        measurements.append(row)
    if measurements[0]["decoding_policy"] != measurements[1]["decoding_policy"]:
        raise RuntimeError("Warm reload changed the effective decoding policy")
    write_json(args.child_output, {"schema_version": 1, "repetition": args.child_index,
        "pid": os.getpid(), "workload_identity": workload_identity(manifest), "source": source_identity(),
        "environment": environment(), "measurements": measurements,
        "conditions": "Paired sequential loads in one fresh child. First includes cold Python heavy imports and CUDA context; second reuses imports/context. Both use local weights; OS cache uncontrolled."})


def settle_child(child: subprocess.Popen, grace_seconds: float = 15) -> dict[str, Any]:
    """Terminate only the dedicated process group created by this parent."""
    forced = False
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            forced = True
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
    return {"exit_code": child.returncode, "forced_shutdown": forced}


def parent_measure(args: argparse.Namespace) -> None:
    manifest = verify_package(args.package)
    if manifest["kind"] != "base" or manifest["runtime"]["backend"] != "transformers":
        raise ValueError("Paired startup measurements require a Transformers base-model package")
    if args.repetitions < 1 or not math.isfinite(args.child_timeout) or not 0 < args.child_timeout <= 3600:
        raise ValueError("Repetitions must be positive and child timeout finite within (0, 3600] seconds")
    identity = workload_identity(manifest)
    if args.plan_only:
        print(json.dumps({"gpu_launched": False, "fresh_children": args.repetitions,
            "sequential_loads_per_child": 2, "total_real_loads": 2 * args.repetitions,
            "conditions": ["first load with cold imports/CUDA context", "same-process warm reload"],
            "workload_identity": identity,
            "cache_limitations": "Weights already cached; OS cache uncontrolled; no disk/hardware-cold claim."}, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("Run the actual CUDA startup experiment on Linux/WSL2")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"schema_version": 1, "frozen_unix_seconds": time.time(), "source": source_identity(),
        "workload_identity": identity, "environment": environment(), "repetitions": args.repetitions,
        "child_timeout_seconds": args.child_timeout,
        "load_definition": "perf_counter around backend.load plus CUDA synchronize; excludes generation and unload",
        "conditions": {"first": "Fresh Python child; cold heavy imports and CUDA context initialization",
                       "warm": "Same child after successful generation/unload; warm Python imports and CUDA context"},
        "cache": "Weights in local HF cache for both conditions; OS file cache uncontrolled",
        "scheduling": "Sequential children; do not run concurrently with training/serving benchmarks",
        "privacy": "Only a synthetic readiness prompt hash and generated content hashes are recorded."}
    write_json(args.output / "protocol.json", protocol)
    children = []
    results = []
    try:
        for repetition in range(args.repetitions):
            child_output = args.output / f"repetition-{repetition}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--package", str(args.package),
                "--child-output", str(child_output), "--child-index", str(repetition),
                "--expected-artifact", manifest["artifact_id"]]
            print("MEASURE_REAL_CUDA_STARTUP", repetition, flush=True)
            child = None
            with (args.output / f"repetition-{repetition}.log").open("w", encoding="utf-8") as log:
                try:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                    child.wait(timeout=args.child_timeout)
                finally:
                    if child is not None:
                        outcome = settle_child(child)
                        children.append({"repetition": repetition, "pid": child.pid, **outcome})
            if child is None or child.returncode != 0:
                raise RuntimeError("Owned startup child failed; retain its runtime log and earlier observations")
            result = json.loads(child_output.read_text(encoding="utf-8"))
            if result["workload_identity"] != identity or result["repetition"] != repetition:
                raise RuntimeError("Startup child returned a different workload identity")
            results.append(result)
        policies = {row["decoding_policy"]["sha256"] for result in results for row in result["measurements"]}
        if len(policies) != 1:
            raise RuntimeError("Startup repetitions changed the effective decoding policy")
        first = [result["measurements"][0]["load_seconds"] for result in results]
        warm = [result["measurements"][1]["load_seconds"] for result in results]
        unloads = [row["unload_seconds"] for result in results for row in result["measurements"]]
        summary = {"schema_version": 1, "repetitions": args.repetitions, "workload_identity": identity,
            "decoding_policy_sha256": next(iter(policies)), "source": protocol["source"],
            "first_load_seconds": {"raw": first, "median": statistics.median(first), "min": min(first), "max": max(first)},
            "warm_reload_seconds": {"raw": warm, "median": statistics.median(warm), "min": min(warm), "max": max(warm)},
            "unload_seconds": {"raw": unloads, "median": statistics.median(unloads)},
            "paired_first_over_warm_ratios": [a / b for a, b in zip(first, warm, strict=True)],
            "owned_children": children,
            "limitations": ["Small paired sample; medians describe this machine/configuration only.",
                "First-load difference includes Python imports and CUDA context initialization, not just weight loading.",
                "Both conditions use local cached weights; OS cache is uncontrolled and disk-cold behavior is not measured.",
                "Warm reload follows a real generation and unload, so runtime/OS allocator history is part of its condition.",
                "This measures backend readiness and separate short generation, not HTTP gateway startup."]}
        write_json(args.output / "summary.json", summary)
        (args.output / "report.md").write_text(
            "# Measured paired backend startup\n\n"
            f"{args.repetitions} fresh Python children each loaded the same immutable base package twice. "
            "The second load followed real generation and unload in the same child.\n\n"
            f"First-load median: **{statistics.median(first):.4f} seconds**. "
            f"Same-process warm-reload median: **{statistics.median(warm):.4f} seconds**.\n\n"
            "First load includes cold heavy Python imports and CUDA context initialization. Both conditions use "
            "locally cached weights with uncontrolled OS cache. These observations do not establish disk-cold "
            "startup or isolate weight I/O. The small paired sample supports descriptive reporting, not stable "
            "tail estimates. Raw load/unload/generation timings, CUDA memory, policy identity, versions and "
            "source hashes are in the repetition files and summary.json.\n", encoding="utf-8")
    except BaseException as exc:
        write_json(args.output / "incomplete.json", {"exception_type": type(exc).__name__,
            "completed_repetitions": len(results), "owned_children": children,
            "state": "Incomplete: preserve observations and logs; do not claim a complete startup comparison."})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("runtime/quality-002/base-package"))
    parser.add_argument("--output", type=Path, default=Path("evidence/startup"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--child-timeout", type=float, default=300)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--expected-artifact", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.package, args.output = args.package.resolve(), args.output.resolve()
    if args.child_output is not None:
        if args.child_index is None or args.expected_artifact is None:
            raise ValueError("Child measurement requires its frozen artifact and repetition identity")
        asyncio.run(child_measure(args))
    else:
        parent_measure(args)


if __name__ == "__main__":
    main()
