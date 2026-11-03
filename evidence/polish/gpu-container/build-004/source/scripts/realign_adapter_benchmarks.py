"""Archive a completed matrix and rerun only adapter arms after a precision confound.

This is an explicit protocol amendment, not a claim that the corrected experiment
was frozen before the original measurements. Only root/orchestrator launches GPU
work, after other experiments stop; importing this module never starts a server.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
from typing import Any

from llm_platform.benchmark import BenchmarkConfig, compare_reports, environment, run_benchmark
from llm_platform.registry import verify_package

from run_serving_experiments import (
    WORKLOADS,
    metadata_for,
    semantic_ids,
    server_session,
    software_inventory,
    summary_markdown,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRECISION = {"base_compute_dtype": "bfloat16", "adapter_compute_dtype": "bfloat16",
             "adapter_storage_dtype": "float32"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Evidence must be an ordinary directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence symlinks are forbidden")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_hash(path)
        elif not path.is_dir():
            raise ValueError("Special evidence files are forbidden")
    return result


def tree_identity(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def disjoint(*paths: Path) -> None:
    resolved = [Path(path).resolve() for path in paths]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("Serving evidence, archive, and runtime work paths must be disjoint")


def checked_path(root: Path, relative: str) -> Path:
    """Resolve and verify every move endpoint before any recursive rename."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Move paths must be safe relative POSIX paths")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError("Move path escapes the serving evidence directory")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Serving evidence root cannot be a symlink")
    target = root.joinpath(*parts.parts)
    if target.resolve() == root.resolve() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Resolved move endpoint escapes serving evidence")
    for parent in (target, *target.parents):
        if parent.is_symlink():
            raise ValueError("Symlink move endpoints are forbidden")
        if parent == root:
            break
    return target


def move_within(root: Path, source_relative: str, destination_relative: str) -> None:
    source, destination = checked_path(root, source_relative), checked_path(root, destination_relative)
    if not source.exists() or destination.exists():
        raise FileExistsError("Move requires an existing source and a new destination")
    if source.resolve() in destination.resolve().parents:
        raise ValueError("Cannot move a directory inside itself")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A within-directory rename preserves all files and refuses an existing target.
    # No deletion, shell interpolation, or cross-shell filesystem command is used.
    source.rename(destination)


def archive_original(serving: Path, archive: Path) -> dict[str, str]:
    disjoint(serving, archive)
    if archive.exists():
        raise FileExistsError("Original evidence archive already exists")
    before = tree_hashes(serving)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(serving, archive)
    if tree_hashes(serving) != before or tree_hashes(archive) != before:
        raise ValueError("Original evidence changed during archive copy; no active results were moved")
    return before


def source_identity() -> dict[str, Any]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    paths = [Path(__file__).resolve(), PROJECT_ROOT / "scripts" / "run_serving_experiments.py",
             PROJECT_ROOT / "pyproject.toml", *sorted((PROJECT_ROOT / "src" / "llm_platform").rglob("*.py"))]
    return {"code_revision": revision, "code_dirty": dirty,
            "source_files_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): file_hash(path) for path in paths}}


def assert_precision(value: dict[str, Any], label: str) -> None:
    if any(value.get(key) != expected for key, expected in PRECISION.items()):
        raise ValueError(f"{label} does not verify BF16 base/adapter compute and FP32 adapter storage")
    if not isinstance(value.get("precision_source"), str) or not value["precision_source"].strip():
        raise ValueError(f"{label} is missing the precision verification source")


def expected_names(workloads: list[dict[str, Any]], repetitions: int = 2) -> tuple[set[str], set[str]]:
    adapter = {f"{engine}-adapter-r{rep}-{workload['name']}" for engine in ("transformers", "vllm")
               for rep in range(repetitions) for workload in workloads}
    all_names = adapter | {name.replace("-adapter-", "-base-") for name in adapter}
    all_names |= {f"prefix-{toggle}-r{rep}" for toggle in ("off", "on") for rep in range(repetitions)}
    return adapter, all_names


def read_original(serving: Path, manifests: dict[str, dict[str, Any]]) -> tuple[dict, dict[str, dict]]:
    if (serving / "superseded").exists() or (serving / "precision-correction.json").exists():
        raise FileExistsError("A precision amendment already exists; do not overwrite prior correction evidence")
    protocol = json.loads((serving / "protocol.json").read_text(encoding="utf-8"))
    index = json.loads((serving / "index.json").read_text(encoding="utf-8"))
    if (serving / "incomplete.json").exists() or index.get("completed_runs") != 28:
        raise ValueError("Wait for the original complete 28-run matrix before correcting adapter arms")
    if (protocol.get("workloads") != list(WORKLOADS) or protocol.get("requests_per_run") != 12
            or protocol.get("repetitions") != 2 or protocol.get("excluded_warmup_requests") != 2):
        raise ValueError("Original protocol differs from the supported three-workload/12-request/two-repetition matrix")
    if protocol.get("source_artifacts", {}).get("adapter") != manifests["transformers"]["artifact_id"]:
        raise ValueError("Correction adapter does not match the originally measured adapter identity")
    adapter_names, all_names = expected_names(protocol["workloads"])
    present = {path.parent.name for path in (serving / "runs").glob("*/report.json")}
    if present != all_names:
        raise ValueError("Original active run inventory must contain exactly the complete expected 28 runs")
    reports = {name: json.loads((serving / "runs" / name / "report.json").read_text(encoding="utf-8"))
               for name in all_names}
    for name in adapter_names:
        engine = name.split("-", 1)[0]
        report, manifest = reports[name], manifests[engine]
        identities = semantic_ids(manifest)
        if (report["model"] != manifest["artifact_id"] or
                any(report["metadata"].get(key) != value for key, value in identities.items())):
            raise ValueError("Correction would change the adapter weights or tokenizer instead of compute precision")
    for engine in ("transformers", "vllm"):
        for rep in range(2):
            session = serving / "sessions" / f"{engine}-adapter-r{rep}"
            shutdown = json.loads((session / "shutdown.json").read_text(encoding="utf-8"))
            if not shutdown or any(item.get("forced_shutdown") for item in shutdown):
                raise ValueError("Original adapter server sessions must have finished cleanly before correction")
    for required in (serving / "comparison.md", serving / "comparisons"):
        if not required.exists():
            raise ValueError("Original completed comparison outputs are missing")
    return protocol, reports


def rebuild_derived(serving: Path, protocol: dict, names: set[str]) -> tuple[dict[str, dict], list[dict]]:
    present = {path.parent.name for path in (serving / "runs").glob("*/report.json")}
    if present != names:
        raise ValueError("Active corrected matrix is incomplete or contains unexpected run reports")
    reports = [json.loads((serving / "runs" / name / "report.json").read_text(encoding="utf-8"))
               for name in sorted(names)]
    comparisons = {}
    for weight in ("base", "adapter"):
        for workload in protocol["workloads"]:
            pair = [report for report in reports if report["metadata"]["weight_label"] == weight
                    and report["metadata"]["workload"] == workload["name"]]
            if len(pair) != 4:
                raise ValueError("Expected two repetitions for each engine in every comparison")
            comparisons[f"{weight}-{workload['name']}.json"] = compare_reports(pair)
    cache = [report for report in reports if report["metadata"]["workload"] == "repeated-prefix-c4"]
    if len(cache) != 4:
        raise ValueError("Expected the four original prefix-cache reports")
    comparisons["prefix-cache-toggle.json"] = compare_reports(cache)
    return comparisons, reports


async def execute(args: argparse.Namespace) -> None:
    disjoint(args.serving, args.archive, args.work)
    if args.plan_only:
        print(json.dumps({"state": "plan only; no archive, move, or GPU operation performed",
                          "fresh_adapter_sessions": 4, "replacement_runs": 12, "retained_base_prefix_runs": 16,
                          "prerequisite": "Original complete 28-run matrix and all other GPU experiments finished"}, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("Real adapter serving correction must run on Linux/WSL2")
    if args.archive.exists() or args.work.exists():
        raise FileExistsError("Use a new original archive and runtime work directory")
    if args.serving.is_symlink():
        raise ValueError("Serving evidence cannot be a symlink")
    serving, archive, work = args.serving.resolve(), args.archive.resolve(), args.work.resolve()
    packages = {"transformers": args.adapter_package.resolve(), "vllm": args.vllm_package.resolve()}
    manifests = {engine: verify_package(package) for engine, package in packages.items()}
    if any(manifest["kind"] != "adapter" or manifest["runtime"] != {
            "backend": engine, "dtype": "bfloat16", "quantization": "none"}
           for engine, manifest in manifests.items()):
        raise ValueError("Correction requires unquantized BF16 adapter packages bound to each engine")
    if semantic_ids(manifests["transformers"]) != semantic_ids(manifests["vllm"]):
        raise ValueError("Correction engine packages differ in actual weight/tokenizer identity")
    protocol, old_reports = read_original(serving, manifests)
    inventories = {"transformers": software_inventory(sys.executable), "vllm": software_inventory(args.vllm_python)}
    source = source_identity()
    original = archive_original(serving, archive)
    adapter_names, all_names = expected_names(protocol["workloads"])
    old_paths = ["index.json", "comparisons", "comparison.md", *[f"runs/{name}" for name in sorted(adapter_names)],
                 *[f"sessions/{engine}-adapter-r{rep}" for engine in ("transformers", "vllm") for rep in range(2)]]
    moved_files = {relative for relative in original
                   if any(relative == prefix or relative.startswith(prefix + "/") for prefix in old_paths)}
    unaffected = {relative: value for relative, value in original.items() if relative not in moved_files}
    amendment = {"schema_version": 1, "status": "in_progress", "started_at_unix_seconds": time.time(),
        "reason": "Adversarial review found PEFT's original Transformers LoRA default computed FP32 while vLLM auto selected BF16. Only affected adapter serving arms are rerun with explicit verified BF16 compute; archived numbers remain historical confounded observations.",
        "method": "Post-discovery protocol amendment, not a newly claimed original pre-frozen design. Same immutable adapter/base/tokenizer, prompts, decoding policy, request counts and workloads; four fresh adapter sessions with aligned compute precision. Base/prefix and earlier training/quality evidence are retained unchanged.",
        "original_archive": {"path": str(Path("..") / archive.name) if archive.parent == serving.parent else str(archive),
                             "tree_sha256": tree_identity(original), "files_sha256": original},
        "original_protocol_sha256": original["protocol.json"], "source": source,
        "engine_environments": inventories, "environment": environment(), "expected_precision": PRECISION,
        "replacement_runs": sorted(adapter_names), "retained_runs": sorted(all_names - adapter_names),
        "completed_replacement_runs": [], "replacements": {}, "session_replacements": {},
        "limitations": ["Only adapter-serving arms were rerun after discovering a precision confound; execution date/order and source differ from the original base/cache measurements.",
                        "Two repetitions of twelve requests per arm do not establish stable tails or broad engine superiority.",
                        "vLLM precision identity is trusted-launch configuration plus tensor-header verification, not hardware attestation."]}
    write_json(serving / "precision-correction.json", amendment)
    started_mutation = False
    try:
        # Index is retired first: its absence explicitly prevents a partial matrix
        # from appearing complete while affected evidence is replaced.
        for relative in old_paths:
            move_within(serving, relative, "superseded/" + relative)
            started_mutation = True
        for relative in moved_files:
            if file_hash(checked_path(serving, "superseded/" + relative)) != original[relative]:
                raise ValueError("Superseded evidence differs from the original archive")
        work.mkdir(parents=True, exist_ok=False)
        for engine in ("transformers", "vllm"):
            package, manifest = packages[engine], manifests[engine]
            for rep in range(2):
                session_name = f"{engine}-adapter-r{rep}"
                print("START_REAL_ALIGNED_ADAPTER_SESSION", session_name, flush=True)
                async with server_session(engine, package, manifest, [package], work / session_name,
                    serving / "sessions" / session_name, args.vllm_python, inventories[engine], prefix_cache=False) as session:
                    assert_precision(session["startup"]["ready_engine_state"], "Runtime health")
                    for workload in protocol["workloads"]:
                        name = f"{session_name}-{workload['name']}"
                        config = BenchmarkConfig(requests=protocol["requests_per_run"], concurrency=workload["concurrency"],
                            warmup=2, max_tokens=workload["max_tokens"], stream=workload["stream"], repetition=rep)
                        metadata = {**metadata_for(manifest, engine, session, False), "weight_label": "adapter",
                                    "workload": workload["name"], "precision_correction": "precision-correction.json",
                                    "experiment_source": source}
                        assert_precision(metadata, "Benchmark metadata")
                        print("MEASURE_REAL_ALIGNED_ADAPTER_HTTP", name, flush=True)
                        report = await run_benchmark(session["url"], manifest["artifact_id"], workload["prompts"],
                            serving / "runs" / name, config, metadata, session["api_key"])
                        assert_precision(report["metadata"], "Measured report")
                        old = old_reports[name]
                        if report["config"] != old["config"] or any(report["metadata"].get(key) != old["metadata"].get(key)
                            for key in ("weights_id", "tokenizer_id", "prompt_set_sha256", "decoding_policy_sha256")):
                            raise ValueError("Precision correction changed a controlled workload/weight/tokenizer/decoding setting")
                        relative = f"runs/{name}"
                        amendment["replacements"][relative] = {"original_files_sha256": tree_hashes(archive / relative),
                                                              "corrected_files_sha256": tree_hashes(serving / relative)}
                        amendment["completed_replacement_runs"].append(name)
                        write_json(serving / "precision-correction.json", amendment)
                relative = f"sessions/{session_name}"
                amendment["session_replacements"][relative] = {"original_files_sha256": tree_hashes(archive / relative),
                                                               "corrected_files_sha256": tree_hashes(serving / relative)}
                write_json(serving / "precision-correction.json", amendment)
        # No complete index is produced unless all comparison guards succeed.
        comparisons, reports = rebuild_derived(serving, protocol, all_names)
        for relative, expected in unaffected.items():
            if file_hash(checked_path(serving, relative)) != expected:
                raise ValueError("An unaffected base/cache/lifecycle/protocol file changed during correction")
        if tree_hashes(archive) != original:
            raise ValueError("Original archive changed during correction")
        for name, comparison in comparisons.items():
            write_json(serving / "comparisons" / name, comparison)
        note = ("\n## Adapter precision correction\n\n"
                "Adapter arms were rerun after review identified FP32-vs-BF16 LoRA compute as a confound. "
                "Corrected adapter compute is verified BF16 with FP32 artifact storage on both engines. "
                "Original observations are preserved in the immutable sibling serving-original archive and superseded/. "
                "Base/cache observations retain their original source/runtime; adapter reruns use code revision `"
                + source["code_revision"] + "` (dirty=" + str(source["code_dirty"]) + "). "
                "Exact old/new raw/session hashes and source-file hashes are in precision-correction.json. "
                "This is a post-discovery amendment; the original protocol.json remains byte-for-byte unchanged.\n")
        (serving / "comparison.md").write_text(summary_markdown(reports, 12, 2) + note, encoding="utf-8")
        amendment.update(status="complete", completed_at_unix_seconds=time.time(),
                         unchanged_original_files_sha256=unaffected,
                         corrected_report_sha256={name: file_hash(serving / "runs" / name / "report.json") for name in sorted(adapter_names)},
                         derived_files_sha256={**{f"comparisons/{name}": file_hash(serving / "comparisons" / name) for name in comparisons},
                                              "comparison.md": file_hash(serving / "comparison.md")})
        write_json(serving / "precision-correction.json", amendment)
        write_json(serving / "index.json", {"completed_runs": len(reports), "requests_per_run": 12, "repetitions": 2,
            "total_measured_requests": sum(report["summary"]["requests"] for report in reports),
            "successful_measured_requests": sum(report["summary"]["successful_requests"] for report in reports),
            "all_run_reports": sorted(f"runs/{name}/report.json" for name in all_names),
            "precision_correction": "precision-correction.json", "source": source})
    except BaseException as exc:
        if started_mutation and (serving / "index.json").exists():
            move_within(serving, "index.json", "superseded/failed-correction-derived/index.json")
        amendment.update(status="incomplete", exception_type=type(exc).__name__,
                         state="No complete corrected index. Preserve immutable original archive, superseded raw evidence, and any partial corrected outputs.")
        write_json(serving / "precision-correction.json", amendment)
        write_json(serving / "precision-correction-incomplete.json", {"exception_type": type(exc).__name__,
            "completed_replacement_runs": len(amendment["completed_replacement_runs"]), "status": "incomplete"})
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--serving", type=Path, default=Path("evidence/serving"))
    result.add_argument("--archive", type=Path, default=Path("evidence/serving-original"))
    result.add_argument("--adapter-package", type=Path, default=Path("evidence/adapter-package"))
    result.add_argument("--vllm-package", type=Path, default=Path("runtime/serving-001/packages/adapter-vllm"))
    result.add_argument("--vllm-python", type=Path, required=True)
    result.add_argument("--work", type=Path, default=Path("runtime/aligned-adapters-001"))
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    asyncio.run(execute(parser().parse_args()))


if __name__ == "__main__":
    main()
