"""CPU-only archive/control-flow fixtures; no fixture is a GPU measurement."""
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from llm_platform.benchmark import BenchmarkConfig
from llm_platform.registry import create_package
from test_registry import make_package


@pytest.fixture
def realign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("realign_adapter_benchmarks")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def prompt_hash(prompts):
    return hashlib.sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def fixture_matrix(tmp_path, module):
    source, metadata = make_package(tmp_path, kind="adapter")
    metadata["runtime"]["dtype"] = "bfloat16"
    adapter = tmp_path / "adapter"
    tf_manifest = create_package(source, adapter, metadata)
    metadata["runtime"]["backend"] = "vllm"
    vllm = tmp_path / "vllm-adapter"
    vllm_manifest = create_package(adapter, vllm, metadata)
    manifests = {"transformers": tf_manifest, "vllm": vllm_manifest}
    serving = tmp_path / "serving"
    protocol = {"schema_version": 1, "cpu_fixture": True, "workloads": list(deepcopy(module.WORKLOADS)),
                "requests_per_run": 12, "repetitions": 2, "excluded_warmup_requests": 2,
                "source_artifacts": {"adapter": tf_manifest["artifact_id"], "base": "b" * 64}}
    write(serving / "protocol.json", protocol)
    reports = {}
    for engine in ("transformers", "vllm"):
        for weight in ("base", "adapter"):
            for rep in range(2):
                for workload in protocol["workloads"]:
                    name = f"{engine}-{weight}-r{rep}-{workload['name']}"
                    config = BenchmarkConfig(requests=12, concurrency=workload["concurrency"], warmup=2,
                                             max_tokens=workload["max_tokens"], stream=workload["stream"], repetition=rep)
                    meta = {**module.semantic_ids(manifests[engine]), "engine": engine, "weight_label": weight,
                            "workload": workload["name"], "prompt_set_sha256": prompt_hash(workload["prompts"]),
                            "decoding_policy_sha256": "d" * 64, "cache_state": {"prefix_caching": False},
                            "engine_settings": {"dtype": "bfloat16", "adapter": weight == "adapter"}}
                    if weight == "base":
                        meta["weights_id"] = "base-fixture-weights"
                    report = {"cpu_fixture": True, "model": manifests[engine]["artifact_id"] if weight == "adapter" else "b" * 64,
                              "config": asdict(config), "metadata": meta,
                              "summary": {"requests": 12, "successful_requests": 12, "output_tokens_per_second": 1,
                                          "latency_seconds": {"p50": 1}, "ttft_seconds": {"p50": 1}, "error_rate": 0}}
                    reports[name] = report
    for toggle in ("off", "on"):
        for rep in range(2):
            report = deepcopy(reports[f"vllm-base-r{rep}-short-stream-c1"])
            report["config"]["concurrency"] = 4
            report["metadata"]["workload"] = "repeated-prefix-c4"
            report["metadata"]["prompt_set_sha256"] = "cache-fixture-prompt"
            report["metadata"]["cache_state"] = {"prefix_caching": toggle == "on"}
            reports[f"prefix-{toggle}-r{rep}"] = report
    for name, report in reports.items():
        directory = serving / "runs" / name
        write(directory / "report.json", report)
        write(directory / "requests.jsonl", {"cpu_fixture": True, "historical_run": name})
        write(directory / "gpu.jsonl", {"cpu_fixture": True, "not_a_gpu_measurement": True})
    for engine in ("transformers", "vllm"):
        for weight in ("base", "adapter"):
            for rep in range(2):
                directory = serving / "sessions" / f"{engine}-{weight}-r{rep}"
                write(directory / "startup.json", {"cpu_fixture": True})
                write(directory / "shutdown.json", [{"forced_shutdown": False, "exit_code": 0, "cpu_fixture": True}])
    for rep in range(2):
        directory = serving / "sessions" / f"vllm-base-prefix-on-r{rep}"
        write(directory / "startup.json", {"cpu_fixture": True})
        write(directory / "shutdown.json", [{"forced_shutdown": False, "exit_code": 0, "cpu_fixture": True}])
    write(serving / "lifecycle" / "lifecycle.json", {"cpu_fixture": True})
    write(serving / "index.json", {"completed_runs": 28, "cpu_fixture": True})
    for weight in ("base", "adapter"):
        for workload in protocol["workloads"]:
            write(serving / "comparisons" / f"{weight}-{workload['name']}.json", {"cpu_fixture": True})
    write(serving / "comparisons" / "prefix-cache-toggle.json", {"cpu_fixture": True})
    (serving / "comparison.md").write_text("Original CPU fixture summary\n")
    args = module.parser().parse_args(["--serving", str(serving), "--archive", str(tmp_path / "serving-original"),
        "--adapter-package", str(adapter), "--vllm-package", str(vllm), "--vllm-python", "unused",
        "--work", str(tmp_path / "work")])
    return args, protocol, reports, manifests


def test_plan_and_help_cannot_archive_move_or_launch_gpu(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "realign_adapter_benchmarks.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0
    serving, archive, work = tmp_path / "missing", tmp_path / "archive", tmp_path / "work"
    result = subprocess.run([sys.executable, str(script), "--vllm-python", "unused", "--serving", str(serving),
                             "--archive", str(archive), "--work", str(work), "--plan-only"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["replacement_runs"] == 12
    assert not serving.exists() and not archive.exists() and not work.exists()


def test_archive_is_exact_and_existing_archive_cannot_be_overwritten(tmp_path, realign):
    source = tmp_path / "serving"
    write(source / "protocol.json", {"cpu_fixture": True})
    write(source / "nested" / "raw.json", {"original": "keep"})
    before = realign.tree_hashes(source)
    archive = tmp_path / "archive"
    assert realign.archive_original(source, archive) == before
    assert realign.tree_hashes(source) == realign.tree_hashes(archive) == before
    with pytest.raises(FileExistsError):
        realign.archive_original(source, archive)
    assert realign.tree_hashes(archive) == before


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "runs/../outside", "runs\\outside", "C:/outside", ""])
def test_move_paths_cannot_escape_or_rename_unrelated_files(tmp_path, realign, relative):
    source = tmp_path / "serving"
    write(source / "runs" / "fixture.json", {"keep": True})
    unrelated = tmp_path / "outside"
    unrelated.write_text("unchanged")
    before = realign.tree_hashes(source)
    with pytest.raises(ValueError):
        realign.move_within(source, relative, "superseded/fixture")
    with pytest.raises(ValueError):
        realign.move_within(source, "runs/fixture.json", relative)
    assert unrelated.read_text() == "unchanged"
    assert realign.tree_hashes(source) == before


@pytest.mark.parametrize("bad_health", [False, True])
async def test_cpu_realign_preserves_original_bytes_and_never_indexes_partial_correction(tmp_path, realign, monkeypatch, bad_health):
    args, protocol, old_reports, manifests = fixture_matrix(tmp_path, realign)
    original = realign.tree_hashes(args.serving)
    monkeypatch.setattr(realign, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(realign, "source_identity", lambda: {"code_revision": "c" * 40, "code_dirty": False, "source_files_sha256": {"cpu-fixture": "f" * 64}})
    monkeypatch.setattr(realign, "environment", lambda: {"cpu_fixture": True})
    monkeypatch.setattr(realign, "software_inventory", lambda python: {"cpu_fixture": True})
    events = []

    @asynccontextmanager
    async def fixture_session(engine, package, manifest, all_packages, work, output, vllm_python, inventory, prefix_cache):
        assert all_packages == [package] and not prefix_cache
        events.append(("start", engine))
        precision = {**realign.PRECISION, "precision_source": "CPU fixture; not a runtime measurement"}
        if bad_health:
            precision["adapter_compute_dtype"] = "float32"
        startup = {"ready_engine_state": {**precision, "decoding_policy": {"sha256": "d" * 64}, "max_model_len": 2048},
                   "engine_software": inventory, "cpu_fixture": True}
        write(output / "startup.json", startup)
        try:
            yield {"url": "http://cpu-fixture", "api_key": "private-fixture-key-never-publish", "startup": startup}
        finally:
            write(output / "shutdown.json", [{"forced_shutdown": False, "exit_code": 0, "cpu_fixture": True}])
            events.append(("close", engine))

    async def fixture_benchmark(url, model, prompts, output, config, metadata, api_key):
        assert api_key == "private-fixture-key-never-publish"
        report = deepcopy(old_reports[output.name])
        assert report["config"] == asdict(config)
        report["metadata"] = {**metadata, "prompt_set_sha256": prompt_hash(prompts)}
        write(output / "report.json", report)
        write(output / "requests.jsonl", {"cpu_fixture": True, "corrected": True})
        write(output / "gpu.jsonl", {"cpu_fixture": True, "not_a_gpu_measurement": True})
        return report

    monkeypatch.setattr(realign, "server_session", fixture_session)
    monkeypatch.setattr(realign, "run_benchmark", fixture_benchmark)
    if bad_health:
        with pytest.raises(ValueError, match="Runtime health"):
            await realign.execute(args)
        assert not (args.serving / "index.json").exists()
        assert (args.serving / "precision-correction-incomplete.json").exists()
        amendment = json.loads((args.serving / "precision-correction.json").read_text())
        assert amendment["status"] == "incomplete"
        assert not amendment["completed_replacement_runs"]
        assert events == [("start", "transformers"), ("close", "transformers")]
    else:
        await realign.execute(args)
        index = json.loads((args.serving / "index.json").read_text())
        assert index["completed_runs"] == 28
        assert index["total_measured_requests"] == 336
        amendment = json.loads((args.serving / "precision-correction.json").read_text())
        assert amendment["status"] == "complete"
        assert len(amendment["completed_replacement_runs"]) == 12
        assert len(amendment["session_replacements"]) == 4
        assert len(list((args.serving / "comparisons").glob("*.json"))) == 7
        assert len(events) == 8
        for relative, digest in amendment["unchanged_original_files_sha256"].items():
            assert realign.file_hash(args.serving / relative) == digest == original[relative]
    assert realign.tree_hashes(args.archive) == original
    assert amendment["original_archive"]["tree_sha256"] == realign.tree_identity(original)
    assert amendment["original_protocol_sha256"] == original["protocol.json"]
    assert realign.file_hash(args.serving / "protocol.json") == original["protocol.json"]
    for relative, digest in original.items():
        superseded = args.serving / "superseded" / relative
        if superseded.is_file():
            assert realign.file_hash(superseded) == digest
    assert (args.serving / "superseded" / "index.json").is_file()
    assert len(list((args.serving / "superseded" / "runs").iterdir())) == 12
    evidence = "".join(path.read_text() for path in args.serving.rglob("*") if path.is_file())
    assert "private-fixture-key-never-publish" not in evidence


def test_derived_rebuild_rejects_actual_adapter_precision_mismatch_before_writes(tmp_path, realign):
    args, protocol, reports, manifests = fixture_matrix(tmp_path, realign)
    for name, report in reports.items():
        if "-adapter-" in name:
            report["metadata"].update(**realign.PRECISION, precision_source="CPU fixture")
            report["metadata"]["engine_settings"].update(realign.PRECISION)
            write(args.serving / "runs" / name / "report.json", report)
    name = "vllm-adapter-r1-short-stream-c1"
    reports[name]["metadata"]["engine_settings"]["adapter_storage_dtype"] = "float16"
    write(args.serving / "runs" / name / "report.json", reports[name])
    before = realign.tree_hashes(args.serving)
    _, names = realign.expected_names(protocol["workloads"])
    with pytest.raises(ValueError, match="precision differs"):
        realign.rebuild_derived(args.serving, protocol, names)
    assert realign.tree_hashes(args.serving) == before
