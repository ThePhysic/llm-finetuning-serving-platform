"""CPU-only orchestration checks; these never establish measured GPU results."""
from contextlib import asynccontextmanager
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from llm_platform.registry import create_package
from test_registry import make_package


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("measure_load_rates")


def test_rate_sweep_help_and_plan_never_load_artifacts_or_start_servers(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_load_rates.py"
    help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=20)
    assert help_result.returncode == 0
    assert "--rates" in help_result.stdout
    work, output = tmp_path / "work", tmp_path / "evidence"
    result = subprocess.run([sys.executable, str(script), "--vllm-python", "missing-python",
                             "--base-package", "missing-base", "--vllm-package", "missing-vllm",
                             "--work", str(work), "--output", str(output), "--plan-only"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["measured_runs"] == 4
    assert plan["request_rates"] == [1.0, 32.0]
    assert not work.exists() and not output.exists()


async def test_existing_evidence_refused_before_platform_or_package_operations(tmp_path, sweep, monkeypatch):
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "result.json").write_text('{"keep": true}')
    args = sweep.parser().parse_args(["--vllm-python", "unused", "--work", str(tmp_path / "work"),
                                     "--output", str(output)])
    monkeypatch.setattr(sweep, "verify_package", lambda path: pytest.fail("Existing evidence must fail before package loading"))
    with pytest.raises(FileExistsError):
        await sweep.execute(args)
    assert (output / "result.json").read_text() == '{"keep": true}'
    assert not args.work.exists()


@pytest.mark.parametrize("rates", [[1], [1, 1], [1, 0], [1, float("inf")], [1, float("nan")]])
def test_rate_configuration_rejects_ambiguous_or_nonfinite_pacing(tmp_path, sweep, rates):
    args = sweep.parser().parse_args(["--vllm-python", "unused", "--work", str(tmp_path / "work"),
                                     "--output", str(tmp_path / "output")])
    args.rates = rates
    with pytest.raises(ValueError):
        sweep.validate_args(args)


def test_pacing_report_keeps_client_delay_and_server_queue_distinct(sweep):
    rows = [{"index": 0, "client_schedule_delay_seconds": 0},
            {"index": 1, "client_schedule_delay_seconds": 1},
            {"index": 2, "client_schedule_delay_seconds": 2}]
    result = sweep.pacing_observations(rows, 32,
                                     {"queue_seconds_sum": 1, "queue_seconds_count": 10},
                                     {"queue_seconds_sum": 3, "queue_seconds_count": 15})
    assert result["configured_request_rate"] == 32
    assert result["dispatch_span_requests_per_second"] < 1
    assert result["client_schedule_delay_seconds"]["maximum"] == 2
    assert result["gateway_queue_mean_seconds_including_warmup"] == 0.4
    unavailable = sweep.pacing_observations(rows, 32,
                                          {"queue_seconds_sum": None, "queue_seconds_count": None},
                                          {"queue_seconds_sum": None, "queue_seconds_count": None})
    assert unavailable["gateway_queue_mean_seconds_including_warmup"] is None


@pytest.mark.parametrize("fail_second_arm", [False, True])
async def test_cpu_orchestration_preserves_fair_pairs_cleanup_and_partial_results(tmp_path, sweep, monkeypatch, fail_second_arm):
    original, metadata = make_package(tmp_path)
    metadata["runtime"]["dtype"] = "bfloat16"
    base = tmp_path / "base"
    create_package(original, base, metadata)
    metadata["runtime"]["backend"] = "vllm"
    vllm = tmp_path / "vllm"
    create_package(base, vllm, metadata)
    args = sweep.parser().parse_args(["--base-package", str(base), "--vllm-package", str(vllm),
        "--vllm-python", "unused", "--work", str(tmp_path / "work"), "--output", str(tmp_path / "output"),
        "--requests", "2"])
    # Patch only this module's platform handle; changing the shared os.name on
    # Windows would alter pathlib dispatch for unrelated tests.
    monkeypatch.setattr(sweep, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(sweep, "software_inventory", lambda python: {"cpu_fixture": True})
    monkeypatch.setattr(sweep, "environment", lambda: {"cpu_fixture": True})
    monkeypatch.setattr(sweep, "source_identity", lambda: {"code_revision": "c" * 40, "cpu_fixture": True})
    events, arms = [], []
    queue = {"count": 0, "sum": 0.0}

    @asynccontextmanager
    async def fixture_session(engine, package, manifest, all_packages, work, output, vllm_python, inventory, prefix_cache):
        assert not prefix_cache
        assert all_packages == [package]
        events.append(("start", engine))
        try:
            yield {"url": "http://cpu-fixture", "api_key": "private-fixture-key-never-publish", "startup": {
                "ready_engine_state": {"decoding_policy": {"sha256": "d" * 64}, "max_model_len": 2048},
                "engine_software": inventory}}
        finally:
            events.append(("close", engine))

    async def fixture_snapshot(session):
        return {"queue_seconds_sum": queue["sum"], "queue_seconds_count": queue["count"],
                "error": None, "prometheus_text": "CPU fixture telemetry"}

    async def fixture_benchmark(url, model, prompts, output, config, metadata, api_key):
        assert api_key == "private-fixture-key-never-publish"
        assert config.concurrency == 4 and config.warmup == 2 and config.max_tokens == 32 and config.stream
        arms.append((metadata["engine"], config.request_rate))
        output.mkdir(parents=True, exist_ok=False)
        if fail_second_arm and len(arms) == 2:
            (output / "partial.json").write_text('{"cpu_fixture": true}')
            raise RuntimeError("fixture failure")
        rows = [{"index": index, "client_schedule_delay_seconds": index * 0.01} for index in range(config.requests)]
        (output / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        queue["count"] += config.requests + config.warmup
        queue["sum"] += 0.4
        return {"schema_version": 1, "model": model, "config": asdict(config),
                "metadata": {**metadata, "prompt_set_sha256": "e" * 64},
                "summary": {"requests": config.requests, "successful_requests": config.requests,
                            "request_per_second": 1.0, "error_rate": 0}}

    monkeypatch.setattr(sweep, "server_session", fixture_session)
    monkeypatch.setattr(sweep, "queue_snapshot", fixture_snapshot)
    monkeypatch.setattr(sweep, "run_benchmark", fixture_benchmark)
    if fail_second_arm:
        with pytest.raises(RuntimeError, match="fixture failure"):
            await sweep.execute(args)
        incomplete = json.loads((args.output / "incomplete.json").read_text())
        assert incomplete["completed_runs"] == 1
        assert events == [("start", "transformers"), ("close", "transformers")]
        assert not (args.output / "index.json").exists()
        assert len(list((args.output / "runs").glob("*/report.json"))) == 1
    else:
        await sweep.execute(args)
        assert events == [("start", "transformers"), ("close", "transformers"),
                          ("start", "vllm"), ("close", "vllm")]
        assert arms == [("transformers", 1), ("transformers", 32), ("vllm", 1), ("vllm", 32)]
        index = json.loads((args.output / "index.json").read_text())
        assert index["completed_runs"] == 4
        assert index["total_measured_requests"] == 8
        comparisons = list((args.output / "comparisons").glob("*.json"))
        assert len(comparisons) == 2
        for path in comparisons:
            pair = json.loads(path.read_text())["comparison"]
            assert pair[0]["config"]["request_rate"] == pair[1]["config"]["request_rate"]
            assert {row["engine"] for row in pair} == {"transformers", "vllm"}
    evidence = "".join(path.read_text() for path in args.output.rglob("*") if path.is_file())
    assert "private-fixture-key-never-publish" not in evidence
