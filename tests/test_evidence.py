from __future__ import annotations

import json
from pathlib import Path
import shutil
import hashlib
import subprocess
import copy
import time

import pytest

from llm_platform.data import file_hash, load_records, write_jsonl
from llm_platform.contracts import serving_decoding_policy
from llm_platform.evaluation import make_blinded_review
from llm_platform.evidence import (PendingEvidence, _adapter_precision, _execution_snapshot, _graceful_shutdown, _isolated_samples,
                                   _pacing, _runtime_memory, _serving_policy, _shutdown_inventory,
                                   _source_hashes, _startup_summary, _tree_hashes, _within, validate_evidence)
from llm_platform.provenance import capture_source_snapshot


def copy_primary_data(destination: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    shutil.copytree(repository / "data" / "triage-v3", destination / "data" / "triage-v3")
    shutil.copy2(repository / "data" / "synthetic-v2.jsonl", destination / "data" / "synthetic-v2.jsonl")


def test_no_artifacts_are_pending_not_invented_success(tmp_path):
    report = validate_evidence(tmp_path)
    assert report["status"] == "pending"
    assert all(check["status"] == "pending" for check in report["criteria"].values())
    assert report["external_requirements"]
    assert "not project-completion" in report["scope"]


def test_real_frozen_data_can_pass_while_gpu_evidence_remains_pending(tmp_path):
    copy_primary_data(tmp_path)
    report = validate_evidence(tmp_path)
    assert report["criteria"]["frozen_dataset"]["status"] == "pass"
    assert report["criteria"]["gpu_training"]["status"] == "pending"
    assert report["status"] == "pending"


def test_corrupted_frozen_test_is_a_failure(tmp_path):
    copy_primary_data(tmp_path)
    test = tmp_path / "data" / "triage-v3" / "test.jsonl"
    test.write_bytes(test.read_bytes() + b"\n")
    report = validate_evidence(tmp_path)
    assert report["status"] == "fail"
    assert report["criteria"]["frozen_dataset"]["status"] == "fail"
    assert "integrity" in report["criteria"]["frozen_dataset"]["reason"]


def test_gpu_true_flag_alone_does_not_establish_training(tmp_path):
    directory = tmp_path / "evidence" / "training"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text('{"gpu_evidence":true}', encoding="utf-8")
    result = validate_evidence(tmp_path)["criteria"]["gpu_training"]
    assert result["status"] != "pass"


def test_tampered_raw_output_is_a_failure_before_accepting_report(tmp_path):
    copy_primary_data(tmp_path)
    directory = tmp_path / "evidence" / "evaluation" / "zero_shot"
    directory.mkdir(parents=True)
    write_jsonl(directory / "outputs.jsonl", [{"output": "changed after evaluation"}])
    (directory / "report.json").write_text(json.dumps({"outputs_sha256": "0" * 64}), encoding="utf-8")
    result = validate_evidence(tmp_path)["criteria"]["controlled_quality"]
    assert result["status"] == "fail"
    assert "raw output hash mismatch" in result["reason"]


def test_valid_json_without_generation_finish_is_not_success_evidence(tmp_path):
    copy_primary_data(tmp_path)
    directory = tmp_path / "evidence" / "evaluation" / "zero_shot"
    directory.mkdir(parents=True)
    write_jsonl(directory / "outputs.jsonl", [{"output": "{}", "error": None, "latency_s": 1.0}])
    write_jsonl(directory / "scores.jsonl", [])
    (directory / "report.json").write_text(json.dumps({
        "outputs_sha256": file_hash(directory / "outputs.jsonl"),
        "scores_sha256": file_hash(directory / "scores.jsonl"),
    }), encoding="utf-8")
    result = validate_evidence(tmp_path)["criteria"]["controlled_quality"]
    assert result["status"] == "fail"
    assert "incomplete generation" in result["reason"]


def test_fake_human_completion_flag_cannot_replace_annotations(tmp_path):
    directory = tmp_path / "evidence" / "human-review"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text('{"status":"complete"}', encoding="utf-8")
    result = validate_evidence(tmp_path)["criteria"]["blinded_human_review"]
    assert result["status"] == "fail"
    assert "without annotated evidence" in result["reason"]


def test_unannotated_real_review_packet_stays_pending(tmp_path):
    copy_primary_data(tmp_path)
    records = load_records(tmp_path / "data" / "triage-v3" / "test.jsonl")
    # Deliberately empty generation fixtures are used only to prepare a pending
    # annotation packet. They are never accepted as real model evidence.
    outputs = [{"id": row["id"], "output": ""} for row in records]
    directory = tmp_path / "evidence" / "human-review"
    status = make_blinded_review(records, {name: outputs for name in ("zero_shot", "prompted", "lora")},
                                directory, sample_size=2)
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    result = validate_evidence(tmp_path)["criteria"]["blinded_human_review"]
    assert result["status"] == "pending"
    assert "genuine human annotations" in result["reason"]


@pytest.mark.parametrize("name", ["../outside.json", "nested/../../outside.json", "C:\\outside.json"])
def test_evidence_references_cannot_escape_the_project(tmp_path, name):
    with pytest.raises(ValueError, match="relative|escapes"):
        _within(tmp_path, name)


@pytest.fixture
def recorded_source(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Historical source verification requires git")
    source = tmp_path / "src" / "worker.py"
    source.parent.mkdir()
    source.write_bytes(b"recorded training implementation\n")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], shell=False,
                              capture_output=True, check=True).stdout.decode().strip()

    git("init")
    git("config", "user.name", "Evidence Test")
    git("config", "user.email", "evidence-test@example.invalid")
    git("config", "core.autocrlf", "false")
    git("add", "src/worker.py")
    git("commit", "-m", "Record source used by experiment")
    return source, hashlib.sha256(source.read_bytes()).hexdigest(), git("rev-parse", "HEAD"), git


def test_recorded_git_bytes_remain_verifiable_after_later_review_fix(recorded_source):
    source, expected, revision, _ = recorded_source
    source.write_bytes(b"subsequent adversarial review correction\n")
    _source_hashes(source.parents[1], {"src/worker.py": expected}, "training", code_revision=revision)
    # A renamed/removed source still has verifiable immutable historical content.
    source.unlink()
    _source_hashes(source.parents[1], {"src/worker.py": expected}, "training", code_revision=revision)


def test_another_commit_with_different_bytes_does_not_satisfy_provenance(recorded_source):
    source, expected, _, git = recorded_source
    source.write_bytes(b"different committed source\n")
    git("add", "src/worker.py")
    git("commit", "-m", "Commit a different implementation")
    wrong_revision = git("rev-parse", "HEAD")
    source.write_bytes(b"third uncommitted source\n")
    with pytest.raises(ValueError, match="both checkout and recorded revision"):
        _source_hashes(source.parents[1], {"src/worker.py": expected}, "training", code_revision=wrong_revision)


@pytest.mark.parametrize("revision", [None, "main", "HEAD", "a" * 39, "a" * 40 + ";echo invalid"])
def test_unverifiable_revision_never_invokes_git(tmp_path, monkeypatch, revision):
    source = tmp_path / "worker.py"
    source.write_bytes(b"new source")

    def reject_subprocess(*args, **kwargs):
        raise AssertionError("Untrusted revision must be rejected before subprocess execution")

    monkeypatch.setattr("llm_platform.evidence.subprocess.run", reject_subprocess)
    with pytest.raises(ValueError, match="no immutable recorded revision"):
        _source_hashes(tmp_path, {"worker.py": hashlib.sha256(b"old source").hexdigest()},
                       "evaluation", code_revision=revision)


def test_uncommitted_old_bytes_cannot_be_recovered_from_dirty_flag(recorded_source):
    source, _, revision, _ = recorded_source
    uncommitted = b"uncommitted experiment source\n"
    source.write_bytes(b"later source\n")
    with pytest.raises(ValueError, match="both checkout and recorded revision"):
        _source_hashes(source.parents[1], {"src/worker.py": hashlib.sha256(uncommitted).hexdigest()},
                       "training", code_revision=revision)


def serving_metadata():
    policy = serving_decoding_policy(151645, None, 151643)
    return {"decoding_policy": policy, "decoding_policy_sha256": policy["sha256"],
            "actual_startup": {"ready_engine_state": {"decoding_policy": copy.deepcopy(policy)}}}


def test_serving_audit_requires_full_semantic_policy_and_digest():
    metadata = serving_metadata()
    _serving_policy(metadata)
    metadata["decoding_policy"]["repetition_penalty"] = 1.1
    with pytest.raises(ValueError, match="explicit platform defaults"):
        _serving_policy(metadata)


def test_rehashing_a_different_policy_does_not_restore_fairness():
    metadata = serving_metadata()
    policy = metadata["decoding_policy"]
    policy["repetition_penalty"] = 1.1
    fields = {key: value for key, value in policy.items() if key != "sha256"}
    policy["sha256"] = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata["decoding_policy_sha256"] = policy["sha256"]
    metadata["actual_startup"]["ready_engine_state"]["decoding_policy"] = copy.deepcopy(policy)
    with pytest.raises(ValueError, match="explicit platform defaults"):
        _serving_policy(metadata)


def test_reported_policy_must_match_ready_engine_policy():
    metadata = serving_metadata()
    metadata["actual_startup"]["ready_engine_state"]["decoding_policy"] = serving_decoding_policy(151643, None, 151643)
    with pytest.raises(ValueError, match="recorded ready engine"):
        _serving_policy(metadata)


def prepare_execution_evidence(recorded_source, historical):
    source, _, revision, _ = recorded_source
    root = source.parents[1]
    manifest = capture_source_snapshot(root, root / "evidence" / "execution-source",
                                       revision=revision if historical else None)
    protocol = {"source_commit": revision, "frozen_at_unix": time.time()}
    if historical:
        clarification = {"status": "retrospective exact-source clarification",
                         "quality_protocol_source_commit": revision,
                         "execution_source_snapshot_sha256": manifest["snapshot_sha256"],
                         "source_snapshot_mode": manifest["mode"]}
        (root / "evidence" / "execution-provenance.json").write_text(json.dumps(clarification), encoding="utf-8")
    else:
        protocol["execution_source_snapshot_sha256"] = manifest["snapshot_sha256"]
    (root / "evidence" / "quality-protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    return root, manifest


def test_current_execution_snapshot_is_linked_to_frozen_protocol(recorded_source):
    root, manifest = prepare_execution_evidence(recorded_source, historical=False)
    details = _execution_snapshot(root)
    assert details["snapshot_sha256"] == manifest["snapshot_sha256"]
    assert details["contemporaneous_capture"] is True


def test_retrospective_execution_snapshot_does_not_invent_startup_capture(recorded_source):
    root, _ = prepare_execution_evidence(recorded_source, historical=True)
    protocol = json.loads((root / "evidence" / "quality-protocol.json").read_text())
    assert "execution_source_snapshot_sha256" not in protocol
    details = _execution_snapshot(root)
    assert details["contemporaneous_capture"] is False
    assert details["mode"].startswith("retrospective")


def test_execution_snapshot_rejects_changed_preserved_source(recorded_source):
    root, _ = prepare_execution_evidence(recorded_source, historical=True)
    source = root / "evidence" / "execution-source" / "src" / "worker.py"
    source.write_bytes(b"tampered archived source\n")
    with pytest.raises(ValueError, match="source file hash differs"):
        _execution_snapshot(root)


def test_retrospective_snapshot_requires_matching_clarification(recorded_source):
    root, _ = prepare_execution_evidence(recorded_source, historical=True)
    path = root / "evidence" / "execution-provenance.json"
    clarification = json.loads(path.read_text())
    clarification["source_snapshot_mode"] = "captured at startup"
    path.write_text(json.dumps(clarification), encoding="utf-8")
    with pytest.raises(ValueError, match="Retrospective execution clarification"):
        _execution_snapshot(root)


def test_snapshot_claim_cannot_omit_committed_source_even_with_consistent_rehash(recorded_source):
    source, _, revision, git = recorded_source
    root = source.parents[1]
    second = root / "src" / "second.py"
    second.write_bytes(b"# another imported module\n")
    git("add", "src/second.py")
    git("commit", "-m", "Record another module")
    new_revision = git("rev-parse", "HEAD")
    updated = (source, hashlib.sha256(source.read_bytes()).hexdigest(), new_revision, git)
    root, _ = prepare_execution_evidence(updated, historical=True)
    directory = root / "evidence" / "execution-source"
    manifest = json.loads((directory / "manifest.json").read_text())
    del manifest["files"]["src/second.py"]
    (directory / "src" / "second.py").unlink()
    unsigned = {key: value for key, value in manifest.items() if key != "snapshot_sha256"}
    manifest["snapshot_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="omits or invents committed source"):
        _execution_snapshot(root)


def test_runtime_memory_audit_recomputes_observed_median_and_session_peak():
    samples = [{"runtime": {"allocated_bytes": value, "session_peak_allocated_bytes": peak,
                            "reserved_bytes": 10_000}} for value, peak in [(1000, 6000), (5000, 8000), (3000, 8000)]]
    summary = {"runtime_allocated_sample_median_bytes": 3000, "runtime_session_peak_allocated_bytes": 8000}
    _runtime_memory(samples, summary)
    summary["runtime_session_peak_allocated_bytes"] = 10_000
    with pytest.raises(ValueError, match="differs from raw runtime samples"):
        _runtime_memory(samples, summary)


def test_unsupported_runtime_memory_stays_null_instead_of_invented_zero():
    samples = [{"runtime": {"allocated_bytes": None, "session_peak_allocated_bytes": None, "reserved_bytes": None}}]
    summary = {"runtime_allocated_sample_median_bytes": None, "runtime_session_peak_allocated_bytes": None}
    _runtime_memory(samples, summary)
    summary["runtime_allocated_sample_median_bytes"] = 0
    with pytest.raises(ValueError, match="differs from raw runtime samples"):
        _runtime_memory(samples, summary)


@pytest.mark.parametrize("bad_value", [-1, float("nan"), float("inf"), True])
def test_runtime_memory_samples_require_finite_nonnegative_byte_values(bad_value):
    samples = [{"runtime": {"allocated_bytes": None, "session_peak_allocated_bytes": None, "reserved_bytes": bad_value}}]
    with pytest.raises(ValueError, match="runtime reserved_bytes"):
        _runtime_memory(samples, {})


@pytest.mark.parametrize("completed", [False, True])
def test_partial_shutdown_inventory_is_pending_only_while_matrix_in_progress(tmp_path, completed):
    serving = tmp_path / "serving"
    session = serving / "sessions" / "transformers-base-r0"
    session.mkdir(parents=True)
    (session / "shutdown.json").write_text("[]", encoding="utf-8")
    if completed:
        (serving / "index.json").write_text('{"completed_runs":28}', encoding="utf-8")
        with pytest.raises(ValueError, match="shutdown inventory"):
            _shutdown_inventory(serving, expected=10)
    else:
        with pytest.raises(PendingEvidence, match="still in progress"):
            _shutdown_inventory(serving, expected=10)


def test_missing_shutdown_inventory_fails_after_completion_index_exists(tmp_path):
    (tmp_path / "index.json").write_text('{"completed_runs":28}', encoding="utf-8")
    with pytest.raises(ValueError, match="shutdown inventory"):
        _shutdown_inventory(tmp_path, expected=10)


def test_complete_shutdown_inventory_can_pass_before_final_index_is_written(tmp_path):
    for number in range(10):
        session = tmp_path / "sessions" / f"session-{number}"
        session.mkdir(parents=True)
        (session / "shutdown.json").write_text("[]", encoding="utf-8")
    assert len(_shutdown_inventory(tmp_path, expected=10)) == 10


def test_identical_storage_does_not_hide_adapter_compute_precision_mismatch():
    precision = {"base_compute_dtype": "bfloat16", "adapter_compute_dtype": "bfloat16", "adapter_storage_dtype": "float32",
                 "precision_source": "actual-runtime-parameters-and-safetensors-header"}
    metadata = {"weight_label": "adapter", **precision, "engine_settings": dict(precision),
                "actual_startup": {"ready_engine_state": dict(precision)}}
    _adapter_precision(metadata)
    metadata["actual_startup"]["ready_engine_state"]["adapter_compute_dtype"] = "float32"
    with pytest.raises(ValueError, match="actual ready engine"):
        _adapter_precision(metadata)


def test_tree_inventory_detects_archive_byte_mutation(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    raw = directory / "requests.jsonl"
    raw.write_bytes(b'{"latency":1}\n')
    frozen = _tree_hashes(directory)
    raw.write_bytes(b'{"latency":2}\n')
    assert _tree_hashes(directory) != frozen


def test_startup_aggregate_recomputes_paired_raw_loads_and_ratios():
    results = [{"measurements": [{"load_seconds": first, "unload_seconds": 0.1},
                                  {"load_seconds": warm, "unload_seconds": 0.2}]}
               for first, warm in [(9, 3), (6, 3), (12, 4)]]
    summary = {"first_load_seconds": {"raw": [9, 6, 12], "median": 9, "min": 6, "max": 12},
               "warm_reload_seconds": {"raw": [3, 3, 4], "median": 3, "min": 3, "max": 4},
               "unload_seconds": {"raw": [0.1, 0.2] * 3, "median": 0.15000000000000002},
               "paired_first_over_warm_ratios": [3, 2, 3]}
    _startup_summary(summary, results)
    summary["warm_reload_seconds"]["median"] = 2
    with pytest.raises(ValueError, match="differs from raw repetitions"):
        _startup_summary(summary, results)


def test_target_pacing_does_not_become_invented_achieved_rate():
    rows = [{"index": index, "client_schedule_delay_seconds": delay} for index, delay in enumerate([0, 0.5, 1.0])]
    before, after = {"queue_seconds_sum": 10, "queue_seconds_count": 20}, {"queue_seconds_sum": 13, "queue_seconds_count": 23}
    result = _pacing(rows, 2, before, after)
    assert result["configured_request_rate"] == 2
    assert result["dispatch_span_requests_per_second"] == 1
    assert result["client_schedule_delay_seconds"] == {"median": 0.5, "maximum": 1, "total": 1.5}
    assert result["gateway_queue_mean_seconds_including_warmup"] == 1


def test_measurement_isolation_rejects_even_boundary_overlap(tmp_path):
    from datetime import datetime

    directory = tmp_path / "evidence"
    directory.mkdir()
    begin, end = "2026-11-03T02:56:35+00:00", "2026-11-03T02:57:00+00:00"
    (directory / "measurement-isolation.json").write_text(json.dumps({
        "incidents": [{"conservative_window_utc": [begin, end]}]}), encoding="utf-8")
    boundary = datetime.fromisoformat(begin).timestamp()
    _isolated_samples(tmp_path, [{"time": boundary - 2}, {"time": boundary - 1}])
    with pytest.raises(ValueError, match="overlap an excluded CUDA probe"):
        _isolated_samples(tmp_path, [{"time": boundary - 1}, {"time": boundary}])


def test_empty_shutdown_file_cannot_count_as_recorded_graceful_process_exit(tmp_path):
    path = tmp_path / "shutdown.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="no child processes"):
        _graceful_shutdown(path)
    path.write_text(json.dumps([{"forced_shutdown": False, "exit_code": -15, "shutdown_seconds": 0.25}]), encoding="utf-8")
    _graceful_shutdown(path)
