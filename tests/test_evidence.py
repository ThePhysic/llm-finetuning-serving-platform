from __future__ import annotations

import json
from pathlib import Path
import shutil
import hashlib
import subprocess

import pytest

from llm_platform.data import file_hash, load_records, write_jsonl
from llm_platform.evaluation import make_blinded_review
from llm_platform.evidence import _source_hashes, _within, validate_evidence


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
