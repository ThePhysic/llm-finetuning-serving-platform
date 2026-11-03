"""Offline advisory identity tests do not establish security or exploitability."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("audit_dependency_aliases", Path(__file__).resolve().parents[1] / "scripts" / "audit_dependency_aliases.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(package, version, vulnerabilities):
    return {"dependencies": [{"name": package, "version": version, "vulns": vulnerabilities}]}


def test_alias_union_is_transitive_across_duplicate_records_and_inventories():
    first = {"id": "PYSEC-A", "aliases": ["CVE-A"], "fix_versions": ["2"]}
    second = {"id": "GHSA-A", "aliases": ["CVE-A", "OSV-A"]}
    third = {"id": "OSV-A", "aliases": ["VENDOR-A"]}
    result = MODULE.deduplicate({"train": report("Torch", "1", [first, first]),
                                 "engine": report("torch", "1", [second, third])})
    assert result["raw_occurrences"] == 4
    assert result["alias_deduplicated_advisories"] == 1
    assert set(result["clusters"][0]["identities"]) == {"PYSEC-A", "CVE-A", "GHSA-A", "OSV-A", "VENDOR-A"}
    assert result["inventories"]["train"]["raw_occurrences"] == 2
    assert result["inventories"]["train"]["alias_deduplicated_advisories"] == 1
    assert result["clusters"][0]["exploitability"].startswith("Not determined")


def test_distinct_issues_are_not_merged_merely_because_the_package_or_fix_matches():
    result = MODULE.deduplicate({"one": report("some_Package", "1", [
        {"id": "CVE-ONE", "fix_versions": ["2"]}, {"id": "CVE-TWO", "fix_versions": ["2"]}])})
    assert result["alias_deduplicated_advisories"] == 2
    assert result["inventories"]["one"]["affected_packages"]["some-package"]["alias_deduplicated_advisories"] == 2


def test_empty_findings_preserve_skips_instead_of_claiming_a_clean_inventory():
    result = MODULE.deduplicate({"one": {"dependencies": [{"name": "local-project", "skip_reason": "not on PyPI"}]}})
    assert result["inventories"]["one"]["skipped"] == [{"name": "local-project", "reason": "not on PyPI"}]
    assert result["alias_deduplicated_advisories"] == 0


@pytest.mark.parametrize("bad", [{"id": ""}, {"id": "A", "aliases": "B"}, {"id": "A", "aliases": [None]}])
def test_malformed_identity_cannot_silently_change_audit_counts(bad):
    with pytest.raises(ValueError):
        MODULE.deduplicate({"one": report("p", "1", [bad])})


def test_dependency_gpu_probe_refuses_hidden_cuda_before_any_artifact_write(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "polish_training_smoke.py"
    work, output = tmp_path / "work", tmp_path / "result.json"
    result = subprocess.run([sys.executable, str(script), "--gpu", "--work", str(work), "--output", str(output)],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "required before importing Torch" in result.stderr
    assert not work.exists() and not output.exists()
