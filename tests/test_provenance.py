from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from llm_platform.provenance import capture_source_snapshot


@pytest.fixture
def source_repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Source provenance requires git")
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args, input_data=None):
        return subprocess.run(["git", "-C", str(root), *args], input=input_data,
                              shell=False, capture_output=True, check=True).stdout

    git("init")
    git("config", "user.name", "Source Provenance Test")
    git("config", "user.email", "source-test@example.invalid")
    git("config", "core.autocrlf", "false")
    contents = {"src/model.py": b"# original model module\r\n",
                "src/pkg/helper.py": b"# original helper\n", "scripts/run.py": b"# runner\n"}
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "docs").mkdir()
    (root / "docs" / "example.py").write_bytes(b"# outside captured source scopes\n")
    (root / "src" / "config.json").write_text("{}", encoding="utf-8")
    git("add", "src", "scripts", "docs")
    git("commit", "-m", "Record source tree")
    revision = git("rev-parse", "HEAD").decode().strip()
    return root, contents, revision, git


def test_current_snapshot_preserves_raw_bytes_hashes_and_scope(source_repo, tmp_path):
    root, contents, revision, _ = source_repo
    # Unrelated documentation edits do not make project source dirty.
    (root / "docs" / "example.py").write_bytes(b"# later documentation example\n")
    destination = tmp_path / "captured"
    manifest = capture_source_snapshot(root, destination)
    assert manifest["source_revision"] == revision
    assert manifest["mode"] == "pre-generation filesystem source capture"
    assert set(manifest["files"]) == set(contents)
    for name, content in contents.items():
        assert (destination / name).read_bytes() == content
        assert manifest["files"][name] == hashlib.sha256(content).hexdigest()
    supplied = manifest["snapshot_sha256"]
    unsigned = {key: value for key, value in manifest.items() if key != "snapshot_sha256"}
    assert supplied == hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert json.loads((destination / "manifest.json").read_text()) == manifest


@pytest.mark.parametrize("name", ["src/model.py", "scripts/new_uncommitted.py"])
def test_current_snapshot_rejects_dirty_tracked_and_untracked_source(source_repo, tmp_path, name):
    root, _, _, _ = source_repo
    (root / name).write_bytes(b"# uncommitted implementation\n")
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="Commit source"):
        capture_source_snapshot(root, destination)
    assert not destination.exists()


def test_snapshot_refuses_to_overwrite_preserved_source(source_repo, tmp_path):
    root, _, _, _ = source_repo
    destination = tmp_path / "snapshot"
    manifest = capture_source_snapshot(root, destination)
    before = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        capture_source_snapshot(root, destination)
    assert (destination / "manifest.json").read_bytes() == before
    assert json.loads(before)["snapshot_sha256"] == manifest["snapshot_sha256"]


def test_exact_historical_revision_excludes_later_edits_and_new_files(source_repo, tmp_path):
    root, contents, revision, git = source_repo
    (root / "src" / "model.py").write_bytes(b"# later committed correction\n")
    (root / "scripts" / "new.py").write_bytes(b"# new script absent at startup\n")
    git("add", "src", "scripts")
    git("commit", "-m", "Later corrected source")
    (root / "src" / "model.py").write_bytes(b"# still later uncommitted correction\n")
    destination = tmp_path / "historical"
    manifest = capture_source_snapshot(root, destination, revision=revision)
    assert manifest["mode"] == "retrospective exact Git tree reconstruction"
    assert manifest["source_revision"] == revision
    assert set(manifest["files"]) == set(contents)
    for name, content in contents.items():
        assert (destination / name).read_bytes() == content
    assert not (destination / "scripts" / "new.py").exists()


@pytest.mark.parametrize("revision", ["main", "HEAD", "../HEAD", "a" * 39, "a" * 40 + ":../src/model.py"])
def test_branch_and_revision_traversal_are_rejected_before_git(tmp_path, monkeypatch, revision):
    def reject_command(*args, **kwargs):
        raise AssertionError("Invalid immutable revision must never reach git")

    monkeypatch.setattr("llm_platform.provenance.subprocess.check_output", reject_command)
    with pytest.raises(ValueError, match="complete immutable revision"):
        capture_source_snapshot(tmp_path, tmp_path / "snapshot", revision=revision)


def test_unavailable_immutable_revision_is_rejected(source_repo, tmp_path):
    root, _, _, _ = source_repo
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        capture_source_snapshot(root, tmp_path / "missing", revision="f" * 40)
    assert not (tmp_path / "missing").exists()


def test_git_tree_source_traversal_is_rejected_without_reading_it(source_repo, tmp_path, monkeypatch):
    root, _, revision, _ = source_repo
    original = subprocess.check_output

    def tree_with_unsafe_path(command, **kwargs):
        if "ls-tree" in command:
            return b"100644 blob " + b"a" * 40 + b"\tsrc/../outside.py\0"
        return original(command, **kwargs)

    monkeypatch.setattr("llm_platform.provenance.subprocess.check_output", tree_with_unsafe_path)
    with pytest.raises(ValueError, match="Unsafe source path"):
        capture_source_snapshot(root, tmp_path / "unsafe", revision=revision)
    assert not (tmp_path / "unsafe").exists()


def test_historical_committed_symlink_is_not_misrepresented_as_python_source(source_repo, tmp_path):
    root, _, _, git = source_repo
    # A real git symlink object exercises history mode without requiring Windows
    # symlink privileges or reading any external file.
    blob = git("hash-object", "-w", "--stdin", input_data=b"../../outside-private.py").decode().strip()
    git("update-index", "--add", "--cacheinfo", f"120000,{blob},src/linked.py")
    git("commit", "-m", "Record an unsupported linked source")
    revision = git("rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="ordinary committed source files"):
        capture_source_snapshot(root, tmp_path / "linked-history", revision=revision)


def test_current_source_symlink_outside_is_rejected_without_secret_read(source_repo, tmp_path, monkeypatch):
    root, _, _, git = source_repo
    outside = tmp_path / "outside-private.py"
    outside.write_bytes(b"test-only payload must never be read by source capture\n")
    linked = root / "src" / "outside.py"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Host does not grant filesystem symlink creation privileges")
    git("add", "src/outside.py")
    git("commit", "-m", "Record linked source for refusal test")
    original = Path.read_bytes

    def guarded_read(path):
        if path.resolve() == outside.resolve():
            pytest.fail("Snapshot attempted to read an external source target")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError, match="ordinary committed source files|cannot follow links"):
        capture_source_snapshot(root, tmp_path / "linked-current")
