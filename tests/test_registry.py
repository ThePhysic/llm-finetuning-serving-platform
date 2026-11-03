import hashlib
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from llm_platform.registry import ArtifactError, Registry, canonical_json, create_package, sha256_file, verify_package


def make_package(tmp_path: Path, name="one", *, kind="base", backend="transformers"):
    source = tmp_path / (name + "-source")
    tokenizer = source / "tokenizer"
    tokenizer.mkdir(parents=True)
    template = "{{ messages }}"
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    (tokenizer / "tokenizer.json").write_text("{}")
    metadata = {
        "schema_version": 1, "kind": kind,
        "base_model": {"id": "example/model", "revision": "a" * 40},
        "tokenizer": {"id": "example/model", "revision": "a" * 40, "path": "tokenizer",
                      "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()},
        "adapter": None,
        "runtime": {"backend": backend, "dtype": "float16", "quantization": "none"},
        "provenance": {"code_revision": "b" * 40, "seed": 42, "dataset_id": "data-1", "experiment_id": name},
        "evaluation": {"report": None, "limitations": ["Test fixture; not an actual model"]},
    }
    if kind == "adapter":
        adapter = source / "adapter"
        adapter.mkdir()
        config = {"base_model_name_or_path": "example/model", "revision": "a" * 40}
        (adapter / "adapter_config.json").write_text(json.dumps(config))
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture only")
        metadata["adapter"] = {"path": "adapter", "config": config}
    destination = tmp_path / (name + "-package")
    manifest = create_package(source, destination, metadata)
    return destination, manifest


def test_package_content_identity_and_corruption(tmp_path):
    package, manifest = make_package(tmp_path)
    assert verify_package(package) == manifest
    (package / "tokenizer" / "tokenizer.json").write_text("tampered")
    with pytest.raises(ArtifactError, match="hashes"):
        verify_package(package)


def test_extra_file_is_rejected(tmp_path):
    package, _ = make_package(tmp_path)
    (package / "untracked-secret.txt").write_text("not a real secret")
    with pytest.raises(ArtifactError, match="hashes"):
        verify_package(package)


def test_register_idempotence_and_rollback(tmp_path):
    first, one = make_package(tmp_path)
    second, two = make_package(tmp_path, "two", kind="adapter")
    registry = Registry(tmp_path / "registry")
    assert registry.register(first) == registry.register(first) == one["artifact_id"]
    registry.register(second)
    assert len(registry.list()) == 2
    registry.promote(one["artifact_id"])
    registry.promote(two["artifact_id"])
    assert registry.rollback()["current"] == one["artifact_id"]
    state = registry.deployment()
    (registry.packages / two["artifact_id"] / "tokenizer" / "tokenizer.json").write_text("corrupt")
    with pytest.raises(ArtifactError):
        registry.promote(two["artifact_id"])
    assert registry.deployment() == state
    assert len(registry.list()) == 1


def test_no_mutation_existing_package(tmp_path):
    package, manifest = make_package(tmp_path)
    with pytest.raises(ArtifactError, match="already exists"):
        create_package(tmp_path / "one-source", package, manifest)
    assert verify_package(package) == manifest


@pytest.mark.parametrize("revision", ["main", "", None, "v1.0"])
def test_mutable_base_revision_rejected(tmp_path, revision):
    package, manifest = make_package(tmp_path)
    manifest["base_model"]["revision"] = revision
    with pytest.raises(ArtifactError, match="immutable"):
        create_package(tmp_path / "one-source", tmp_path / "invalid", manifest)
    assert package.exists()


def test_adapter_model_compatibility(tmp_path):
    _, manifest = make_package(tmp_path, kind="adapter")
    source = tmp_path / "one-source"
    wrong = {"base_model_name_or_path": "wrong/base"}
    (source / "adapter" / "adapter_config.json").write_text(json.dumps(wrong))
    manifest["adapter"]["config"] = wrong
    with pytest.raises(ArtifactError, match="incompatible"):
        create_package(source, tmp_path / "invalid", manifest)


def test_template_identity_and_missing_weights(tmp_path):
    _, manifest = make_package(tmp_path, kind="adapter")
    source = tmp_path / "one-source"
    (source / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"chat_template": "altered"}))
    with pytest.raises(ArtifactError, match="template"):
        create_package(source, tmp_path / "invalid", manifest)
    (source / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (source / "adapter" / "adapter_model.safetensors").unlink()
    with pytest.raises(ArtifactError, match="weights"):
        create_package(source, tmp_path / "invalid2", manifest)


def test_registry_traversal_and_corrupt_state(tmp_path):
    registry = Registry(tmp_path / "registry")
    with pytest.raises(ArtifactError):
        registry.verify("../../other")
    registry.state_path.write_text('{"current": "bad"}')
    with pytest.raises(ArtifactError, match="corrupt"):
        registry.deployment()


def test_registry_lock_rejects_competing_writer(tmp_path):
    package, manifest = make_package(tmp_path)
    registry = Registry(tmp_path / "registry")
    registry.register(package)
    (registry.root / ".deployment.lock").write_text("test")
    with pytest.raises(ArtifactError, match="writer"):
        registry.promote(manifest["artifact_id"])
    assert registry.deployment()["current"] is None


@pytest.mark.parametrize("key,value", [("id", "different/tokenizer"), ("revision", "c" * 40)])
def test_mismatched_tokenizer_rejected_before_registration(tmp_path, key, value):
    package, metadata = make_package(tmp_path)
    metadata["tokenizer"][key] = value
    with pytest.raises(ArtifactError, match="Tokenizer identity"):
        create_package(tmp_path / "one-source", tmp_path / "invalid", metadata)
    assert verify_package(package)["tokenizer"][key] != value


@pytest.mark.parametrize("backend,quantization,dtype", [
    ("transformers", "awq", "float16"), ("transformers", "gptq", "float16"),
    ("transformers", "fp8", "float16"), ("transformers", "none", "auto"),
    ("vllm", "bnb4", "float16"),
])
def test_unsupported_backend_runtime_combinations_rejected(tmp_path, backend, quantization, dtype):
    _, metadata = make_package(tmp_path)
    metadata["runtime"] = {"backend": backend, "quantization": quantization, "dtype": dtype}
    with pytest.raises(ArtifactError, match="unsupported|explicit"):
        create_package(tmp_path / "one-source", tmp_path / "invalid", metadata)


def test_symlinks_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe").write_text("fixture")
    try:
        (source / "linked").symlink_to(source / "safe")
    except OSError:
        pytest.skip("Windows symlink creation requires a host privilege")
    with pytest.raises(ArtifactError, match="Symlinks"):
        create_package(source, tmp_path / "destination", {})


@pytest.mark.parametrize("field,value", [
    ("base_model", []), ("tokenizer", 1), ("runtime", None), ("provenance", []),
    ("evaluation", "invalid"), ("adapter", []), ("adapter_config", []),
    ("adapter_config", 1), ("adapter_config", None), ("tokenizer_config", []),
    ("tokenizer_config", True), ("template", {"default": "{{ messages }}"}),
])
def test_malformed_container_artifact_cannot_break_healthy_registry_listing(tmp_path, field, value):
    package, original = make_package(tmp_path, kind="adapter")
    registry = Registry(tmp_path / "registry")
    registry.register(package)
    corrupt = tmp_path / "corrupt"
    shutil.copytree(package, corrupt)
    manifest = json.loads((corrupt / "manifest.json").read_text())
    if field == "adapter_config":
        manifest["adapter"]["config"] = value
        (corrupt / "adapter" / "adapter_config.json").write_text(json.dumps(value))
    elif field == "tokenizer_config":
        (corrupt / "tokenizer" / "tokenizer_config.json").write_text(json.dumps(value))
    elif field == "template":
        (corrupt / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"chat_template": value}))
        manifest["tokenizer"]["chat_template_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    else:
        manifest[field] = value
    manifest["files"] = {path.relative_to(corrupt).as_posix(): sha256_file(path) for path in corrupt.rglob("*")
                         if path.is_file() and path.name != "manifest.json"}
    manifest.pop("artifact_id")
    identity = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manifest["artifact_id"] = identity
    (corrupt / "manifest.json").write_bytes(canonical_json(manifest))
    corrupt.rename(registry.packages / identity)
    with pytest.raises(ArtifactError):
        registry.verify(identity)
    assert [item["artifact_id"] for item in registry.list()] == [original["artifact_id"]]
    assert registry.deployment()["current"] is None


def test_deeply_nested_corrupt_manifest_is_omitted_without_breaking_discovery(tmp_path):
    package, original = make_package(tmp_path)
    registry = Registry(tmp_path / "registry")
    registry.register(package)
    corrupt = registry.packages / ("e" * 64)
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text('{"bad":' + "[" * 1500 + "0" + "]" * 1500 + "}")
    assert [item["artifact_id"] for item in registry.list()] == [original["artifact_id"]]


def test_registry_rollback_selection_is_inside_the_exclusive_writer_transaction(tmp_path, monkeypatch):
    registry = Registry(tmp_path / "registry")
    identities = []
    for name in ("one", "two", "three"):
        package, manifest = make_package(tmp_path, name)
        identities.append(registry.register(package))
    registry.promote(identities[0])
    registry.promote(identities[1])
    observed, release = threading.Event(), threading.Event()
    original = registry.deployment
    results, errors = [], []

    def gated_read():
        state = original()
        if threading.current_thread().name == "registry-rollback-audit":
            observed.set()
            if not release.wait(2):
                raise RuntimeError("Test rollback release deadline exceeded")
        return state

    def rollback():
        try:
            results.append(registry.rollback())
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(registry, "deployment", gated_read)
    worker = threading.Thread(target=rollback, name="registry-rollback-audit")
    worker.start()
    try:
        assert observed.wait(1)
        with pytest.raises(ArtifactError, match="writer"):
            registry.promote(identities[2])
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and not errors
    assert results[0]["current"] == registry.deployment()["current"] == identities[0]
    assert registry.promote(identities[2])["previous"] == identities[0]


def test_registry_package_container_cannot_redirect_writes_through_an_alias(tmp_path):
    root, outside = tmp_path / "registry", tmp_path / "unrelated-storage"
    root.mkdir()
    outside.mkdir()
    (outside / "keep.txt").write_text("unchanged")
    alias = root / "packages"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("Host cannot create a directory alias")
        # NTFS junctions need no symlink privilege and are a separate reparse
        # type: Path.is_symlink alone does not protect this storage boundary.
        path_arg, target_arg = str(alias).replace("'", "''"), str(outside).replace("'", "''")
        result = subprocess.run(["powershell", "-NoProfile", "-Command",
            f"New-Item -ItemType Junction -Path '{path_arg}' -Target '{target_arg}' | Out-Null"],
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
    with pytest.raises(ArtifactError, match="storage"):
        Registry(root)
    assert sorted(path.name for path in outside.iterdir()) == ["keep.txt"]
    assert (outside / "keep.txt").read_text() == "unchanged"
