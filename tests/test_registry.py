import hashlib
import json
from pathlib import Path

import pytest

from llm_platform.registry import ArtifactError, Registry, create_package, verify_package


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
