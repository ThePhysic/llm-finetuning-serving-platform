"""Content-addressed model packages and atomic deployment pointers.

The registry is deliberately a local, single-writer store. Model weights remain in
the Hugging Face cache; only adapters, tokenizers and provenance are packaged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from typing import Any


class ArtifactError(ValueError):
    """The package or its compatibility metadata could not be verified."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _safe_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ArtifactError("Package paths must be relative POSIX paths")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {".", "..", ""} for part in name.split("/")):
        raise ArtifactError("Unsafe package path")
    return path


def _files(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("Package must be an ordinary directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArtifactError("Symlinks are not permitted in model packages")
        if not path.is_file():
            if not path.is_dir():
                raise ArtifactError("Special files are not permitted")
            continue
        name = path.relative_to(root).as_posix()
        if name == "manifest.json":
            continue
        _safe_name(name)
        files[name] = sha256_file(path)
    return files


def _revision(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", value):
        raise ArtifactError(f"{label} must be an immutable hexadecimal commit revision")


def _metadata(manifest: dict[str, Any], root: Path) -> None:
    try:
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["kind"] not in {"base", "adapter"}:
            raise ArtifactError("Unsupported package schema or kind")
        for component in ("base_model", "tokenizer"):
            value = manifest[component]
            if not isinstance(value["id"], str) or not value["id"].strip():
                raise ArtifactError(f"Missing {component} identifier")
            _revision(value["revision"], component)
        if any(manifest["tokenizer"][key] != manifest["base_model"][key] for key in ("id", "revision")):
            raise ArtifactError("Tokenizer identity must match the supported base model revision")
        tokenizer = manifest["tokenizer"]
        token_path = root.joinpath(*_safe_name(tokenizer["path"]).parts)
        if not token_path.is_dir():
            raise ArtifactError("Packaged tokenizer directory is missing")
        if not re.fullmatch(r"[a-f0-9]{64}", tokenizer["chat_template_sha256"]):
            raise ArtifactError("Missing chat-template identity")
        config_path = token_path / "tokenizer_config.json"
        if not config_path.is_file():
            raise ArtifactError("Tokenizer configuration is missing")
        token_config = json.loads(config_path.read_text(encoding="utf-8"))
        template_file = token_path / "chat_template.jinja"
        template = template_file.read_text(encoding="utf-8") if template_file.is_file() else token_config.get("chat_template")
        if template is None:
            raise ArtifactError("Packaged chat template is missing")
        # Transformers saves a string template as raw UTF-8; named templates use
        # canonical JSON to retain a stable identity.
        template_bytes = template.encode("utf-8") if isinstance(template, str) else canonical_json(template)
        if hashlib.sha256(template_bytes).hexdigest() != tokenizer["chat_template_sha256"]:
            raise ArtifactError("Packaged chat template does not match its identity")
        runtime = manifest["runtime"]
        if runtime["backend"] not in {"transformers", "vllm"}:
            raise ArtifactError("Unsupported execution backend")
        if runtime["quantization"] not in {"none", "bnb4", "awq", "gptq", "fp8"}:
            raise ArtifactError("Unsupported serving quantization")
        if runtime["dtype"] not in {"float32", "float16", "bfloat16", "auto"}:
            raise ArtifactError("Unsupported runtime dtype")
        supported_quantization = {"transformers": {"none", "bnb4"},
                                  "vllm": {"none", "awq", "gptq", "fp8"}}
        if runtime["quantization"] not in supported_quantization[runtime["backend"]]:
            raise ArtifactError("Quantization format is unsupported by the selected backend")
        if runtime["backend"] == "transformers" and runtime["dtype"] == "auto":
            raise ArtifactError("Transformers packages require an explicit serving dtype")
        provenance = manifest["provenance"]
        _revision(provenance["code_revision"], "code_revision")
        if not isinstance(provenance["seed"], int) or isinstance(provenance["seed"], bool):
            raise ArtifactError("Experiment seed must be an integer")
        evaluation = manifest["evaluation"]
        if not isinstance(evaluation["limitations"], list) or not all(isinstance(x, str) for x in evaluation["limitations"]):
            raise ArtifactError("Known limitations must be recorded")
        if evaluation["report"] is not None and not isinstance(evaluation["report"], str):
            raise ArtifactError("Evaluation report must be a reference or null")
        if evaluation["report"] is None and not evaluation["limitations"]:
            raise ArtifactError("Unevaluated artifacts must state their limitation")
        adapter = manifest["adapter"]
        if manifest["kind"] == "adapter":
            if not provenance.get("dataset_id") or not provenance.get("experiment_id"):
                raise ArtifactError("Adapter dataset and experiment identities are required")
            adapter_path = root.joinpath(*_safe_name(adapter["path"]).parts)
            adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
            if adapter["config"] != adapter_config:
                raise ArtifactError("Adapter metadata differs from packaged configuration")
            if adapter_config.get("base_model_name_or_path") != manifest["base_model"]["id"]:
                raise ArtifactError("Adapter is incompatible with the base model")
            revision = adapter_config.get("revision")
            if revision != manifest["base_model"]["revision"]:
                raise ArtifactError("Adapter base revision mismatch")
            if not (adapter_path / "adapter_model.safetensors").is_file():
                raise ArtifactError("Safetensors adapter weights are missing")
        elif adapter is not None:
            raise ArtifactError("Base packages cannot carry an adapter")
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Incomplete or malformed package metadata: {exc}") from exc


def verify_package(package: Path) -> dict[str, Any]:
    """Verify every byte and the identity before a runtime sees the package."""
    root = Path(package)
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ArtifactError("Package symlinks are forbidden")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError("Missing or malformed manifest") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("Manifest must be an object")
    supplied = manifest.get("artifact_id")
    content = {key: value for key, value in manifest.items() if key != "artifact_id"}
    try:
        expected_identity = _digest(content)
    except (ValueError, TypeError) as exc:
        raise ArtifactError("Manifest contains unsupported JSON values") from exc
    if not isinstance(supplied, str) or supplied != expected_identity:
        raise ArtifactError("Manifest identity mismatch")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ArtifactError("Manifest must identify packaged files")
    for name, digest in expected.items():
        _safe_name(name)
        if name == "manifest.json" or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ArtifactError("Invalid file hash entry")
    actual = _files(root)
    if actual != expected:
        raise ArtifactError("Package contents differ from their immutable file hashes")
    _metadata(manifest, root)
    return manifest


def create_package(source: Path, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a new package without mutating an existing destination."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ArtifactError("Package destination already exists")
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        raise ArtifactError("Package destination cannot be inside its source")
    _files(source)  # reject symlinks before copying
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".package-", dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        manifest = {key: value for key, value in metadata.items() if key not in {"files", "artifact_id"}}
        manifest.setdefault("schema_version", 1)
        manifest["files"] = _files(temporary)
        manifest["artifact_id"] = _digest(manifest)
        (temporary / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        verify_package(temporary)
        # rename is deliberately not replace: an immutable package cannot be overwritten.
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Registry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ArtifactError("Registry root cannot be a symlink")
        self.packages = self.root / "packages"
        self.packages.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "deployment.json"

    @staticmethod
    def _id(artifact_id: str) -> str:
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[a-f0-9]{64}", artifact_id):
            raise ArtifactError("Model selection requires a full artifact SHA256")
        return artifact_id

    def register(self, package: Path) -> str:
        manifest = verify_package(package)
        artifact_id = manifest["artifact_id"]
        target = self.packages / artifact_id
        if target.exists():
            self.verify(artifact_id)
            return artifact_id
        temporary = Path(tempfile.mkdtemp(prefix=".register-", dir=self.packages))
        try:
            shutil.copytree(package, temporary, dirs_exist_ok=True)
            verify_package(temporary)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return artifact_id

    def verify(self, artifact_id: str) -> tuple[Path, dict[str, Any]]:
        path = self.packages / self._id(artifact_id)
        manifest = verify_package(path)
        if manifest["artifact_id"] != artifact_id:
            raise ArtifactError("Registry directory does not match package identity")
        return path, manifest

    def list(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.packages.iterdir()):
            if path.name.startswith("."):
                continue
            try:
                _, manifest = self.verify(path.name)
                result.append(manifest)
            except ArtifactError:
                # Corrupt artifacts remain inspectable on disk but never discoverable for serving.
                continue
        return result

    def deployment(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"current": None, "previous": None, "generation": 0}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._id(state["current"])
            if state["previous"] is not None:
                self._id(state["previous"])
            if not isinstance(state["generation"], int):
                raise ValueError("Invalid generation")
            return state
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ArtifactError("Deployment state is corrupt") from exc

    def promote(self, artifact_id: str) -> dict[str, Any]:
        """Persist a verified pointer. The gateway separately verifies runtime loading.

        An exclusive cross-process lock rejects overlapping local CLI/admin writers;
        stale locks require operator inspection instead of unsafe automatic removal.
        """
        self.verify(artifact_id)
        lock = self.root / ".deployment.lock"
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ArtifactError("Another deployment writer holds the registry lock") from exc
        try:
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            state = self.deployment()
            if state["current"] == artifact_id:
                return state
            updated = {"current": artifact_id, "previous": state["current"],
                       "generation": state["generation"] + 1, "updated_at": time.time()}
            _atomic_json(self.state_path, updated)
            return updated
        finally:
            lock.unlink(missing_ok=True)

    def rollback(self) -> dict[str, Any]:
        previous = self.deployment()["previous"]
        if previous is None:
            raise ArtifactError("No prior verified deployment exists")
        return self.promote(previous)
