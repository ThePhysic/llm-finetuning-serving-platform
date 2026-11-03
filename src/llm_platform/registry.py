"""Content-addressed model packages and atomic deployment pointers.

The registry is deliberately a local, single-writer store. Model weights remain in
the Hugging Face cache; only adapters, tokenizers and provenance are packaged.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
import math
from typing import Any


class ArtifactError(ValueError):
    """The package or its compatibility metadata could not be verified."""


def _linklike(path: Path) -> bool:
    """Reject symlinks and Windows directory junction/reparse aliases."""
    if path.is_symlink():
        return True
    try:
        return bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


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
    if _linklike(root) or not root.is_dir():
        raise ArtifactError("Package must be an ordinary directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if _linklike(path):
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
        for component in ("base_model", "tokenizer", "runtime", "provenance", "evaluation"):
            if not isinstance(manifest[component], dict):
                raise ArtifactError(f"{component} metadata must be an object")
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
        if not isinstance(token_config, dict):
            raise ArtifactError("Tokenizer configuration must be an object")
        template_file = token_path / "chat_template.jinja"
        template = template_file.read_text(encoding="utf-8") if template_file.is_file() else token_config.get("chat_template")
        if not isinstance(template, str) or not template.strip():
            raise ArtifactError("Supported packaged chat templates must be nonempty strings")
        # Both supported backends require a single explicit UTF-8 template.
        template_bytes = template.encode("utf-8")
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
        if evaluation["report"] is not None and (not isinstance(evaluation["report"], str) or not evaluation["report"].strip()):
            raise ArtifactError("Evaluation report must be a reference or null")
        if evaluation["report"] is None and not evaluation["limitations"]:
            raise ArtifactError("Unevaluated artifacts must state their limitation")
        adapter = manifest["adapter"]
        if manifest["kind"] == "adapter":
            if any(not isinstance(provenance.get(key), str) or not provenance[key].strip()
                   for key in ("dataset_id", "experiment_id")):
                raise ArtifactError("Adapter dataset and experiment identities are required")
            if not isinstance(adapter, dict) or not isinstance(adapter.get("config"), dict):
                raise ArtifactError("Adapter metadata and configuration must be objects")
            adapter_path = root.joinpath(*_safe_name(adapter["path"]).parts)
            adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
            if not isinstance(adapter_config, dict):
                raise ArtifactError("Packaged adapter configuration must be an object")
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
    except ArtifactError:
        raise
    except (KeyError, TypeError, AttributeError, OSError, ValueError, RecursionError) as exc:
        raise ArtifactError(f"Incomplete or malformed package metadata: {exc}") from exc


def verify_package(package: Path) -> dict[str, Any]:
    """Verify every byte and the identity before a runtime sees the package."""
    root = Path(package)
    if _linklike(root) or _linklike(root / "manifest.json"):
        raise ArtifactError("Package symlinks are forbidden")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise ArtifactError("Missing or malformed manifest") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("Manifest must be an object")
    supplied = manifest.get("artifact_id")
    content = {key: value for key, value in manifest.items() if key != "artifact_id"}
    try:
        expected_identity = _digest(content)
    except (ValueError, TypeError, RecursionError) as exc:
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
        if _linklike(self.root):
            raise ArtifactError("Registry root cannot be a symlink")
        self.packages = self.root / "packages"
        if _linklike(self.packages):
            raise ArtifactError("Registry package storage cannot be a symlink or junction")
        self.packages.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "deployment.json"

    def _check_storage(self) -> None:
        if _linklike(self.root) or _linklike(self.packages) or _linklike(self.state_path):
            raise ArtifactError("Registry storage aliases are forbidden")

    @staticmethod
    def _id(artifact_id: str) -> str:
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[a-f0-9]{64}", artifact_id):
            raise ArtifactError("Model selection requires a full artifact SHA256")
        return artifact_id

    def register(self, package: Path) -> str:
        self._check_storage()
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
        self._check_storage()
        path = self.packages / self._id(artifact_id)
        manifest = verify_package(path)
        if manifest["artifact_id"] != artifact_id:
            raise ArtifactError("Registry directory does not match package identity")
        return path, manifest

    def list(self) -> list[dict[str, Any]]:
        self._check_storage()
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
        self._check_storage()
        if not self.state_path.exists():
            return {"current": None, "previous": None, "generation": 0}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._id(state["current"])
            if state["previous"] is not None:
                self._id(state["previous"])
            if type(state["generation"]) is not int or state["generation"] < 1:
                raise ValueError("Invalid generation")
            updated = state.get("updated_at")
            if updated is not None and (type(updated) not in {int, float} or not math.isfinite(updated)):
                raise ValueError("Invalid deployment timestamp")
            canonical_json(state)  # reject non-finite values anywhere in persisted state
            return state
        except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
            raise ArtifactError("Deployment state is corrupt") from exc

    @contextmanager
    def _deployment_writer(self):
        self._check_storage()
        lock = self.root / ".deployment.lock"
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ArtifactError("Another deployment writer holds the registry lock") from exc
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(str(os.getpid()).encode())
            yield
        finally:
            lock.unlink(missing_ok=True)

    def _select(self, state: dict[str, Any], artifact_id: str) -> dict[str, Any]:
        self.verify(artifact_id)
        if state["current"] == artifact_id:
            return state
        updated = {"current": artifact_id, "previous": state["current"],
                   "generation": state["generation"] + 1, "updated_at": time.time()}
        _atomic_json(self.state_path, updated)
        return updated

    def promote(self, artifact_id: str) -> dict[str, Any]:
        """Verify and select under one exclusive cross-process writer lock.

        Runtime loading is separately verified by the gateway. Stale locks require
        operator inspection rather than unsafe automatic removal.
        """
        with self._deployment_writer():
            return self._select(self.deployment(), artifact_id)

    def rollback(self) -> dict[str, Any]:
        # Read the previous identity under the same lock as pointer mutation.
        with self._deployment_writer():
            state = self.deployment()
            if state["previous"] is None:
                raise ArtifactError("No prior verified deployment exists")
            return self._select(state, state["previous"])
