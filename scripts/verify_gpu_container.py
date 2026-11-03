"""Build and verify the historical GPU image against actual HTTP generation.

Build and smoke use separate fresh output directories. A successful build alone
is never a successful GPU smoke. Run smoke only during an exclusive GPU window.
Credentials are ephemeral environment values and are redacted from saved output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from llm_platform.provenance import capture_source_snapshot  # noqa: E402
from llm_platform.registry import sha256_file, verify_package  # noqa: E402

SECURITY_SCOPE = "HISTORICAL/TRUSTED-LOCAL-ONLY; known dependency advisories; no untrusted deployment claim"
PROMPT = "Synthetic test: a shipment is delayed; the carrier reports no scan since yesterday. Summarize briefly."


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fresh_output(path: Path) -> Path:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


class Recorder:
    def __init__(self, output: Path, secrets_to_hide: tuple[str, ...] = ()):
        self.output, self.secrets_to_hide = output, secrets_to_hide
        self.commands: list[dict] = []

    def redact(self, text: str) -> str:
        for value in self.secrets_to_hide:
            text = text.replace(value, "[REDACTED]")
        return text

    def run(self, args: list[str], *, env: dict | None = None, timeout: float = 120,
            check: bool = True, label: str | None = None) -> str:
        started = time.time()
        result = subprocess.run(args, env=env, cwd=ROOT, shell=False, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout)
        entry = {"argv": [self.redact(item) for item in args], "started_at_unix": started,
                 "seconds": time.time() - started, "returncode": result.returncode,
                 "stdout": self.redact(result.stdout), "stderr": self.redact(result.stderr)}
        self.commands.append(entry)
        write_json(self.output / "commands.json", self.commands)
        if label:
            write_json(self.output / f"{label}.json", entry)
        if check and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {args[0]} {args[1]}")
        return result.stdout.strip()


def stage_build_context(root: Path, output: Path) -> dict:
    # The Dockerfile/dependency inputs must also be committed, not just Python.
    scopes = ["src", "scripts", "docker/Dockerfile.gpu", "requirements", "pyproject.toml", ".dockerignore"]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *scopes], cwd=root, text=True)
    if dirty.strip():
        raise ValueError("Commit GPU container sources, Dockerfile and dependency inputs before building")
    snapshot = capture_source_snapshot(root, output / "source")
    context = output / "build-context"
    context.mkdir()
    for path in (output / "source" / "src").rglob("*.py"):
        target = context / path.relative_to(output / "source")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    inputs = [root / "docker" / "Dockerfile.gpu", root / "pyproject.toml", root / ".dockerignore"]
    inputs += sorted((root / "requirements").glob("*"))
    hashes = {}
    for source in inputs:
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError("Build inputs must be ordinary files inside the repository")
        relative = source.relative_to(root)
        target = context / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = sha256_file(target)
    hashes.update({name: sha for name, sha in snapshot["files"].items() if name.startswith("src/")})
    return {"source_revision": snapshot["source_revision"], "snapshot_sha256": snapshot["snapshot_sha256"],
            "context_files_sha256": hashes, "security_scope": SECURITY_SCOPE}


def build(root: Path, output: Path) -> dict:
    output = fresh_output(output)
    recorder = Recorder(output)
    status = {"schema_version": 1, "operation": "build", "status": "failed", "gpu_verified": False,
              "recorded_at": datetime.now(timezone.utc).isoformat(), "security_scope": SECURITY_SCOPE}
    try:
        status.update(stage_build_context(root, output))
        recorder.run(["docker", "version", "--format", "{{json .}}"], label="docker-version")
        dockerfile = (output / "build-context/docker/Dockerfile.gpu").read_text(encoding="utf-8")
        base_image = next(line.split()[1] for line in dockerfile.splitlines() if line.upper().startswith("FROM "))
        base_inventory = json.loads(recorder.run(["docker", "image", "inspect", base_image,
                                                 "--format", "{{json .}}"], label="base-image-inspect"))
        status["base_image"] = {"reference": base_image, "image_id": base_inventory["Id"],
                                "repo_digests": base_inventory.get("RepoDigests") or []}
        tag = "llm-platform-gpu-verify:" + uuid.uuid4().hex
        status["image_tag"] = tag
        recorder.run(["docker", "build", "--pull=false", "--progress=plain", "--tag", tag,
                      "--file", str(output / "build-context/docker/Dockerfile.gpu"),
                      str(output / "build-context")], timeout=3600, label="build")
        image = json.loads(recorder.run(["docker", "image", "inspect", tag, "--format",
                                       "{{json .}}"], label="image-inspect"))
        status.update(status="built", image_id=image["Id"], repo_digests=image.get("RepoDigests") or [],
                      configured_user=image["Config"]["User"])
        if status["configured_user"] in {"", "root", "0"}:
            raise ValueError("Built image is configured as root")
        return status
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=recorder.redact(str(exc)))
        raise
    finally:
        write_json(output / "result.json", status)


def usage(value: dict) -> None:
    if not isinstance(value, dict) or any(type(value.get(key)) is not int or value[key] < 0
                                         for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ValueError("Missing authoritative nonnegative token usage")
    if value["completion_tokens"] < 1 or value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]:
        raise ValueError("Invalid authoritative token accounting")


def identity(value: dict, model: str, revision: str) -> None:
    if value.get("model") != model or value.get("system_fingerprint") != model:
        raise ValueError("Response served a different artifact")
    metadata = value.get("metadata", {})
    if metadata.get("artifact_id") != model or metadata.get("base_model_revision") != revision:
        raise ValueError("Response revision metadata mismatch")


def validate_completion(value: dict, model: str, revision: str) -> None:
    identity(value, model, revision)
    choice = value["choices"][0]
    if choice.get("finish_reason") not in {"stop", "length"} or not choice["message"].get("content"):
        raise ValueError("Empty or unfinished completion")
    usage(value.get("usage"))


def validate_sse(data: list[str], model: str, revision: str) -> dict:
    if not data or data[-1] != "[DONE]" or data.count("[DONE]") != 1:
        raise ValueError("Stream omitted its unique terminal DONE")
    events = [json.loads(item) for item in data[:-1]]
    contents, finishes, usages = [], [], []
    for event in events:
        if "error" in event:
            raise ValueError("Stream contained an error")
        identity(event, model, revision)
        if "usage" in event:
            if len(finishes) != 1 or event.get("choices") != []:
                raise ValueError("Stream usage must follow its terminal finish")
            usage(event["usage"])
            usages.append(event["usage"])
        for choice in event.get("choices", []):
            if finishes or usages:
                raise ValueError("Stream emitted choices after its terminal finish")
            contents.append(choice.get("delta", {}).get("content", ""))
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    if not "".join(contents) or len(finishes) != 1 or finishes[0] not in {"stop", "length"} or len(usages) != 1:
        raise ValueError("Stream lacks content, unique finish or authoritative usage")
    return {"events": events, "content": "".join(contents), "usage": usages[0], "done": True}


def validate_build_inventory(build_output: Path, built: dict) -> dict:
    snapshot = json.loads((build_output / "source/manifest.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != snapshot.get("snapshot_sha256") or digest != built.get("snapshot_sha256"):
        raise ValueError("Captured source manifest integrity failure")
    if snapshot.get("source_revision") != built.get("source_revision"):
        raise ValueError("Build/source revision mismatch")
    for folder, files in (("source", snapshot["files"]), ("build-context", built["context_files_sha256"])):
        for name, expected in files.items():
            path = build_output / folder / name
            if not path.resolve().is_relative_to((build_output / folder).resolve()) or path.is_symlink():
                raise ValueError("Unsafe captured build input")
            if sha256_file(path) != expected:
                raise ValueError("Captured build input integrity failure")
    return snapshot


def owned_container_id(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Cleanup requires the exact newly created full container ID")
    return value


def run_command(image: str, name: str, base: Path, adapter: Path, cache: Path, bootstrap: Path) -> list[str]:
    for path in (base, adapter, cache, bootstrap):
        if "," in str(path) or "\n" in str(path):
            raise ValueError("Docker mount paths cannot contain comma or newline")
    args = ["docker", "run", "--detach", "--name", name, "--gpus", "all", "--publish", "127.0.0.1::8000",
            "--env", "LLM_API_KEY", "--env", "LLM_ADMIN_KEY", "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1", "--tmpfs", "/runtime:rw,uid=10001,gid=10001,mode=0700,size=256m"]
    for source, target, readonly in ((base, "/inputs/base", True), (adapter, "/inputs/adapter", True),
                                    (cache, "/runtime/huggingface", False), (bootstrap, "/bootstrap.py", True)):
        args += ["--mount", f"type=bind,source={source.resolve()},target={target}" + (",readonly" if readonly else "")]
    return args + ["--entrypoint", "python", image, "/bootstrap.py"]


def smoke(root: Path, build_output: Path, output: Path, base: Path, adapter: Path, cache: Path) -> dict:
    built = json.loads((build_output / "result.json").read_text(encoding="utf-8"))
    if built.get("status") != "built" or not re.fullmatch(r"sha256:[a-f0-9]{64}", built.get("image_id", "")):
        raise ValueError("A successful recorded build with immutable image ID is required")
    base_info, adapter_info = verify_package(base), verify_package(adapter)
    if base_info["base_model"] != adapter_info["base_model"]:
        raise ValueError("Base and adapter must use the identical base revision")
    if not cache.is_dir():
        raise ValueError("Existing recorded Hugging Face cache is required")
    output = fresh_output(output)
    api_key, admin_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    recorder = Recorder(output, (api_key, admin_key))
    env = {**os.environ, "LLM_API_KEY": api_key, "LLM_ADMIN_KEY": admin_key}
    status = {"schema_version": 1, "operation": "gpu_http_smoke", "status": "failed", "gpu_verified": False,
              "security_scope": SECURITY_SCOPE, "build_result_sha256": sha256_file(build_output / "result.json"),
              "image_id": built["image_id"], "base_artifact_id": base_info["artifact_id"],
              "adapter_artifact_id": adapter_info["artifact_id"], "synthetic_inputs_only": True,
              "limitations": ["Functional HTTP smoke; no throughput, dependency security or quality certification."]}
    container_id = None
    try:
        # Copy and validate the captured bootstrap, never a later live file.
        snapshot = validate_build_inventory(build_output, built)
        bootstrap = build_output / "source/scripts/gpu_container_entry.py"
        if sha256_file(bootstrap) != snapshot["files"]["scripts/gpu_container_entry.py"]:
            raise ValueError("Captured container bootstrap integrity failure")
        status.update(source_revision=built["source_revision"], snapshot_sha256=built["snapshot_sha256"],
                      bootstrap_sha256=sha256_file(bootstrap))
        recorder.run(["docker", "version", "--format", "{{json .}}"], label="docker-version")
        recorder.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
                     label="host-gpu")
        args = run_command(built["image_id"], "llm-gpu-verify-" + uuid.uuid4().hex, base, adapter, cache, bootstrap)
        container_id = owned_container_id(recorder.run(args, env=env, timeout=120, label="container-start"))
        status["container_id"] = container_id
        ports = json.loads(recorder.run(["docker", "inspect", container_id, "--format",
                                       "{{json .NetworkSettings.Ports}}"], label="ports"))
        bindings = ports["8000/tcp"]
        if len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1":
            raise ValueError("Gateway must publish only a host loopback port")
        url = "http://127.0.0.1:" + bindings[0]["HostPort"]
        auth = {"Authorization": "Bearer " + api_key}
        admin = {"Authorization": "Bearer " + admin_key}
        facts: dict = {"url": url}
        started = time.monotonic()
        with httpx.Client(base_url=url, timeout=180, trust_env=False) as client:
            while True:
                try:
                    response = client.get("/health/ready")
                    if response.status_code == 200:
                        facts["ready"] = response.json()
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() - started > 300:
                    raise TimeoutError("Container did not become ready within 300 seconds")
                time.sleep(1)
            facts["startup_seconds"] = time.monotonic() - started
            live = client.get("/health/live")
            live.raise_for_status()
            facts["live"] = live.json()
            for label, route, headers in (("unauthenticated_models", "/v1/models", {}),
                    ("unauthenticated_admin", "/admin/status", {}), ("inference_key_admin", "/admin/status", auth)):
                response = client.get(route, headers=headers)
                facts[label] = response.status_code
                if response.status_code not in {401, 403}:
                    raise ValueError(f"Access boundary failed: {label}")
            response = client.get("/v1/models", headers=auth)
            response.raise_for_status()
            facts["models"] = response.json()
            if {base_info["artifact_id"], adapter_info["artifact_id"]} - {item["id"] for item in facts["models"]["data"]}:
                raise ValueError("Both verified models must be discoverable")

            def complete(model: str, max_tokens: int = 32) -> dict:
                response = client.post("/v1/chat/completions", headers=auth, json={"model": model,
                    "messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens, "temperature": 0})
                response.raise_for_status()
                value = response.json()
                validate_completion(value, model, base_info["base_model"]["revision"])
                return value

            facts["base_completion"] = complete(base_info["artifact_id"])
            data = []
            with client.stream("POST", "/v1/chat/completions", headers=auth, json={"model": base_info["artifact_id"],
                    "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 32, "temperature": 0,
                    "stream": True, "stream_options": {"include_usage": True}}) as response:
                response.raise_for_status()
                data = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
            facts["stream_completion"] = validate_sse(data, base_info["artifact_id"], base_info["base_model"]["revision"])
            response = client.post("/admin/promote", headers=admin, json={"artifact_id": adapter_info["artifact_id"]})
            response.raise_for_status()
            facts["promotion"] = response.json()
            facts["adapter_completion"] = complete(adapter_info["artifact_id"])
            response = client.post("/admin/rollback", headers=admin)
            response.raise_for_status()
            facts["rollback"] = response.json()
            facts["rollback_completion"] = complete(base_info["artifact_id"])
            cancelled_after_content = False
            with client.stream("POST", "/v1/chat/completions", headers=auth, json={"model": base_info["artifact_id"],
                    "messages": [{"role": "user", "content": "List the integers from one to one thousand."}],
                    "max_tokens": 128, "temperature": 0, "stream": True}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        event = json.loads(line[6:])
                        if any(item.get("delta", {}).get("content") for item in event.get("choices", [])):
                            cancelled_after_content = True
                            break
            if not cancelled_after_content:
                raise ValueError("Disconnect test never observed generated content")
            deadline = time.monotonic() + 30
            while True:
                response = client.get("/admin/status", headers=admin)
                response.raise_for_status()
                after = response.json()
                if after["engine"].get("active_requests") == 0:
                    facts["after_disconnect"] = after
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Disconnected request did not release its backend worker")
                time.sleep(.2)
            facts["disconnect_after_content"] = cancelled_after_content
            facts["recovery_completion"] = complete(base_info["artifact_id"])
            response = client.get("/metrics", headers=auth)
            response.raise_for_status()
            metrics = response.text
            (output / "metrics.txt").write_text(metrics, encoding="utf-8")
            observed = re.search(r"^llm_gpu_memory_bytes ([^\n]+)$", metrics, re.MULTILINE)
            if observed is None or not math.isfinite(float(observed[1])) or float(observed[1]) <= 0:
                raise ValueError("Loaded gateway did not report positive finite CUDA allocation")
            if "llm_active_requests 0.0" not in metrics or "llm_requests_total" not in metrics:
                raise ValueError("Gateway request/release metrics are missing")
            facts["loaded_cuda_allocated_bytes"] = float(observed[1])
        proof = json.loads(recorder.run(["docker", "exec", container_id, "cat", "/runtime/container-proof.json"],
                                       label="container-proof"))
        if proof["uid"] == 0 or proof["cuda_probe_allocated_bytes"] <= 0 or proof["cuda_probe_sum"] != 1024:
            raise ValueError("Container nonroot/CUDA proof failed")
        installed_sources = {name: digest for name, digest in snapshot["files"].items()
                             if name.startswith("src/llm_platform/")}
        if proof.get("installed_source_sha256") != installed_sources:
            raise ValueError("Installed container source differs from captured source")
        write_json(output / "container-proof.json", proof)
        write_json(output / "http-smoke.json", facts)
        # Mounted inputs must still verify; this is independent of read-only args.
        if verify_package(base) != base_info or verify_package(adapter) != adapter_info:
            raise ValueError("Input package integrity changed")
        status.update(status="passed", gpu_verified=True)
        return status
    except Exception as exc:
        status.update(error_type=type(exc).__name__, error=recorder.redact(str(exc)))
        raise
    finally:
        if container_id is not None:
            # Only this exact ID returned by our docker run is ever stopped.
            try:
                recorder.run(["docker", "logs", container_id], check=False, label="container-logs")
                recorder.run(["docker", "stop", "--time", "30", container_id], check=False, label="container-stop")
                state = recorder.run(["docker", "inspect", container_id, "--format", "{{json .State}}"],
                                     check=False, label="container-shutdown")
                if state:
                    status["shutdown"] = json.loads(state)
                if not state or status["shutdown"].get("Running") or status["shutdown"].get("ExitCode") != 0:
                    status.update(status="failed", gpu_verified=False, shutdown_error="Gateway did not shut down cleanly")
            except Exception as exc:
                status.update(status="failed", gpu_verified=False, shutdown_error=recorder.redact(str(exc)))
            finally:
                try:
                    recorder.run(["docker", "rm", container_id], check=False, label="container-remove")
                except Exception as exc:
                    status.update(status="failed", gpu_verified=False, cleanup_error=recorder.redact(str(exc)))
        write_json(output / "result.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "smoke"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-output", type=Path)
    parser.add_argument("--base", type=Path, default=ROOT / "runtime/quality-002/base-package")
    parser.add_argument("--adapter", type=Path, default=ROOT / "evidence/adapter-package")
    parser.add_argument("--cache", type=Path, default=ROOT / "runtime/huggingface")
    parser.add_argument("--exclusive-gpu-window", action="store_true", help="Operator confirms exclusive GPU ownership")
    args = parser.parse_args()
    if args.operation == "build":
        result = build(ROOT, args.output)
    else:
        if not args.exclusive_gpu_window or args.build_output is None:
            parser.error("smoke requires --build-output and --exclusive-gpu-window")
        result = smoke(ROOT, args.build_output.resolve(), args.output, args.base.resolve(), args.adapter.resolve(),
                       args.cache.resolve())
    print(json.dumps(result, indent=2))
    if result["status"] not in {"built", "passed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
