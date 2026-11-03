"""Nonroot bootstrap for the historical GPU-container HTTP verification only.

The actual gateway remains the installed ``llm-platform serve`` command. Model
packages are mounted read-only; credentials are inherited from the environment.
"""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform


def main() -> None:
    if os.getuid() == 0:
        raise RuntimeError("GPU verification requires a nonroot container user")
    import torch
    import llm_platform
    from llm_platform.registry import Registry, verify_package

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the container")
    probe = torch.ones(1024, device="cuda")
    torch.cuda.synchronize()
    proof = {"uid": os.getuid(), "gid": os.getgid(), "python": platform.python_version(),
             "versions": {name: importlib.metadata.version(name) for name in
                          ("torch", "transformers", "peft", "accelerate", "safetensors")},
             "installed_distributions": dict(sorted((distribution.metadata["Name"], distribution.version)
                                                      for distribution in importlib.metadata.distributions())),
             "cuda_runtime": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0),
             "cuda_probe_allocated_bytes": torch.cuda.memory_allocated(),
             "cuda_probe_sum": probe.sum().item(),
             "security_scope": "historical dependencies; trusted local artifacts only"}
    installed = Path(llm_platform.__file__).resolve().parent
    proof["installed_source_sha256"] = {
        "src/llm_platform/" + path.relative_to(installed).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(installed.rglob("*.py"))}
    del probe
    torch.cuda.empty_cache()
    registry = Registry(Path("/runtime/registry"))
    packages = {name: Path("/inputs") / name for name in ("base", "adapter")}
    manifests = {name: verify_package(path) for name, path in packages.items()}
    if manifests["base"]["base_model"] != manifests["adapter"]["base_model"]:
        raise RuntimeError("Base and adapter package revisions differ")
    proof["registered"] = {name: registry.register(path) for name, path in packages.items()}
    Path("/runtime/container-proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    # Exec releases the probe process's CUDA context before the real CLI loads.
    os.execvp("llm-platform", ["llm-platform", "serve", "--registry", "/runtime/registry",
              "--model", manifests["base"]["artifact_id"], "--host", "0.0.0.0", "--port", "8000",
              "--concurrency", "1", "--max-queue", "2", "--max-input-tokens", "1024",
              "--max-output-tokens", "128", "--timeout", "120"])


if __name__ == "__main__":
    main()
