"""Verify a package and exec vLLM with an auditable immutable runtime identity.

Run inside a Linux vLLM environment. Keep the engine on loopback behind the
gateway; the API gateway owns public authentication and administrative control.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from llm_platform.registry import verify_package


def launch_command(package: Path, manifest: dict[str, Any], port: int = 8001,
                   max_model_len: int = 2048, gpu_memory_utilization: float = 0.60,
                   max_num_seqs: int = 8, prefix_caching: bool = False,
                   enforce_eager: bool = True) -> tuple[list[str], dict[str, Any]]:
    if manifest["runtime"]["backend"] != "vllm":
        raise ValueError("Create a vLLM runtime package before launching")
    if not 1 <= port <= 65535 or max_model_len < 16 or max_num_seqs < 1:
        raise ValueError("Invalid server limits")
    if not 0.1 <= gpu_memory_utilization <= 0.95:
        raise ValueError("GPU memory utilization must be between 0.1 and 0.95")
    package = package.resolve()
    model_name = manifest["artifact_id"]
    served_base = model_name if manifest["kind"] == "base" else "base-" + manifest["base_model"]["revision"]
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--host", "127.0.0.1",
               "--port", str(port), "--model", manifest["base_model"]["id"],
               "--revision", manifest["base_model"]["revision"],
               "--tokenizer", str(package / manifest["tokenizer"]["path"]),
               "--tokenizer-revision", manifest["tokenizer"]["revision"],
               "--served-model-name", served_base, "--dtype", manifest["runtime"]["dtype"],
               "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization),
               "--max-num-seqs", str(max_num_seqs), "--generation-config", "vllm",
               "--disable-log-requests", "--disable-uvicorn-access-log", "--uvicorn-log-level", "warning",
               "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching"]
    if enforce_eager:
        command.append("--enforce-eager")
    quantization = manifest["runtime"].get("quantization", "none")
    if quantization != "none":
        command.extend(["--quantization", {"bnb4": "bitsandbytes"}.get(quantization, quantization)])
    if manifest["kind"] == "adapter":
        rank = manifest["adapter"]["config"].get("r", 16)
        command.extend(["--enable-lora", "--max-lora-rank", str(rank),
                        "--lora-modules", model_name + "=" + str(package / manifest["adapter"]["path"])])
    identity = {"artifact_id": model_name, "model_name": model_name,
                "base_revision": manifest["base_model"]["revision"],
                "tokenizer_revision": manifest["tokenizer"]["revision"],
                "chat_template_sha256": manifest["tokenizer"]["chat_template_sha256"],
                "quantization": quantization, "dtype": manifest["runtime"]["dtype"],
                "prefix_caching": prefix_caching, "enforce_eager": enforce_eager,
                "max_model_len": max_model_len, "max_num_seqs": max_num_seqs,
                "gpu_memory_utilization": gpu_memory_utilization, "command": command}
    return command, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    manifest = verify_package(args.package)
    command, identity = launch_command(args.package, manifest, args.port, args.max_model_len,
                                       args.gpu_memory_utilization, args.max_num_seqs,
                                       args.prefix_caching, not args.cuda_graphs)
    identity.update(pid=os.getpid(), launch_unix_seconds=time.time(),
                    vllm_version=importlib.metadata.version("vllm"))
    args.identity.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.identity.with_suffix(".tmp")
    temporary.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.identity)
    # No shell interpolation. No API keys are included in command or identity.
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
