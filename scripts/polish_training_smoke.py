"""Compatibility-only GPU smoke for an isolated candidate dependency environment.

Requires an explicitly assigned exclusive GPU window. This never overwrites the
historical adapter/evaluation/benchmark artifacts or claims a new quality result.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def api_smoke(base_package: Path, adapter_package: Path, candidate_package: Path, work: Path):
    import httpx
    import uvicorn
    from llm_platform.registry import Registry
    from llm_platform.serving import Settings, create_app

    registry = Registry(work / "registry")
    base_id, adapter_id = registry.register(base_package), registry.register(adapter_package)
    candidate_id = registry.register(candidate_package)
    api_key, admin_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    app = create_app(Settings(initial_model=base_id, api_key=api_key, admin_key=admin_key), registry)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_level="error"))
    task = asyncio.create_task(server.serve())
    result = {"transport": "genuine loopback TCP HTTP to Uvicorn; real candidate PyTorch model execution",
              "base_artifact_id": base_id, "adapter_artifact_id": adapter_id}
    try:
        async with asyncio.timeout(120):
            while not server.started:
                if task.done():
                    task.result()
                    raise RuntimeError("Candidate gateway exited before startup")
                await asyncio.sleep(0.05)
        url = f"http://127.0.0.1:{port}"
        messages = [{"role": "user", "content": "Warehouse reports no stock, inventory shows twelve units, and the latest inventory synchronization timed out. Give a short factual assessment."}]
        async with httpx.AsyncClient(base_url=url, timeout=120, headers={"Authorization": f"Bearer {api_key}"}) as client:
            async with httpx.AsyncClient(base_url=url, timeout=120, headers={"Authorization": f"Bearer {admin_key}"}) as admin:
                ready = await client.get("/health/ready")
                assert ready.status_code == 200 and ready.json()["model"] == base_id
                async with httpx.AsyncClient(base_url=url, timeout=10) as anonymous:
                    result["unauthenticated_models_status"] = (await anonymous.get("/v1/models")).status_code
                assert result["unauthenticated_models_status"] == 401
                result["inference_key_admin_status"] = (await client.get("/admin/status")).status_code
                assert result["inference_key_admin_status"] == 401
                completion = await client.post("/v1/chat/completions", json={"model": base_id, "messages": messages, "max_tokens": 16})
                completion.raise_for_status()
                result["base_completion"] = completion.json()
                assert completion.json()["model"] == base_id and completion.json()["usage"]["completion_tokens"] > 0
                promotion = await admin.post("/admin/promote", json={"artifact_id": adapter_id})
                promotion.raise_for_status()
                assert promotion.json()["current"] == adapter_id and not promotion.json()["cleanup_pending"]
                state = (await admin.get("/admin/status")).json()
                result["adapter_engine_health"] = state["engine"]
                assert state["engine"]["adapter_compute_dtype"] == "bfloat16"
                output, terminal, done, usage = [], False, False, None
                async with client.stream("POST", "/v1/chat/completions", json={"model": adapter_id, "messages": messages,
                    "max_tokens": 24, "stream": True, "stream_options": {"include_usage": True}}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            done = True
                            break
                        event = json.loads(payload)
                        assert "error" not in event and event["model"] == adapter_id
                        usage = event.get("usage") or usage
                        for choice in event.get("choices", []):
                            output.append(choice.get("delta", {}).get("content", ""))
                            terminal |= choice.get("finish_reason") in {"stop", "length"}
                assert done and terminal and "".join(output) and usage
                result["adapter_stream"] = {"content": "".join(output), "usage": usage, "terminal": terminal, "done": done}
                async with client.stream("POST", "/v1/chat/completions", json={"model": adapter_id,
                    "messages": [{"role": "user", "content": "Count the integers from one to five hundred, one per line."}],
                    "max_tokens": 256, "stream": True}) as response:
                    response.raise_for_status()
                    received = False
                    async for line in response.aiter_lines():
                        if line.startswith("data:") and line[5:].strip() != "[DONE]":
                            event = json.loads(line[5:])
                            if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                                active_state = (await admin.get("/admin/status")).json()["engine"]
                                result["active_requests_before_disconnect"] = active_state["active_requests"]
                                assert active_state["active_requests"] > 0
                                received = True
                                break
                    assert received
                async with asyncio.timeout(30):
                    while True:
                        state = (await admin.get("/admin/status")).json()
                        if state["engine"].get("active_requests") == 0 and app.state.manager.leases == 0:
                            break
                        await asyncio.sleep(0.05)
                result["disconnect_released_engine_and_lease"] = True
                rollback = await admin.post("/admin/rollback")
                rollback.raise_for_status()
                assert rollback.json()["current"] == base_id and not rollback.json()["cleanup_pending"]
                result["rollback"] = rollback.json()
                result["ready_after_rollback"] = (await client.get("/health/ready")).json()
                assert result["ready_after_rollback"]["model"] == base_id
                promoted = await admin.post("/admin/promote", json={"artifact_id": candidate_id})
                promoted.raise_for_status()
                assert promoted.json()["current"] == candidate_id and not promoted.json()["cleanup_pending"]
                reloaded = await client.post("/v1/chat/completions", json={"model": candidate_id, "messages": messages,
                                                                           "max_tokens": 16})
                reloaded.raise_for_status()
                assert reloaded.json()["model"] == candidate_id and reloaded.json()["usage"]["completion_tokens"] > 0
                result["new_candidate_adapter_reload_completion"] = reloaded.json()
                restored = await admin.post("/admin/rollback")
                restored.raise_for_status()
                assert restored.json()["current"] == base_id
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 45)
    assert app.state.manager.current is None and not app.state.manager.cleanup_status()
    result["graceful_shutdown"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-package", type=Path, default=Path("runtime/quality-002/base-package"))
    parser.add_argument("--adapter-package", type=Path, default=Path("evidence/adapter-package"))
    args = parser.parse_args()
    if not args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise ValueError("An explicitly assigned GPU window, --gpu and visible CUDA device are required before importing Torch")
    if args.work.exists() or args.output.exists():
        raise FileExistsError("Use fresh runtime/output paths; historical and candidate evidence cannot be overwritten")
    args.work.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    record = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "in_progress",
              "purpose": "New dependency compatibility smoke only; not a replacement quality/performance experiment",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "historical_artifacts_modified": False, "steps": []}
    write(args.output, record)
    try:
        import torch
        import torch.nn.functional as functional
        from llm_platform.data import content_hash, load_records, write_jsonl
        from llm_platform.registry import verify_package
        from llm_platform.training import TrainConfig, train
        record["versions"] = {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "peft", "accelerate", "safetensors", "huggingface-hub", "triton", "fastapi", "httpx")}
        record["cuda_runtime"] = torch.version.cuda
        assert torch.cuda.is_available()
        record["gpu"] = torch.cuda.get_device_name()
        record["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        torch.manual_seed(42)
        tensor = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        assert torch.isfinite(tensor @ tensor).all()
        query = torch.randn((1, 2, 16, 32), device="cuda", dtype=torch.bfloat16)
        assert torch.isfinite(functional.scaled_dot_product_attention(query, query, query)).all()
        torch.cuda.synchronize()
        record["steps"].append({"name": "actual_cuda_bfloat16_matmul_and_sdpa", "status": "passed"})
        del tensor, query
        base = verify_package(args.base_package)
        rows = {name: load_records(root / "data/triage-v3" / f"{name}.jsonl")[:2] for name in ("train", "validation")}
        subset = args.work / "smoke-data"
        subset.mkdir()
        for name, examples in rows.items():
            write_jsonl(subset / f"{name}.jsonl", examples)
        record["smoke_data"] = {"source": "Frozen triage-v3 train/validation only; first two rows each, no test records",
                                "selected_ids": {name: [row["id"] for row in values] for name, values in rows.items()}}
        config = TrainConfig(model_id=base["base_model"]["id"], revision=base["base_model"]["revision"],
            dataset_id="compatibility-smoke-" + content_hash(rows), experiment_id="candidate-dependency-two-step-smoke",
            max_steps=2, gradient_accumulation_steps=1, validation_every=2, checkpoint_every=1,
            dtype="bfloat16", device="cuda")
        record["train_config"] = asdict(config)
        trained = train(config, subset / "train.jsonl", subset / "validation.jsonl", args.work / "train")
        record["new_training_artifact_id"] = trained["artifact_id"]
        record["new_training_metrics"] = trained["metrics"]
        assert trained["metrics"]["completed_optimizer_steps"] == 2
        record["steps"].append({"name": "real_two_step_lora_training_and_export", "status": "passed"})
        gc.collect()
        torch.cuda.empty_cache()
        # Serving reuses the actual historical adapter and cached immutable base
        # revision. Hub metadata resolution is real and is never mocked here.
        record["http"] = asyncio.run(api_smoke(args.base_package, args.adapter_package, Path(trained["package"]), args.work))
        record["steps"].append({"name": "real_tcp_sync_sse_auth_cancel_identity_promotion_rollback", "status": "passed"})
        record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", exception_type=type(exc).__name__)
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output, record)


if __name__ == "__main__":
    main()
