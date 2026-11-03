"""CPU contract checks; these never start Docker or claim GPU operation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_gpu_container.py"
spec = importlib.util.spec_from_file_location("verify_gpu_container", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
MODEL, REVISION = "a" * 64, "b" * 40


def event(choices, **extra):
    return {"model": MODEL, "system_fingerprint": MODEL,
            "metadata": {"artifact_id": MODEL, "base_model_revision": REVISION}, "choices": choices, **extra}


def valid_stream():
    return [event([{"delta": {"role": "assistant"}, "finish_reason": None}]),
            event([{"delta": {"content": "Actual generated text"}, "finish_reason": None}]),
            event([{"delta": {}, "finish_reason": "stop"}]),
            event([], usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})]


def sse(events):
    return [json.dumps(value) for value in events] + ["[DONE]"]


def test_stream_requires_real_content_terminal_usage_and_identity():
    result = harness.validate_sse(sse(valid_stream()), MODEL, REVISION)
    assert result["content"] == "Actual generated text"
    assert result["usage"]["completion_tokens"] == 3
    assert result["done"] is True


@pytest.mark.parametrize("mutation", ["no_finish", "no_usage", "no_content", "error", "wrong_model",
                                      "wrong_revision", "false_usage", "post_finish_text", "early_usage"])
def test_rejects_partial_or_falsely_attributed_streams(mutation):
    events = valid_stream()
    if mutation == "no_finish":
        events.pop(2)
    elif mutation == "no_usage":
        events.pop()
    elif mutation == "no_content":
        events[1]["choices"][0]["delta"]["content"] = ""
    elif mutation == "error":
        events[1] = {"error": {"code": "engine_error"}}
    elif mutation == "wrong_model":
        events[0]["model"] = "different"
    elif mutation == "wrong_revision":
        events[1]["metadata"]["base_model_revision"] = "different"
    elif mutation == "false_usage":
        events[-1]["usage"]["completion_tokens"] = True
    elif mutation == "post_finish_text":
        events.insert(3, copy.deepcopy(events[1]))
    elif mutation == "early_usage":
        events[2], events[3] = events[3], events[2]
    with pytest.raises(ValueError):
        harness.validate_sse(sse(events), MODEL, REVISION)


@pytest.mark.parametrize("data", [[], ["[DONE]"], ["[DONE]", "[DONE]"]])
def test_done_alone_never_proves_stream_generation(data):
    with pytest.raises(ValueError):
        harness.validate_sse(data, MODEL, REVISION)


def test_mounts_are_readonly_and_secrets_are_environment_only(tmp_path):
    args = harness.run_command("sha256:" + "c" * 64, "owned", tmp_path / "base", tmp_path / "adapter",
                               tmp_path / "cache", tmp_path / "entry.py")
    mounts = [args[index + 1] for index, value in enumerate(args) if value == "--mount"]
    assert all("readonly" in mount.split(",") for mount in mounts
               if "target=/inputs/" in mount or "target=/bootstrap.py" in mount)
    assert "readonly" not in next(mount for mount in mounts if "target=/runtime/huggingface" in mount).split(",")
    assert "127.0.0.1::8000" in args
    assert args[args.index("--env") + 1] == "LLM_API_KEY"
    assert all("LLM_API_KEY=" not in item and "LLM_ADMIN_KEY=" not in item for item in args)
    assert args[-3:] == ["python", "sha256:" + "c" * 64, "/bootstrap.py"]


@pytest.mark.parametrize("value", ["", "name", "abcdef", "--all", "a" * 64 + "\n", "A" * 64])
def test_cleanup_refuses_names_options_or_truncated_ids(value):
    with pytest.raises(ValueError):
        harness.owned_container_id(value)
    assert harness.owned_container_id("a" * 64) == "a" * 64


def test_recorder_redacts_process_output_and_argv(monkeypatch, tmp_path):
    secret = "ephemeral-test-value"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, secret, secret))
    recorder = harness.Recorder(tmp_path, (secret,))
    recorder.run(["docker", "test", secret])
    saved = (tmp_path / "commands.json").read_text()
    assert secret not in saved
    assert "[REDACTED]" in saved


def test_preserved_build_inventory_rejects_tampering(tmp_path):
    for folder in ("source", "build-context"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "module.py").write_bytes(b"original\n")
    digest = hashlib.sha256(b"original\n").hexdigest()
    snapshot = {"source_revision": "a" * 40, "files": {"module.py": digest}}
    snapshot["snapshot_sha256"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (tmp_path / "source/manifest.json").write_text(json.dumps(snapshot))
    built = {"source_revision": "a" * 40, "snapshot_sha256": snapshot["snapshot_sha256"],
             "context_files_sha256": {"module.py": digest}}
    assert harness.validate_build_inventory(tmp_path, built) == snapshot
    (tmp_path / "build-context/module.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="integrity"):
        harness.validate_build_inventory(tmp_path, built)


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "evidence"
    harness.fresh_output(output)
    marker = output / "result.json"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        harness.fresh_output(output)
    assert marker.read_text() == "original"
