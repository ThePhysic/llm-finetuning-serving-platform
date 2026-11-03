"""Offline controls for the historical engine's localhost exposure and cache risk."""
import json
import sys

import pytest

from llm_platform.backends import launch_vllm


def manifest():
    return {"kind": "base", "artifact_id": "a" * 64,
            "base_model": {"id": "trusted/model", "revision": "b" * 40},
            "tokenizer": {"path": "tokenizer", "revision": "b" * 40,
                          "chat_template_sha256": "c" * 64},
            "runtime": {"backend": "vllm", "dtype": "bfloat16", "quantization": "none"}}


@pytest.mark.parametrize("key", [None, "", "contains whitespace", "non-ascii-\u00e9"])
def test_launcher_requires_valid_environment_worker_key_before_loading(monkeypatch, tmp_path, key):
    monkeypatch.setattr(sys, "argv", ["launcher", "--package", str(tmp_path),
                                     "--identity", str(tmp_path / "identity.json")])
    if key is None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("VLLM_API_KEY", key)
    monkeypatch.setattr(launch_vllm, "verify_package", lambda path: pytest.fail("must reject before package loading"))
    with pytest.raises(SystemExit) as error:
        launch_vllm.main()
    assert error.value.code == 2
    assert not (tmp_path / "identity.json").exists()


def test_launcher_disables_cors_and_keeps_worker_key_out_of_command_and_identity(monkeypatch, tmp_path):
    key = "ephemeral-test-key-unique-to-this-fixture"
    monkeypatch.setenv("VLLM_API_KEY", key)
    identity = tmp_path / "identity.json"
    monkeypatch.setattr(sys, "argv", ["launcher", "--package", str(tmp_path), "--identity", str(identity)])
    monkeypatch.setattr(launch_vllm, "verify_package", lambda path: manifest())
    monkeypatch.setattr(launch_vllm.importlib.metadata, "version", lambda name: "0.8.5.post1")
    executed = []
    monkeypatch.setattr(launch_vllm.os, "execv", lambda executable, command: executed.append(command))
    launch_vllm.main()
    command = executed[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert json.loads(command[command.index("--allowed-origins") + 1]) == []
    assert key not in json.dumps(command)
    assert key not in identity.read_text()
    assert json.loads(identity.read_text())["allowed_origins"] == []


def test_prefix_cache_requires_dedicated_synthetic_workload(tmp_path):
    with pytest.raises(ValueError, match="isolated synthetic"):
        launch_vllm.launch_command(tmp_path, manifest(), prefix_caching=True)
    command, identity = launch_vllm.launch_command(tmp_path, manifest(), prefix_caching=True,
                                                 isolated_prefix_cache=True)
    assert "--enable-prefix-caching" in command
    assert identity["isolated_prefix_cache"] is True
