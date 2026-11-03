"""Offline end-to-end CLI checks; none is evidence of model training or GPU speed."""
import json
from pathlib import Path
import subprocess
import sys

from typer.testing import CliRunner

from llm_platform.cli import app


runner = CliRunner()


def invoke_ok(arguments):
    result = runner.invoke(app, [str(value) for value in arguments])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return json.loads(result.stdout)


def snapshot(directory: Path):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_fresh_dataset_cli_roundtrip_and_existing_output_preserved(tmp_path):
    corpus = tmp_path / "incidents.jsonl"
    prepared = tmp_path / "dataset"
    generated = invoke_ok(["data", "generate", corpus])
    assert generated["records"] > 100
    manifest = invoke_ok(["data", "prepare", corpus, prepared, "--seed", 42])
    verified = invoke_ok(["data", "verify", prepared])
    assert verified == manifest
    assert manifest["files"]["train.jsonl"]["count"] > 0
    assert manifest["files"]["validation.jsonl"]["count"] > 0
    assert manifest["files"]["test.jsonl"]["count"] > 0
    before = snapshot(prepared)
    corpus_before = corpus.read_bytes()
    assert runner.invoke(app, ["data", "generate", str(corpus)]).exit_code != 0
    assert corpus.read_bytes() == corpus_before
    assert runner.invoke(app, ["data", "prepare", str(corpus), str(prepared), "--seed", "7"]).exit_code != 0
    assert snapshot(prepared) == before


def test_cli_rejects_tampered_dataset_and_preserves_existing_experiments(tmp_path):
    corpus, prepared = tmp_path / "incidents.jsonl", tmp_path / "dataset"
    invoke_ok(["data", "generate", corpus])
    invoke_ok(["data", "prepare", corpus, prepared])
    train = prepared / "train.jsonl"
    train.write_bytes(train.read_bytes() + b"\n")
    result = runner.invoke(app, ["data", "verify", str(prepared)])
    assert result.exit_code != 0
    output = tmp_path / "existing-run"
    output.mkdir()
    (output / "important-results.json").write_text('{"keep": true}')
    before = snapshot(output)
    config = tmp_path / "config.json"
    config.write_text("{}")
    result = runner.invoke(app, ["train", str(config), str(prepared), str(output)])
    assert result.exit_code != 0
    assert snapshot(output) == before


def test_invalid_input_and_artifact_do_not_create_valid_release(tmp_path):
    bad = tmp_path / "malformed.jsonl"
    bad.write_text('{"instruction": "missing target and family"}\n')
    dataset = tmp_path / "bad-dataset"
    assert runner.invoke(app, ["data", "prepare", str(bad), str(dataset)]).exit_code != 0
    assert not (dataset / "manifest.json").exists()
    registry = tmp_path / "registry"
    invalid = tmp_path / "bad-package"
    invalid.mkdir()
    (invalid / "manifest.json").write_text('{"artifact_id": "not-a-valid-id"}')
    result = runner.invoke(app, ["registry", "register", str(invalid), "--registry", str(registry)])
    assert result.exit_code != 0
    assert not (registry / "deployment.json").exists()
    assert invoke_ok(["registry", "list", "--registry", registry]) == []


def test_help_doctor_and_dataset_commands_do_not_import_training_libraries(tmp_path):
    # A separate interpreter gives a reliable import boundary even when the full
    # test suite also runs model-library tests in the parent process.
    script = """
import builtins, json, sys
real_import = builtins.__import__
blocked = {'torch', 'transformers', 'peft', 'vllm', 'accelerate', 'bitsandbytes'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Lightweight command imported a GPU training library: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from typer.testing import CliRunner
from llm_platform.cli import app
runner = CliRunner()
for args in [['--help'], ['doctor'], ['data', 'generate', sys.argv[1]], ['data', 'prepare', sys.argv[1], sys.argv[2]], ['data', 'verify', sys.argv[2]]]:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, str(args) + result.output + repr(result.exception)
print(json.dumps({'status': 'ok'}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "raw.jsonl"),
                             str(tmp_path / "prepared")], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "ok"


def test_nonlocal_cli_refuses_start_without_separate_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_ADMIN_KEY", raising=False)
    import uvicorn
    called = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: called.append((args, kwargs)))
    result = runner.invoke(app, ["serve", "--registry", str(tmp_path / "registry"), "--host", "0.0.0.0"])
    assert result.exit_code != 0
    assert not called
    assert not (tmp_path / "registry").exists()
    monkeypatch.setenv("LLM_API_KEY", "api-fixture")
    monkeypatch.setenv("LLM_ADMIN_KEY", "api-fixture")
    result = runner.invoke(app, ["serve", "--registry", str(tmp_path / "registry"), "--host", "0.0.0.0"])
    assert result.exit_code != 0
    assert not called


def test_evaluation_rejects_invalid_variant_before_loading_model(tmp_path):
    result = runner.invoke(app, ["evaluate", str(tmp_path / "missing-package"),
                                 str(tmp_path / "missing-dataset"), str(tmp_path / "output"),
                                 "--variant", "unsupported"])
    assert result.exit_code != 0
    assert "Unknown prompt variant" in result.output
    assert not (tmp_path / "output").exists()
