"""CPU checks validate data/masking; they are not evidence of GPU adaptation."""
import json
import builtins
import importlib.util
import os
from types import SimpleNamespace

import pytest

from llm_platform.training import TrainConfig, completion_example, read_records


@pytest.mark.parametrize("overrides", [
    {"max_steps": 0}, {"rank": True}, {"dropout": 1}, {"dtype": "int4"},
    {"device": "cpu", "dtype": "float16"}, {"device": "cpu", "dtype": "float32", "qlora": True},
    {"learning_rate": float("nan")}, {"target_modules": ()},
    {"max_grad_norm": -1}, {"max_grad_norm": 0}, {"max_grad_norm": float("nan")},
    {"weight_decay": -1}, {"weight_decay": float("nan")}, {"weight_decay": float("inf")},
    {"seed": True}, {"seed": -1}, {"seed": 2**32}, {"warmup_steps": True}, {"warmup_steps": 1.5},
    {"target_modules": "q_proj"}, {"target_modules": (" ",)},
    {"qlora": "false"}, {"gradient_checkpointing": 1}, {"learning_rate": True},
])
def test_bad_training_settings_rejected(overrides):
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_unknown_configuration_is_not_silently_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"gradient_accumlation_steps": 8}))
    with pytest.raises(ValueError, match="Unknown training"):
        TrainConfig.from_json(path)


@pytest.mark.parametrize("text", ['{"seed":42,"seed":7}', '{"learning_rate":1e999}', '{"target_modules":"q_proj"}'])
def test_configuration_json_cannot_bypass_strict_parsing(tmp_path, text):
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        TrainConfig.from_json(path)


def test_direct_training_reader_rejects_duplicate_record_keys_and_nonfinite_metadata(tmp_path):
    from llm_platform.synthetic import generate_records

    path = tmp_path / "train.jsonl"
    row = generate_records()[0]
    path.write_text('{"family":"injected",' + json.dumps(row)[1:] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_records(path)
    text = json.dumps(row).replace('"synthetic": true', '"synthetic":1e999')
    path.write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        read_records(path)


def test_training_reader_honors_an_explicit_character_budget(tmp_path):
    from llm_platform.synthetic import generate_records
    from llm_platform.data import write_jsonl

    record = generate_records()[0]
    record["messages"][0]["content"] += " " * 16500
    path = tmp_path / "long.jsonl"
    write_jsonl(path, [record])
    with pytest.raises(ValueError, match="character limit"):
        read_records(path)
    assert len(read_records(path, max_chars=20000)) == 1


def test_training_config_copies_mutable_target_module_lists():
    modules = ["q_proj"]
    config = TrainConfig(target_modules=modules)
    modules.append("v_proj")
    assert config.target_modules == ("q_proj",)


def test_heldout_and_regression_records_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    base = {"messages": [{"role": "user", "content": "x"}], "target": "x"}
    for change in ({"kind": "regression"}, {"metadata": {"split": "test"}}):
        path.write_text(json.dumps({**base, **change}) + "\n")
        with pytest.raises(ValueError):
            read_records(path)


class TemplateTokenizer:
    """Small tokenizer substitute to exercise loss-boundary validation only."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if add_generation_prompt:
            return [1, 2, 3]
        return [1, 2, 3, 4, 5, 6]


def test_loss_masks_prompt_and_supervises_completion():
    record = {"messages": [{"role": "user", "content": "Report"}], "target": {"category": "unknown"}}
    result = completion_example(TemplateTokenizer(), record, 8)
    assert result["labels"] == [-100, -100, -100, 4, 5, 6]
    assert result["input_ids"] == [1, 2, 3, 4, 5, 6]


def test_training_does_not_silently_truncate_target():
    record = {"messages": [{"role": "user", "content": "Report"}], "target": "target"}
    with pytest.raises(ValueError, match="no truncation"):
        completion_example(TemplateTokenizer(), record, 4)


def test_template_rewrite_cannot_corrupt_completion_mask():
    class RewritingTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1, 2, 3] if add_generation_prompt else [1, 8, 3, 4]

    record = {"messages": [{"role": "user", "content": "Report"}], "target": "target"}
    with pytest.raises(ValueError, match="prefix"):
        completion_example(RewritingTokenizer(), record, 8)


@pytest.mark.parametrize("completed", [False, True])
def test_nonempty_resume_output_is_refused_before_heavy_imports_or_writes(tmp_path, monkeypatch, completed):
    from llm_platform.training import train

    output = tmp_path / "old-run"
    output.mkdir()
    sentinel = output / ("result.json" if completed else "events.jsonl")
    sentinel.write_bytes(b"preserved original run\n")
    original = builtins.__import__

    def refuse_heavy(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "peft", "transformers"}:
            raise AssertionError("Output preflight must run before model dependencies are loaded")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_heavy)
    with pytest.raises(ValueError, match="fresh output directory"):
        train(TrainConfig(resume_from="old-checkpoint"), tmp_path / "train.jsonl", tmp_path / "validation.jsonl", output)
    assert sentinel.read_bytes() == b"preserved original run\n"
    assert set(output.iterdir()) == {sentinel}


@pytest.fixture
def tiny_cpu_training(monkeypatch):
    # Optional real CPU integration: no hosted model, weight download or GPU use.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if any(importlib.util.find_spec(name) is None for name in ("torch", "peft", "transformers")):
        pytest.skip("Real CPU resume integration requires the optional training dependencies")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    assert not torch.cuda.is_available()
    torch.manual_seed(19)
    model_config = GPT2Config(vocab_size=32, n_positions=32, n_ctx=32, n_embd=8, n_layer=1, n_head=1,
                              bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model_config._name_or_path = "Audit/TinyGPT2"
    template = GPT2LMHeadModel(model_config)
    weights = {name: value.detach().clone() for name, value in template.state_dict().items()}
    loaded = []

    class Tokenizer:
        chat_template = "audit-cpu-template-v1"
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            return [1, 3, 4] if add_generation_prompt else [1, 3, 4, 5, 6, 2]

        def save_pretrained(self, destination):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "tokenizer_config.json").write_text(json.dumps({"chat_template": self.chat_template}))

    tokenizer = Tokenizer()

    def model_factory(*args, **kwargs):
        loaded.append(True)
        model = GPT2LMHeadModel(model_config)
        model.load_state_dict(weights)
        return model

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: SimpleNamespace(sha="a" * 40))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", model_factory)
    return tokenizer, loaded


def tiny_run_config(**changes):
    values = dict(model_id="Audit/TinyGPT2", revision="a" * 40, max_steps=2, batch_size=1,
                  gradient_accumulation_steps=1, max_seq_length=32, rank=2, alpha=4, dropout=0.2,
                  target_modules=("c_attn",), dtype="float32", device="cpu", gradient_checkpointing=False,
                  validation_every=1, checkpoint_every=1, warmup_steps=0, dataset_id="cpu-audit-fixture")
    return TrainConfig(**{**values, **changes})


def tiny_data(tmp_path):
    from llm_platform.synthetic import generate_records
    from llm_platform.data import write_jsonl

    rows = generate_records()
    train_path, validation_path = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    write_jsonl(train_path, [rows[0]])
    write_jsonl(validation_path, [rows[40]])
    return train_path, validation_path


@pytest.mark.integration
def test_real_cpu_resume_preserves_weights_and_complete_raw_history(tmp_path, tiny_cpu_training):
    from safetensors.torch import load_file
    from llm_platform.training import train

    _, loaded = tiny_cpu_training
    train_path, validation_path = tiny_data(tmp_path)
    full = tmp_path / "full"
    train(tiny_run_config(), train_path, validation_path, full)
    checkpoint = full / "checkpoints" / "step-000001"
    resumed = tmp_path / "resumed"
    train(tiny_run_config(resume_from=str(checkpoint)), train_path, validation_path, resumed)
    original_weights = load_file(str(full / "package/adapter/adapter_model.safetensors"))
    resumed_weights = load_file(str(resumed / "package/adapter/adapter_model.safetensors"))
    assert all(original_weights[key].equal(resumed_weights[key]) for key in original_weights)
    metrics = json.loads((resumed / "package/training_metrics.json").read_text())
    events = [json.loads(line) for line in (resumed / "events.jsonl").read_text().splitlines()]
    assert events == metrics["history"]
    assert [event["step"] for event in events] == [1, 2]
    before = {path.relative_to(full): path.read_bytes() for path in full.rglob("*") if path.is_file()}
    model_loads = len(loaded)
    with pytest.raises(ValueError, match="Completed training export"):
        train(tiny_run_config(max_steps=3, resume_from=str(full / "checkpoints/step-000002")), train_path, validation_path, full)
    assert len(loaded) == model_loads
    assert {path.relative_to(full): path.read_bytes() for path in full.rglob("*") if path.is_file()} == before


@pytest.mark.integration
def test_named_chat_templates_are_rejected_before_weights_or_tokenizer_saves(tmp_path, tiny_cpu_training, monkeypatch):
    from llm_platform.training import train

    tokenizer, loaded = tiny_cpu_training
    tokenizer.chat_template = {"../../outside": "unsupported named template"}
    monkeypatch.setattr(tokenizer, "save_pretrained", lambda *a, **k: pytest.fail("Unsupported template must never be saved"))
    train_path, validation_path = tiny_data(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="nonempty string chat template"):
        train(tiny_run_config(), train_path, validation_path, output)
    assert not loaded
    assert not (output / "export-source").exists()
    assert not (tmp_path / "outside").exists()


@pytest.mark.integration
def test_real_cpu_training_honors_verified_manifest_character_budget(tmp_path, tiny_cpu_training):
    from llm_platform.data import prepare_dataset, write_jsonl
    from llm_platform.synthetic import generate_records
    from llm_platform.training import train

    rows = generate_records()
    for row in rows:
        if row["kind"] == "triage":
            row["messages"][0]["content"] += " " * 16500
    source = tmp_path / "long-source.jsonl"
    write_jsonl(source, rows)
    frozen = tmp_path / "frozen"
    manifest = prepare_dataset(source, frozen, max_chars=20000)
    result = train(tiny_run_config(max_steps=1, dataset_id=manifest["dataset_id"]), frozen / "train.jsonl",
                   frozen / "validation.jsonl", tmp_path / "trained")
    assert result["metrics"]["completed_optimizer_steps"] == 1
    assert result["metrics"]["gpu_evidence"] is False
