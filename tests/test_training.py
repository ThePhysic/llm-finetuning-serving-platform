"""CPU checks validate data/masking; they are not evidence of GPU adaptation."""
import json

import pytest

from llm_platform.training import TrainConfig, completion_example, read_records


@pytest.mark.parametrize("overrides", [
    {"max_steps": 0}, {"rank": True}, {"dropout": 1}, {"dtype": "int4"},
    {"device": "cpu", "dtype": "float16"}, {"device": "cpu", "dtype": "float32", "qlora": True},
    {"learning_rate": float("nan")}, {"target_modules": ()},
])
def test_bad_training_settings_rejected(overrides):
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_unknown_configuration_is_not_silently_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"gradient_accumlation_steps": 8}))
    with pytest.raises(ValueError, match="Unknown training"):
        TrainConfig.from_json(path)


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
